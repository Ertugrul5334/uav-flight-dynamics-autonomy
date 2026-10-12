"""2D (longitudinal / vertical-plane) flight simulator.

A 3-degree-of-freedom rigid body moving in the x-z plane, sharing the
aircraft parameters and aerodynamic model of the full 6DOF simulator.

State ``[x, h, u, w, theta, q]``: horizontal position, altitude, body-axis
velocities, pitch angle and pitch rate. Inputs ``[delta_e, delta_t]``.

Runs in real time, at a chosen multiple of real time, or as fast as possible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import atmosphere
from .aerodynamics import air_data, lift_drag_coefficients, propeller_thrust
from .control.pid import PID
from .dynamics import rk4_step
from .params import AircraftParams


class Longitudinal2D:
    def __init__(self, params: AircraftParams | None = None, use_isa: bool = True):
        self.P = params or AircraftParams()
        self.use_isa = use_isa

    def derivatives(self, s: np.ndarray, delta: np.ndarray) -> np.ndarray:
        P = self.P
        _, h, u, w, theta, q = s
        de, dt = np.clip(delta, [P.delta_min[0], P.delta_min[3]], [P.delta_max[0], P.delta_max[3]])
        rho = atmosphere.density(h) if self.use_isa else atmosphere.RHO0
        ad = air_data(np.array([u, 0.0, w]), rho)
        thrust = propeller_thrust(P, ad.Va, dt, rho)
        CL, CD = lift_drag_coefficients(P, ad.alpha)
        qS = ad.qbar * P.S
        c2V = P.c / (2 * ad.Va) if ad.Va > 1e-3 else 0.0
        lift = qS * (CL + P.C_L_q * c2V * q + P.C_L_delta_e * de)
        drag = qS * (CD + P.C_D_q * c2V * q + P.C_D_delta_e * de)
        m_aero = qS * P.c * (P.C_m_0 + P.C_m_alpha * ad.alpha + P.C_m_q * c2V * q + P.C_m_delta_e * de)

        ca, sa = np.cos(ad.alpha), np.sin(ad.alpha)
        mg = P.mass * P.gravity
        fx = -ca * drag + sa * lift + thrust - mg * np.sin(theta)
        fz = -sa * drag - ca * lift + mg * np.cos(theta)

        x_dot = u * np.cos(theta) + w * np.sin(theta)
        h_dot = u * np.sin(theta) - w * np.cos(theta)
        u_dot = -q * w + fx / P.mass
        w_dot = q * u + fz / P.mass
        return np.array([x_dot, h_dot, u_dot, w_dot, q, m_aero / P.Jy])

    def step(self, s, delta, dt):
        return rk4_step(lambda z: self.derivatives(z, delta), s, dt)

    def trim(self, Va: float, h: float = 100.0):
        """Level-flight trim: returns ``(state, [delta_e, delta_t])``."""

        def res(z):
            alpha, de, dt = z
            s = np.array([0.0, h, Va * np.cos(alpha), Va * np.sin(alpha), alpha, 0.0])
            d = self.derivatives(s, [de, dt])
            return np.array([d[1], d[2], d[3], d[5]])  # h_dot, u_dot, w_dot, q_dot

        sol = least_squares(res, [0.05, 0.0, 0.6], bounds=([-0.4, -0.5, 0.0], [0.4, 0.5, 1.0]), xtol=1e-12, ftol=1e-12)
        alpha, de, dt = sol.x
        s = np.array([0.0, h, Va * np.cos(alpha), Va * np.sin(alpha), alpha, 0.0])
        return s, np.array([de, dt])

    @staticmethod
    def airspeed(s):
        return float(np.hypot(s[2], s[3]))


@dataclass
class Sim2DLog:
    t: list = field(default_factory=list)
    s: list = field(default_factory=list)
    delta: list = field(default_factory=list)
    theta_c: list = field(default_factory=list)
    wall_time: float = 0.0

    def arrays(self):
        s = np.array(self.s)
        return {
            "t": np.array(self.t),
            "x": s[:, 0], "h": s[:, 1], "u": s[:, 2], "w": s[:, 3],
            "theta": s[:, 4], "q": s[:, 5],
            "Va": np.hypot(s[:, 2], s[:, 3]),
            "delta": np.array(self.delta),
            "theta_c": np.array(self.theta_c),
        }  # fmt: skip

    @property
    def speedup(self):
        return (self.t[-1] - self.t[0]) / self.wall_time if self.wall_time > 0 else np.inf


class Sim2D:
    """2D simulator with cascaded altitude -> pitch PID and airspeed -> throttle PID."""

    def __init__(self, Va: float = 25.0, h0: float = 100.0, dt: float = 0.01, params: AircraftParams | None = None):
        self.model = Longitudinal2D(params)
        self.dt = dt
        self.s, self.trim_delta = self.model.trim(Va, h0)
        self.trim_theta = float(self.s[4])
        self.t = 0.0
        de0, dt0 = self.trim_delta
        self.pitch = PID(1.5, 0.3, 0.25, dt, (de0 - 0.5, de0 + 0.5))
        self.altitude = PID(0.03, 0.003, 0.0, dt, (-np.radians(20), np.radians(20)))
        self.airspeed = PID(0.12, 0.05, 0.0, dt, (-dt0, 1.0 - dt0))
        self.h_cmd = h0
        self.Va_cmd = Va
        self.theta_cmd = None  # set to override the altitude loop (pitch-hold mode)

    def control(self):
        s = self.s
        theta_c = self.theta_cmd
        if theta_c is None:
            theta_c = self.trim_theta + self.altitude.update(self.h_cmd, s[1])
        de = self.trim_delta[0] - self.pitch.update(theta_c, s[4], ydot=s[5])
        dt = self.trim_delta[1] + self.airspeed.update(self.Va_cmd, self.model.airspeed(s))
        return np.array([de, dt]), theta_c

    def run(self, t_final: float, realtime_factor: float | None = None, schedule=None) -> Sim2DLog:
        """Run for ``t_final`` seconds.

        ``schedule(t, sim)`` may change ``h_cmd``/``Va_cmd``/``theta_cmd``
        during the run. ``realtime_factor`` = 1 runs in real time, N runs N
        times faster than real time, ``None`` runs unthrottled.
        """
        log = Sim2DLog()
        wall0 = time.perf_counter()
        t0 = self.t
        for _ in range(int(round(t_final / self.dt))):
            if schedule:
                schedule(self.t, self)
            delta, theta_c = self.control()
            log.t.append(self.t)
            log.s.append(self.s.copy())
            log.delta.append(delta)
            log.theta_c.append(theta_c)
            self.s = self.model.step(self.s, delta, self.dt)
            self.t += self.dt
            if realtime_factor:
                lag = (self.t - t0) / realtime_factor - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
        log.wall_time = time.perf_counter() - wall0
        return log
