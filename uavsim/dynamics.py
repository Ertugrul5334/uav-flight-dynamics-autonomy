"""Full nonlinear 3D six-degree-of-freedom rigid-body model.

State layout
------------
Quaternion mode (13 states)::

    [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]

Euler mode (12 states)::

    [pn, pe, pd, u, v, w, phi, theta, psi, p, q, r]

Translational equations (6 states: NED position + body velocity)::

    [pn, pe, pd]_dot = R_b^n [u, v, w]
    [u, v, w]_dot    = -omega x v + F_body / m

Rotational equations (3 angular rates + 3 orientation states)::

    J omega_dot = M_body - omega x (J omega)
    [phi, theta, psi]_dot = W(phi, theta) omega        (Euler mode)
    q_dot = 0.5 Omega(omega) q                          (quaternion mode)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import atmosphere
from .aerodynamics import aero_forces_moments, air_data
from .params import AircraftParams
from .rotations import (
    euler_rates_matrix,
    euler_to_quat,
    euler_to_rotation,
    quat_kinematics,
    quat_normalize,
    quat_to_euler,
    quat_to_rotation,
)


@dataclass
class FlightState:
    """Convenient named view of the aircraft state (true or estimated)."""

    pn: float
    pe: float
    h: float
    u: float
    v: float
    w: float
    phi: float
    theta: float
    psi: float
    p: float
    q: float
    r: float
    Va: float
    alpha: float
    beta: float
    chi: float  # course angle
    gamma: float  # flight-path angle
    Vg: float  # ground speed


def rk4_step(f, x, dt):
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class SixDOF:
    """Nonlinear 6DOF fixed-wing UAV model.

    Parameters
    ----------
    params:
        Airframe parameters (defaults to the Aerosonde-class UAV).
    attitude:
        ``"quaternion"`` (default, singularity-free) or ``"euler"``.
    use_isa:
        If True air density follows the ISA model with altitude, otherwise a
        constant ``rho`` is used.
    """

    def __init__(
        self,
        params: AircraftParams | None = None,
        attitude: str = "quaternion",
        use_isa: bool = True,
        rho: float = atmosphere.RHO0,
    ):
        if attitude not in ("quaternion", "euler"):
            raise ValueError("attitude must be 'quaternion' or 'euler'")
        self.P = params or AircraftParams()
        self.attitude = attitude
        self.use_isa = use_isa
        self.rho_const = rho
        self._J = self.P.J
        self._Jinv = np.linalg.inv(self._J)

    # ------------------------------------------------------------------ #
    @property
    def n_states(self) -> int:
        return 13 if self.attitude == "quaternion" else 12

    def make_state(self, position, velocity_body, euler, omega) -> np.ndarray:
        """Build a state vector from physical quantities (Euler angles in rad)."""
        att = euler_to_quat(*euler) if self.attitude == "quaternion" else np.asarray(euler, float)
        return np.concatenate(
            [np.asarray(position, float), np.asarray(velocity_body, float), att, np.asarray(omega, float)]
        )

    def split(self, x):
        """Return ``(position, velocity_body, attitude, omega)``."""
        if self.attitude == "quaternion":
            return x[0:3], x[3:6], x[6:10], x[10:13]
        return x[0:3], x[3:6], x[6:9], x[9:12]

    def euler(self, x) -> np.ndarray:
        att = self.split(x)[2]
        return quat_to_euler(att) if self.attitude == "quaternion" else np.array(att)

    def rotation(self, x) -> np.ndarray:
        att = self.split(x)[2]
        return quat_to_rotation(att) if self.attitude == "quaternion" else euler_to_rotation(*att)

    def to_euler_state(self, x) -> np.ndarray:
        pos, vel, _, omega = self.split(x)
        return np.concatenate([pos, vel, self.euler(x), omega])

    def from_euler_state(self, x12) -> np.ndarray:
        return self.make_state(x12[0:3], x12[3:6], x12[6:9], x12[9:12])

    def density(self, x) -> float:
        return atmosphere.density(-x[2]) if self.use_isa else self.rho_const

    # ------------------------------------------------------------------ #
    def forces_moments(self, x, delta, wind_ned=None):
        """Total body force (incl. gravity), body moment and auxiliary data."""
        _, vel, _, omega = self.split(x)
        R = self.rotation(x)
        wind = np.zeros(3) if wind_ned is None else np.asarray(wind_ned, float)
        v_rel = vel - R.T @ wind
        rho = self.density(x)
        ad = air_data(v_rel, rho)
        delta = np.clip(delta, self.P.delta_min, self.P.delta_max)
        F_aero, M = aero_forces_moments(self.P, ad, omega, delta, rho)
        g_body = R.T @ np.array([0.0, 0.0, self.P.mass * self.P.gravity])
        info = {
            "air": ad,
            "specific_force": F_aero / self.P.mass,  # what an accelerometer measures
            "rho": rho,
            "R": R,
        }
        return F_aero + g_body, M, info

    def derivatives(self, x, delta, wind_ned=None) -> np.ndarray:
        F, M, info = self.forces_moments(x, delta, wind_ned)
        _, vel, att, omega = self.split(x)
        R = info["R"]

        pos_dot = R @ vel
        vel_dot = -np.cross(omega, vel) + F / self.P.mass
        omega_dot = self._Jinv @ (M - np.cross(omega, self._J @ omega))

        if self.attitude == "quaternion":
            att_dot = quat_kinematics(att, omega)
        else:
            att_dot = euler_rates_matrix(att[0], att[1]) @ omega
        return np.concatenate([pos_dot, vel_dot, att_dot, omega_dot])

    def step(self, x, delta, dt, wind_ned=None) -> np.ndarray:
        """Advance the state by ``dt`` seconds with 4th-order Runge-Kutta."""
        x_next = rk4_step(lambda s: self.derivatives(s, delta, wind_ned), x, dt)
        if self.attitude == "quaternion":
            x_next[6:10] = quat_normalize(x_next[6:10])
        return x_next

    def flight_state(self, x, wind_ned=None) -> FlightState:
        pos, vel, _, omega = self.split(x)
        R = self.rotation(x)
        phi, theta, psi = self.euler(x)
        wind = np.zeros(3) if wind_ned is None else np.asarray(wind_ned, float)
        ad = air_data(vel - R.T @ wind, self.density(x))
        v_ned = R @ vel
        Vg = float(np.linalg.norm(v_ned))
        chi = float(np.arctan2(v_ned[1], v_ned[0]))
        gamma = float(np.arcsin(np.clip(-v_ned[2] / Vg, -1, 1))) if Vg > 1e-6 else 0.0
        return FlightState(
            pn=pos[0], pe=pos[1], h=-pos[2],
            u=vel[0], v=vel[1], w=vel[2],
            phi=phi, theta=theta, psi=psi,
            p=omega[0], q=omega[1], r=omega[2],
            Va=ad.Va, alpha=ad.alpha, beta=ad.beta,
            chi=chi, gamma=gamma, Vg=Vg,
        )  # fmt: skip
