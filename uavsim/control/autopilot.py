"""Successive-loop-closure PID autopilot for the 6DOF model.

Lateral:      heading/course --PI--> roll command --PD(+I)--> aileron
              sideslip --PI--> rudder
Longitudinal: altitude --PI--> pitch command --PD(+I)--> elevator
              airspeed --PI--> throttle
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dynamics import FlightState
from .pid import PID


@dataclass
class Commands:
    Va: float = 25.0  # airspeed, m/s
    h: float = 100.0  # altitude, m
    chi: float = 0.0  # course or heading, rad
    phi_ff: float = 0.0  # roll feed-forward (e.g. from orbit guidance)
    phi: float | None = None  # direct roll command (bypasses the heading loop)
    theta: float | None = None  # direct pitch command (bypasses the altitude loop)


@dataclass
class AutopilotGains:
    # roll -> aileron
    roll_kp: float = 0.8
    roll_ki: float = 0.05
    roll_kd: float = 0.06
    # heading -> roll
    heading_kp: float = 1.4
    heading_ki: float = 0.05
    # sideslip -> rudder
    beta_kp: float = 0.6
    beta_ki: float = 0.3
    # pitch -> elevator
    pitch_kp: float = 1.5
    pitch_ki: float = 0.3
    pitch_kd: float = 0.25
    # altitude -> pitch
    alt_kp: float = 0.03
    alt_ki: float = 0.003
    # airspeed -> throttle
    Va_kp: float = 0.12
    Va_ki: float = 0.05
    # limits
    roll_max: float = np.radians(35)
    pitch_max: float = np.radians(20)


class Autopilot:
    """Roll, pitch, altitude, heading (+ airspeed, sideslip) PID autopilot.

    ``heading_source`` selects whether the outer lateral loop regulates the
    heading ``psi`` or the ground-track course ``chi`` (used by waypoint
    guidance, since it accounts for wind).
    """

    def __init__(
        self,
        dt: float,
        trim_delta: np.ndarray,
        trim_theta: float = 0.0,
        gains: AutopilotGains | None = None,
        heading_source: str = "course",
        delta_min=(-0.5, -0.5, -0.5, 0.0),
        delta_max=(0.5, 0.5, 0.5, 1.0),
    ):
        g = gains or AutopilotGains()
        self.g = g
        self.dt = dt
        self.trim_delta = np.asarray(trim_delta, float)
        self.trim_theta = trim_theta
        self.heading_source = heading_source
        lo, hi = np.asarray(delta_min), np.asarray(delta_max)
        de0, da0, dr0, dt0 = self.trim_delta

        self.roll = PID(g.roll_kp, g.roll_ki, g.roll_kd, dt, (lo[1] - da0, hi[1] - da0))
        self.heading = PID(g.heading_kp, g.heading_ki, 0.0, dt, (-g.roll_max, g.roll_max), wrap=True)
        self.sideslip = PID(g.beta_kp, g.beta_ki, 0.0, dt, (lo[2] - dr0, hi[2] - dr0))
        # Elevator has negative control effectiveness (C_m_de < 0), so the
        # pitch loop output is negated when applied.
        self.pitch = PID(g.pitch_kp, g.pitch_ki, g.pitch_kd, dt, (de0 - hi[0], de0 - lo[0]))
        self.altitude = PID(g.alt_kp, g.alt_ki, 0.0, dt, (-g.pitch_max, g.pitch_max))
        self.airspeed = PID(g.Va_kp, g.Va_ki, 0.0, dt, (lo[3] - dt0, hi[3] - dt0))
        self.last = {}

    def reset(self):
        for c in (self.roll, self.heading, self.sideslip, self.pitch, self.altitude, self.airspeed):
            c.reset()

    def update(self, cmd: Commands, s: FlightState) -> np.ndarray:
        de0, da0, dr0, dt0 = self.trim_delta
        heading = s.chi if self.heading_source == "course" else s.psi

        if cmd.phi is None:
            phi_c = self.heading.update(cmd.chi, heading) + cmd.phi_ff
        else:
            phi_c = cmd.phi
        phi_c = float(np.clip(phi_c, -self.g.roll_max, self.g.roll_max))
        da = da0 + self.roll.update(phi_c, s.phi, ydot=s.p)
        dr = dr0 + self.sideslip.update(0.0, s.beta)  # C_n_dr < 0

        if cmd.theta is None:
            theta_c = self.trim_theta + self.altitude.update(cmd.h, s.h)
        else:
            theta_c = cmd.theta
        theta_c = float(np.clip(theta_c, -self.g.pitch_max, self.g.pitch_max))
        de = de0 - self.pitch.update(theta_c, s.theta, ydot=s.q)
        dt = dt0 + self.airspeed.update(cmd.Va, s.Va)

        self.last = {"phi_c": phi_c, "theta_c": theta_c}
        return np.array([de, da, dr, dt])
