"""Discrete PID controller with anti-windup and a filtered derivative."""

from __future__ import annotations

import numpy as np

from ..rotations import wrap_angle


class PID:
    """``u = kp e + ki int(e) - kd d(y)/dt`` with practical extensions.

    * Derivative acts on the measurement (no derivative kick on setpoint steps)
      and is either supplied directly (e.g. a gyro rate) or computed with a
      first-order "dirty derivative" filter of time constant ``tau``.
    * Output saturation with integrator clamping (conditional integration) as
      anti-windup.
    * ``wrap=True`` wraps the error to [-pi, pi) for angle loops such as heading.
    """

    def __init__(
        self,
        kp: float,
        ki: float = 0.0,
        kd: float = 0.0,
        dt: float = 0.01,
        limits: tuple[float, float] = (-np.inf, np.inf),
        tau: float = 0.05,
        wrap: bool = False,
    ):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.lo, self.hi = limits
        self.tau = tau
        self.wrap = wrap
        self.reset()

    def reset(self):
        self.integrator = 0.0
        self._prev_y = None
        self._ydot = 0.0

    def update(self, ref: float, y: float, ydot: float | None = None, ff: float = 0.0) -> float:
        err = ref - y
        if self.wrap:
            err = float(wrap_angle(err))

        if ydot is None:
            if self._prev_y is None:
                self._ydot = 0.0
            else:
                dy = y - self._prev_y
                if self.wrap:
                    dy = float(wrap_angle(dy))
                a = (2 * self.tau - self.dt) / (2 * self.tau + self.dt)
                self._ydot = a * self._ydot + (1 - a) * dy / self.dt
            self._prev_y = y
            ydot = self._ydot

        integ_new = self.integrator + err * self.dt
        u_unsat = ff + self.kp * err + self.ki * integ_new - self.kd * ydot
        u = float(np.clip(u_unsat, self.lo, self.hi))
        # Anti-windup: only accept the integrator update if it does not push
        # further into saturation.
        if u == u_unsat or np.sign(err) != np.sign(u_unsat - u):
            self.integrator = integ_new
        return u
