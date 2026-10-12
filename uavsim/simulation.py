"""Simulation loop with real-time / faster-than-real-time pacing and logging."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .dynamics import FlightState, SixDOF


class Wind:
    """Steady NED wind plus optional first-order Gauss-Markov gusts.

    The gust model is a simplified Dryden approximation: each axis is a
    low-pass filtered white-noise process with standard deviation ``sigma``
    and correlation time ``Va/L`` set by length scale ``L``.
    """

    def __init__(self, steady_ned=(0.0, 0.0, 0.0), sigma=(0.0, 0.0, 0.0), L=200.0, Va=25.0, seed=None):
        self.steady = np.asarray(steady_ned, float)
        self.sigma = np.asarray(sigma, float)
        self.tau = L / Va
        self.rng = np.random.default_rng(seed)
        self.gust = np.zeros(3)

    def update(self, dt: float) -> np.ndarray:
        if np.any(self.sigma > 0):
            a = np.exp(-dt / self.tau)
            self.gust = a * self.gust + self.sigma * np.sqrt(1 - a**2) * self.rng.standard_normal(3)
        return self.steady + self.gust


@dataclass
class SimLog:
    t: list = field(default_factory=list)
    x: list = field(default_factory=list)  # 12-state Euler form
    delta: list = field(default_factory=list)
    states: list = field(default_factory=list)  # FlightState objects
    extra: dict = field(default_factory=dict)
    wall_time: float = 0.0

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"t": np.array(self.t), "x": np.array(self.x), "delta": np.array(self.delta)}
        for name in FlightState.__dataclass_fields__:
            out[name] = np.array([getattr(s, name) for s in self.states])
        for k, v in self.extra.items():
            out[k] = np.array(v)
        return out

    def log_extra(self, key, value):
        self.extra.setdefault(key, []).append(value)

    @property
    def speedup(self) -> float:
        """Simulated seconds per wall-clock second."""
        return (self.t[-1] - self.t[0]) / self.wall_time if self.wall_time > 0 and self.t else np.inf


class Simulation:
    """Fixed-step simulation of a :class:`SixDOF` model.

    ``controller(t, sim) -> delta`` is called once per step. ``realtime_factor``
    controls pacing: ``1.0`` runs in real time, ``10.0`` at ten times real time,
    and ``None`` as fast as the CPU allows.
    """

    def __init__(self, model: SixDOF, x0: np.ndarray, dt: float = 0.01, wind: Wind | None = None):
        self.model = model
        self.x = np.array(x0, float)
        self.dt = dt
        self.t = 0.0
        self.wind = wind
        self.wind_ned = np.zeros(3)

    @property
    def state(self) -> FlightState:
        return self.model.flight_state(self.x, self.wind_ned)

    def step(self, delta: np.ndarray):
        if self.wind is not None:
            self.wind_ned = self.wind.update(self.dt)
        self.x = self.model.step(self.x, delta, self.dt, self.wind_ned)
        self.t += self.dt

    def run(self, t_final: float, controller, realtime_factor: float | None = None, log_every: int = 1) -> SimLog:
        log = SimLog()
        n = int(round(t_final / self.dt))
        wall0 = time.perf_counter()
        t0 = self.t
        for k in range(n):
            delta = np.asarray(controller(self.t, self), float)
            if k % log_every == 0:
                log.t.append(self.t)
                log.x.append(self.model.to_euler_state(self.x))
                log.delta.append(delta)
                log.states.append(self.state)
            self.step(delta)
            if realtime_factor:
                target = (self.t - t0) / realtime_factor
                lag = target - (time.perf_counter() - wall0)
                if lag > 0:
                    time.sleep(lag)
        log.wall_time = time.perf_counter() - wall0
        return log
