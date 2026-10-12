"""Noise-based sensor models.

Each sensor runs at its own rate and holds its last sample between updates:

* 3-axis rate gyro: white noise + slowly drifting bias (random walk)
* 3-axis accelerometer: specific force + white noise
* 3-axis magnetometer: Earth field in the body frame + white noise
* Static pressure (barometer) and differential pressure (pitot) sensors
* GPS position and velocity: slowly wandering first-order Gauss-Markov
  position error (the dominant real-world GPS error) plus white noise

Noise levels default to typical MEMS / small-UAV values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import atmosphere
from .dynamics import SixDOF


@dataclass
class SensorParams:
    gyro_sigma: float = np.radians(0.13)  # rad/s
    gyro_bias_walk: float = np.radians(0.002)  # rad/s/sqrt(s)
    accel_sigma: float = 0.0025 * 9.81  # m/s^2
    mag_sigma: float = 0.01  # normalised field units
    mag_inclination: float = np.radians(60.0)
    mag_declination: float = 0.0
    static_pres_sigma: float = 10.0  # Pa
    diff_pres_sigma: float = 2.0  # Pa
    gps_sigma_ne: float = 2.0  # m, stationary std of the Gauss-Markov error
    gps_sigma_h: float = 4.0
    gps_tau: float = 300.0  # s, Gauss-Markov correlation time
    gps_white: float = 0.3  # m, white position noise
    gps_vel_sigma: float = 0.05  # m/s
    rates: dict = field(
        default_factory=lambda: {"imu": 100.0, "mag": 50.0, "baro": 50.0, "pitot": 100.0, "gps": 5.0}
    )


@dataclass
class Measurements:
    t: float = 0.0
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mag: np.ndarray = field(default_factory=lambda: np.zeros(3))
    static_pressure: float = atmosphere.P0
    diff_pressure: float = 0.0
    gps_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [n, e, h]
    gps_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # NED velocity
    gps_new: bool = False
    gps_valid: bool = False


class SensorSuite:
    def __init__(self, model: SixDOF, params: SensorParams | None = None, seed: int | None = None):
        self.model = model
        self.p = params or SensorParams()
        self.rng = np.random.default_rng(seed)
        self.gyro_bias = np.zeros(3)
        self._gps_sig = np.array([self.p.gps_sigma_ne, self.p.gps_sigma_ne, self.p.gps_sigma_h])
        self.gps_err = self._gps_sig * self.rng.standard_normal(3)  # start in steady state
        self._next = {k: 0.0 for k in self.p.rates}
        self.meas = Measurements()
        inc, dec = self.p.mag_inclination, self.p.mag_declination
        self.mag_ned = np.array([np.cos(inc) * np.cos(dec), np.cos(inc) * np.sin(dec), np.sin(inc)])

    def _due(self, name: str, t: float) -> bool:
        if t + 1e-9 >= self._next[name]:
            self._next[name] += 1.0 / self.p.rates[name]
            return True
        return False

    def update(self, t: float, x: np.ndarray, delta: np.ndarray, wind_ned=None) -> Measurements:
        p, rng, m = self.p, self.rng, self.meas
        m.t = t
        m.gps_new = False
        _, vel, _, omega = self.model.split(x)
        _, _, info = self.model.forces_moments(x, delta, wind_ned)
        R = info["R"]
        h = -x[2]

        if self._due("imu", t):
            dt = 1.0 / p.rates["imu"]
            self.gyro_bias += p.gyro_bias_walk * np.sqrt(dt) * rng.standard_normal(3)
            m.gyro = omega + self.gyro_bias + p.gyro_sigma * rng.standard_normal(3)
            m.accel = info["specific_force"] + p.accel_sigma * rng.standard_normal(3)
        if self._due("mag", t):
            m.mag = R.T @ self.mag_ned + p.mag_sigma * rng.standard_normal(3)
        if self._due("baro", t):
            m.static_pressure = atmosphere.pressure(h) + p.static_pres_sigma * rng.standard_normal()
        if self._due("pitot", t):
            m.diff_pressure = 0.5 * info["rho"] * info["air"].Va ** 2 + p.diff_pres_sigma * rng.standard_normal()
        if self._due("gps", t):
            dt = 1.0 / p.rates["gps"]
            a = np.exp(-dt / p.gps_tau)
            self.gps_err = a * self.gps_err + self._gps_sig * np.sqrt(1 - a**2) * rng.standard_normal(3)
            m.gps_pos = np.array([x[0], x[1], h]) + self.gps_err + p.gps_white * rng.standard_normal(3)
            m.gps_vel = R @ vel + p.gps_vel_sigma * rng.standard_normal(3)
            m.gps_new = True
            m.gps_valid = True
        return m
