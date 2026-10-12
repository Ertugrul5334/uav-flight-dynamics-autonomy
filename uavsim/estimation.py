"""Lightweight state estimator that turns noisy sensor data into a FlightState.

* Body rates: low-pass filtered gyros.
* Roll and pitch: complementary filter. Gyro-propagated Euler angles are
  corrected toward the gravity direction recovered from the accelerometer,
  compensated for centripetal acceleration ``omega x [Va, 0, 0]``.
* Heading: complementary filter of gyro yaw rate and tilt-compensated
  magnetometer heading.
* Altitude and airspeed: inverted barometric / pitot pressure, low-pass filtered.
* Position, course and ground speed: GPS, dead-reckoned between fixes.
"""

from __future__ import annotations

import numpy as np

from . import atmosphere
from .dynamics import FlightState
from .rotations import euler_rates_matrix, euler_to_rotation, wrap_angle
from .sensors import Measurements


def _lpf(prev, new, alpha):
    return alpha * prev + (1 - alpha) * new


class ComplementaryEstimator:
    def __init__(
        self,
        dt: float,
        initial: FlightState,
        k_att: float = 0.5,
        k_psi: float = 0.5,
        a_gyro: float = 0.5,
        a_baro: float = 0.95,
        a_pitot: float = 0.9,
        mag_declination: float = 0.0,
    ):
        self.dt = dt
        self.k_att = k_att
        self.k_psi = k_psi
        self.a_gyro, self.a_baro, self.a_pitot = a_gyro, a_baro, a_pitot
        self.dec = mag_declination
        self.s = FlightState(**vars(initial))
        self.omega = np.array([initial.p, initial.q, initial.r])
        self.vel_ned = np.array([initial.Vg * np.cos(initial.chi), initial.Vg * np.sin(initial.chi), 0.0])
        self.t = 0.0

    def update(self, m: Measurements) -> FlightState:
        s, dt = self.s, self.dt
        self.t += dt

        # --- rates ----------------------------------------------------------
        self.omega = _lpf(self.omega, m.gyro, self.a_gyro)
        p, q, r = self.omega

        # --- airspeed and altitude -------------------------------------------
        rho = atmosphere.density(s.h)
        Va_meas = np.sqrt(max(2.0 * m.diff_pressure / rho, 0.0))
        s.Va = _lpf(s.Va, Va_meas, self.a_pitot)
        s.h = _lpf(s.h, atmosphere.pressure_altitude(m.static_pressure), self.a_baro)

        # --- roll / pitch complementary filter -----------------------------
        phi, theta = s.phi, s.theta
        rates = euler_rates_matrix(phi, theta) @ self.omega
        phi += rates[0] * dt
        theta += rates[1] * dt
        g_body = np.cross(self.omega, [s.Va, 0.0, 0.0]) - m.accel  # quasi-steady gravity
        phi_acc = np.arctan2(g_body[1], g_body[2])
        theta_acc = np.arctan2(-g_body[0], np.hypot(g_body[1], g_body[2]))
        phi += self.k_att * dt * wrap_angle(phi_acc - phi)
        theta += self.k_att * dt * (theta_acc - theta)
        s.phi, s.theta = float(wrap_angle(phi)), float(np.clip(theta, -1.5, 1.5))

        # --- heading: gyro + tilt-compensated magnetometer ------------------
        psi = s.psi + rates[2] * dt
        R_tilt = euler_to_rotation(s.phi, s.theta, 0.0)  # body -> level frame
        m_level = R_tilt @ m.mag
        psi_mag = np.arctan2(-m_level[1], m_level[0]) + self.dec
        psi += self.k_psi * dt * wrap_angle(psi_mag - psi)
        s.psi = float(wrap_angle(psi))

        # --- GPS position / course, dead-reckoned between fixes -------------
        if m.gps_new:
            self.vel_ned = m.gps_vel.copy()
            s.pn, s.pe = m.gps_pos[0], m.gps_pos[1]
        else:
            s.pn += self.vel_ned[0] * dt
            s.pe += self.vel_ned[1] * dt
        s.Vg = float(np.hypot(self.vel_ned[0], self.vel_ned[1]))
        if s.Vg > 1.0:
            s.chi = float(np.arctan2(self.vel_ned[1], self.vel_ned[0]))

        s.p, s.q, s.r = float(p), float(q), float(r)
        s.beta = 0.0  # not observable with this sensor set
        s.gamma = float(np.arcsin(np.clip(-self.vel_ned[2] / max(np.linalg.norm(self.vel_ned), 1e-3), -1, 1)))
        s.alpha = s.theta - s.gamma
        return FlightState(**vars(s))
