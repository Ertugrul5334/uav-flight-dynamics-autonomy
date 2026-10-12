"""3D waypoint navigation.

Straight-line path following with a vector-field course law and half-plane
waypoint switching (after Beard & McLain, ch. 10-11). Waypoints are
``[north, east, altitude]``; altitude commands are interpolated along each
leg so the aircraft climbs or descends smoothly between waypoints. When the
last waypoint is reached (and ``loop`` is off) the aircraft loiters on a
circular orbit around it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rotations import wrap_angle


@dataclass
class GuidanceCommand:
    chi: float  # course command, rad
    h: float  # altitude command, m
    Va: float  # airspeed command, m/s
    cross_track: float  # signed lateral path error, m
    leg: int  # index of the active leg's start waypoint
    done: bool
    phi_ff: float = 0.0  # roll feed-forward for orbits, rad


class WaypointNavigator:
    def __init__(
        self,
        waypoints,
        Va: float = 25.0,
        chi_inf: float = np.radians(60),
        k_path: float = 0.02,
        loop: bool = False,
        loiter_radius: float = 150.0,
        loiter_direction: int = 1,
        k_orbit: float = 2.0,
    ):
        wp = np.asarray(waypoints, float)
        if wp.ndim != 2 or wp.shape[1] != 3 or len(wp) < 2:
            raise ValueError("waypoints must be an (N>=2, 3) array of [north, east, altitude]")
        self.wp = wp
        self.Va = Va
        self.chi_inf = chi_inf
        self.k_path = k_path
        self.loop = loop
        self.loiter_radius = loiter_radius
        self.loiter_direction = 1 if loiter_direction >= 0 else -1
        self.k_orbit = k_orbit
        self.leg = 0
        self.done = False
        self.reached: list[int] = []

    def _leg(self, i):
        n = len(self.wp)
        a = self.wp[i % n]
        b = self.wp[(i + 1) % n]
        return a, b

    def _direction(self, i):
        a, b = self._leg(i)
        d = b[:2] - a[:2]
        return d / np.linalg.norm(d)

    def _last_leg(self) -> bool:
        return not self.loop and self.leg >= len(self.wp) - 2

    def update(self, pn: float, pe: float, chi: float) -> GuidanceCommand:
        pos = np.array([pn, pe])

        # Half-plane switching at the end of the current leg.
        while not self.done:
            a, b = self._leg(self.leg)
            q_i = self._direction(self.leg)
            if self._last_leg():
                normal = q_i
            else:
                q_next = self._direction(self.leg + 1)
                normal = q_i + q_next
                nn = np.linalg.norm(normal)
                normal = q_i if nn < 1e-9 else normal / nn
            if (pos - b[:2]) @ normal >= 0.0:
                self.reached.append((self.leg + 1) % len(self.wp))
                if self._last_leg():
                    self.done = True
                else:
                    self.leg += 1
                    continue
            break

        a, b = self._leg(self.leg)
        if self.done:
            return self._orbit(pos, chi, b)

        q = self._direction(self.leg)
        chi_q = np.arctan2(q[1], q[0])
        chi_q = chi + wrap_angle(chi_q - chi)  # unwrap towards the current course
        rel = pos - a[:2]
        e_py = -np.sin(chi_q) * rel[0] + np.cos(chi_q) * rel[1]
        chi_c = chi_q - self.chi_inf * 2.0 / np.pi * np.arctan(self.k_path * e_py)
        L = np.linalg.norm(b[:2] - a[:2])
        s = np.clip(rel @ q / L, 0.0, 1.0)
        h_c = a[2] + s * (b[2] - a[2])
        return GuidanceCommand(float(wrap_angle(chi_c)), float(h_c), self.Va, float(e_py), self.leg, self.done)

    def _orbit(self, pos, chi, center) -> GuidanceCommand:
        """Circular loiter around ``center`` (lambda = +1 clockwise seen from above)."""
        lam, rho = self.loiter_direction, self.loiter_radius
        rel = pos - center[:2]
        d = float(np.linalg.norm(rel))
        varphi = np.arctan2(rel[1], rel[0])
        varphi = chi + wrap_angle(varphi - chi)
        chi_c = varphi + lam * (np.pi / 2 + np.arctan(self.k_orbit * (d - rho) / rho))
        phi_ff = lam * np.arctan(self.Va**2 / (9.80665 * rho)) if d < 2 * rho else 0.0
        return GuidanceCommand(
            float(wrap_angle(chi_c)), float(center[2]), self.Va, d - rho, self.leg, True, float(phi_ff)
        )
