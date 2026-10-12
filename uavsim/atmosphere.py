"""International Standard Atmosphere (troposphere, 0-11 km)."""

from __future__ import annotations

import numpy as np

G0 = 9.80665  # m/s^2
R_AIR = 287.05287  # J/(kg K)
T0 = 288.15  # K
P0 = 101325.0  # Pa
RHO0 = 1.225  # kg/m^3
LAPSE = 0.0065  # K/m


def isa(altitude_m: float) -> tuple[float, float, float]:
    """Return ``(temperature K, pressure Pa, density kg/m^3)`` at a geometric altitude."""
    h = float(np.clip(altitude_m, -500.0, 11000.0))
    T = T0 - LAPSE * h
    p = P0 * (T / T0) ** (G0 / (LAPSE * R_AIR))
    rho = p / (R_AIR * T)
    return T, p, rho


def density(altitude_m: float) -> float:
    return isa(altitude_m)[2]


def pressure(altitude_m: float) -> float:
    return isa(altitude_m)[1]


def pressure_altitude(p: float) -> float:
    """Invert the ISA pressure model (used by the barometer model)."""
    return (T0 / LAPSE) * (1.0 - (p / P0) ** (LAPSE * R_AIR / G0))
