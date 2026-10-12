"""Aerodynamic and propulsive force/moment model.

Forces and moments are built from the stability derivatives in
:class:`~uavsim.params.AircraftParams` using the standard linear-in-the-states
build-up, with a sigmoid blend to a flat-plate model past stall:

    L = qbar S [C_L(alpha) + C_L_q c q/(2Va) + C_L_de de]
    D = qbar S [C_D(alpha) + C_D_q c q/(2Va) + C_D_de de]
    Y = qbar S [C_Y_0 + C_Y_beta beta + C_Y_p b p/(2Va) + C_Y_r b r/(2Va) + C_Y_da da + C_Y_dr dr]
    l = qbar S b [C_l_0 + C_l_beta beta + C_l_p b p/(2Va) + C_l_r b r/(2Va) + C_l_da da + C_l_dr dr]
    m = qbar S c [C_m_0 + C_m_alpha alpha + C_m_q c q/(2Va) + C_m_de de]
    n = qbar S b [C_n_0 + C_n_beta beta + C_n_p b p/(2Va) + C_n_r b r/(2Va) + C_n_da da + C_n_dr dr]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import AircraftParams


@dataclass
class AirData:
    Va: float
    alpha: float
    beta: float
    qbar: float  # dynamic pressure, Pa


def air_data(v_rel_body: np.ndarray, rho: float) -> AirData:
    """Airspeed, angle of attack and sideslip from the air-relative body velocity."""
    ur, vr, wr = v_rel_body
    Va = float(np.sqrt(ur**2 + vr**2 + wr**2))
    if Va < 1e-6:
        return AirData(0.0, 0.0, 0.0, 0.0)
    alpha = float(np.arctan2(wr, ur))
    beta = float(np.arcsin(np.clip(vr / Va, -1.0, 1.0)))
    return AirData(Va, alpha, beta, 0.5 * rho * Va**2)


def lift_drag_coefficients(P: AircraftParams, alpha: float) -> tuple[float, float]:
    """Static C_L(alpha) with stall blending and C_D(alpha) from the drag polar."""
    M, a0 = P.M_stall, P.alpha0_stall
    e_neg = np.exp(-M * (alpha - a0))
    e_pos = np.exp(M * (alpha + a0))
    sigma = (1.0 + e_neg + e_pos) / ((1.0 + e_neg) * (1.0 + e_pos))
    CL_lin = P.C_L_0 + P.C_L_alpha * alpha
    CL_flat = 2.0 * np.sign(alpha) * np.sin(alpha) ** 2 * np.cos(alpha)
    CL = (1.0 - sigma) * CL_lin + sigma * CL_flat
    CD = P.C_D_p + CL_lin**2 / (np.pi * P.e * P.AR)
    return float(CL), float(CD)


def propeller_thrust(P: AircraftParams, Va: float, delta_t: float, rho: float) -> float:
    """Simple momentum-theory propeller model (thrust along body x)."""
    return 0.5 * rho * P.S_prop * P.C_prop * ((P.k_motor * delta_t) ** 2 - Va**2)


def aero_forces_moments(
    P: AircraftParams,
    ad: AirData,
    omega: np.ndarray,
    delta: np.ndarray,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Aerodynamic + propulsive body-frame force and moment (no gravity).

    ``delta`` is ``[elevator, aileron, rudder, throttle]`` (rad, rad, rad, 0..1).
    """
    de, da, dr, dt = delta
    p, q, r = omega
    thrust = propeller_thrust(P, ad.Va, dt, rho)
    if ad.Va < 1e-3:
        return np.array([thrust, 0.0, 0.0]), np.zeros(3)

    qS = ad.qbar * P.S
    c2V = P.c / (2.0 * ad.Va)
    b2V = P.b / (2.0 * ad.Va)
    alpha, beta = ad.alpha, ad.beta

    CL, CD = lift_drag_coefficients(P, alpha)
    lift = qS * (CL + P.C_L_q * c2V * q + P.C_L_delta_e * de)
    drag = qS * (CD + P.C_D_q * c2V * q + P.C_D_delta_e * de)

    ca, sa = np.cos(alpha), np.sin(alpha)
    fx = -ca * drag + sa * lift + thrust
    fz = -sa * drag - ca * lift
    fy = qS * (
        P.C_Y_0
        + P.C_Y_beta * beta
        + P.C_Y_p * b2V * p
        + P.C_Y_r * b2V * r
        + P.C_Y_delta_a * da
        + P.C_Y_delta_r * dr
    )

    ell = qS * P.b * (
        P.C_ell_0
        + P.C_ell_beta * beta
        + P.C_ell_p * b2V * p
        + P.C_ell_r * b2V * r
        + P.C_ell_delta_a * da
        + P.C_ell_delta_r * dr
    )
    m = qS * P.c * (P.C_m_0 + P.C_m_alpha * alpha + P.C_m_q * c2V * q + P.C_m_delta_e * de)
    n = qS * P.b * (
        P.C_n_0
        + P.C_n_beta * beta
        + P.C_n_p * b2V * p
        + P.C_n_r * b2V * r
        + P.C_n_delta_a * da
        + P.C_n_delta_r * dr
    )
    return np.array([fx, fy, fz]), np.array([ell, m, n])
