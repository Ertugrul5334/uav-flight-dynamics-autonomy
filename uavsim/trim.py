"""Trim solver and numerical linearisation.

The trim problem (after Beard & McLain, ch. 5) finds the state and control
inputs that hold the aircraft in a steady flight condition defined by

* airspeed ``Va``,
* flight-path angle ``gamma`` (climb > 0), and
* turn radius ``R`` (``inf`` for straight flight, sign gives turn direction),

with zero sideslip (coordinated flight). It is solved as a bounded nonlinear
least-squares problem in the eight unknowns
``[alpha, beta, phi, theta, de, da, dr, dt]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .dynamics import SixDOF

STATE_NAMES = ["pn", "pe", "pd", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r"]
INPUT_NAMES = ["delta_e", "delta_a", "delta_r", "delta_t"]


@dataclass
class TrimResult:
    x: np.ndarray  # trim state in the model's own attitude representation
    x_euler: np.ndarray  # 12-state Euler form
    delta: np.ndarray  # [de, da, dr, dt]
    alpha: float
    beta: float
    phi: float
    theta: float
    Va: float
    gamma: float
    R: float
    residual: float
    success: bool


def _euler_model(model: SixDOF) -> SixDOF:
    if model.attitude == "euler":
        return model
    return SixDOF(model.P, attitude="euler", use_isa=model.use_isa, rho=model.rho_const)


def _trim_state(z, Va, gamma, R, h):
    alpha, beta, phi, theta = z[:4]
    psi_dot = 0.0 if np.isinf(R) else Va / R * np.cos(gamma)
    u = Va * np.cos(alpha) * np.cos(beta)
    v = Va * np.sin(beta)
    w = Va * np.sin(alpha) * np.cos(beta)
    p = -psi_dot * np.sin(theta)
    q = psi_dot * np.sin(phi) * np.cos(theta)
    r = psi_dot * np.cos(phi) * np.cos(theta)
    return np.array([0.0, 0.0, -h, u, v, w, phi, theta, 0.0, p, q, r])


def trim(
    model: SixDOF,
    Va: float,
    gamma: float = 0.0,
    R: float = np.inf,
    altitude: float = 100.0,
) -> TrimResult:
    em = _euler_model(model)
    P = em.P
    g = P.gravity
    phi0 = 0.0 if np.isinf(R) else np.arctan(Va**2 / (g * R))

    def residuals(z):
        x = _trim_state(z, Va, gamma, R, altitude)
        delta = z[4:8]
        xdot = em.derivatives(x, delta)
        res = np.concatenate(
            [
                xdot[3:6],  # zero translational accelerations
                xdot[9:12],  # zero angular accelerations
                [xdot[2] + Va * np.sin(gamma)],  # climb rate
                [10.0 * z[1]],  # coordinated flight: beta = 0
            ]
        )
        return res

    z0 = np.array([0.05, 0.0, phi0, 0.05 + gamma, 0.0, 0.0, 0.0, 0.6])
    lb = np.concatenate([[-0.4, -0.5, -1.3, -1.3], P.delta_min])
    ub = np.concatenate([[0.4, 0.5, 1.3, 1.3], P.delta_max])
    sol = least_squares(residuals, np.clip(z0, lb + 1e-6, ub - 1e-6), bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12)

    x_e = _trim_state(sol.x, Va, gamma, R, altitude)
    res = float(np.linalg.norm(residuals(sol.x)))
    return TrimResult(
        x=model.from_euler_state(x_e),
        x_euler=x_e,
        delta=sol.x[4:8].copy(),
        alpha=float(sol.x[0]),
        beta=float(sol.x[1]),
        phi=float(sol.x[2]),
        theta=float(sol.x[3]),
        Va=Va,
        gamma=gamma,
        R=R,
        residual=res,
        success=bool(sol.success and res < 1e-6),
    )


def linearize(model: SixDOF, x_euler: np.ndarray, delta: np.ndarray, eps: float = 1e-6):
    """Central-difference Jacobians ``A = df/dx`` and ``B = df/du`` of the
    12-state Euler-angle model about ``(x_euler, delta)``."""
    em = _euler_model(model)
    n, m = 12, 4
    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        A[:, i] = (em.derivatives(x_euler + dx, delta) - em.derivatives(x_euler - dx, delta)) / (2 * eps)
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        B[:, j] = (em.derivatives(x_euler, delta + du) - em.derivatives(x_euler, delta - du)) / (2 * eps)
    return A, B


# Index sets for the classic decoupled sub-models.
LONGITUDINAL = [3, 5, 7, 10, 2]  # u, w, theta, q, pd
LATERAL = [4, 9, 11, 6, 8]  # v, p, r, phi, psi


def modes(A: np.ndarray) -> dict[str, list[complex]]:
    """Eigenvalues of the decoupled longitudinal and lateral sub-matrices."""
    lon = A[np.ix_([3, 5, 10, 7], [3, 5, 10, 7])]
    lat = A[np.ix_([4, 9, 11, 6], [4, 9, 11, 6])]
    return {
        "longitudinal": sorted(np.linalg.eigvals(lon), key=lambda z: abs(z)),
        "lateral": sorted(np.linalg.eigvals(lat), key=lambda z: abs(z)),
    }
