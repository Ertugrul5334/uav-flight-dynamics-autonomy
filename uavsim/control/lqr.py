"""Linear-quadratic regulator about a trim condition, with integral action.

The 12-state Euler model is linearised about trim. North/east position are
dropped (the aircraft is not required to return to a point), leaving the
10-state vector::

    x = [pd, u, v, w, phi, theta, psi, p, q, r]

Integrators on the altitude, airspeed and heading errors are appended
(LQI) so the closed loop tracks those references without steady-state error
even when the plant drifts away from the linearisation point.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are

from ..dynamics import SixDOF
from ..rotations import wrap_angle
from ..trim import TrimResult, linearize

# indices into the 12-state Euler vector
REDUCED_IDX = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
PD, U, PSI = 0, 1, 6  # positions of pd, u and psi within the reduced vector


def lqr(A, B, Q, R):
    """Continuous-time LQR gain ``K`` (u = -K x) and Riccati solution ``P``."""
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def dlqr(A, B, Q, R):
    """Discrete-time LQR gain ``K`` (u = -K x) and Riccati solution ``P``."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def bryson(max_values) -> np.ndarray:
    """Bryson's rule: diagonal weight 1/max^2 for each acceptable deviation."""
    return np.diag(1.0 / np.asarray(max_values, float) ** 2)


DEFAULT_Q = bryson([8.0, 2.0, 1.0, 1.0, np.radians(15), np.radians(10), np.radians(30), 1.0, 1.0, 1.0])
DEFAULT_QI = bryson([20.0, 5.0, np.radians(60)])  # integrals of pd, u, psi errors
DEFAULT_R = bryson([0.25, 0.25, 0.25, 0.4])


def reduced_model(model: SixDOF, tr: TrimResult):
    A, B = linearize(model, tr.x_euler, tr.delta)
    return A[np.ix_(REDUCED_IDX, REDUCED_IDX)], B[REDUCED_IDX, :]


def augmented_model(A: np.ndarray, B: np.ndarray):
    """Append integrators of the pd, u and psi errors to ``(A, B)``."""
    n, m = B.shape
    C = np.zeros((3, n))
    C[0, PD], C[1, U], C[2, PSI] = 1.0, 1.0, 1.0
    Aa = np.block([[A, np.zeros((n, 3))], [C, np.zeros((3, 3))]])
    Ba = np.vstack([B, np.zeros((3, m))])
    return Aa, Ba


def augmented_weights(Q=None, Qi=None):
    Q = DEFAULT_Q if Q is None else Q
    Qi = DEFAULT_QI if Qi is None else Qi
    n = Q.shape[0]
    return np.block([[Q, np.zeros((n, 3))], [np.zeros((3, n)), Qi]])


def tracking_error(x_euler, x_trim, Va_trim, h_ref, Va_ref, psi_ref, max_h_error, max_psi_error):
    """Deviation from the reference trim point, with clipped outer-loop errors.

    Returns ``(e, e_int)``: the 10-state deviation used for feedback and the
    altitude, airspeed and heading errors that drive the integrators.
    """
    x = x_euler[REDUCED_IDX]
    e = x - x_trim
    e[PD] = np.clip(x[PD] + h_ref, -max_h_error, max_h_error)  # pd - pd_ref
    e[U] = x[U] - (x_trim[U] + (Va_ref - Va_trim))
    e[PSI] = np.clip(wrap_angle(x[PSI] - psi_ref), -max_psi_error, max_psi_error)
    Va = np.linalg.norm(x_euler[3:6])
    return e, np.array([e[PD], Va - Va_ref, e[PSI]])


class LQRController:
    """LQI tracking controller for altitude, airspeed and heading."""

    def __init__(
        self,
        model: SixDOF,
        tr: TrimResult,
        dt: float,
        Q: np.ndarray | None = None,
        Qi: np.ndarray | None = None,
        R: np.ndarray | None = None,
        max_h_error: float = 10.0,
        max_psi_error: float = np.radians(20),
    ):
        self.model = model
        self.tr = tr
        self.dt = dt
        self.A, self.B = reduced_model(model, tr)
        self.Aa, self.Ba = augmented_model(self.A, self.B)
        self.K, self.P = lqr(self.Aa, self.Ba, augmented_weights(Q, Qi), DEFAULT_R if R is None else R)
        self.x_trim = tr.x_euler[REDUCED_IDX]
        self.u_trim = tr.delta
        self.z = np.zeros(3)
        self.max_h_error = max_h_error
        self.max_psi_error = max_psi_error
        self.lo, self.hi = model.P.delta_min, model.P.delta_max

    def closed_loop_poles(self):
        return np.linalg.eigvals(self.Aa - self.Ba @ self.K)

    def error(self, x_euler, h_ref, Va_ref, psi_ref):
        return tracking_error(
            x_euler, self.x_trim, self.tr.Va, h_ref, Va_ref, psi_ref, self.max_h_error, self.max_psi_error
        )

    def update(self, x_euler, h_ref, Va_ref=None, psi_ref=0.0) -> np.ndarray:
        Va_ref = self.tr.Va if Va_ref is None else Va_ref
        e, e_int = self.error(x_euler, h_ref, Va_ref, psi_ref)
        z_new = np.clip(self.z + self.dt * e_int, -50.0, 50.0)
        u = self.u_trim - self.K @ np.concatenate([e, z_new])
        u_sat = np.clip(u, self.lo, self.hi)
        if np.allclose(u, u_sat):  # anti-windup: freeze integrators when saturated
            self.z = z_new
        return u_sat
