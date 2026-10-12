"""Linear model predictive control with actuator constraints.

The linearised 10-state model (see :mod:`uavsim.control.lqr`), augmented with
integrators of the altitude, airspeed and heading errors for offset-free
tracking, is discretised with a zero-order hold. At each control update the
finite-horizon problem

    min_U  sum_{k=1..N-1} x_k' Q x_k + x_N' P x_N + sum_{k=0..N-1} du_k' R du_k
    s.t.   x_{k+1} = Ad x_k + Bd du_k,
           delta_min <= u_trim + du_k <= delta_max

is condensed into a box-constrained QP in the stacked inputs ``U`` and solved
with L-BFGS-B, warm-started from the shifted previous solution. ``P`` is the
discrete LQR cost-to-go, so the unconstrained MPC inherits LQR stability.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from ..dynamics import SixDOF
from ..trim import TrimResult
from .lqr import (
    DEFAULT_R,
    REDUCED_IDX,
    augmented_model,
    augmented_weights,
    dlqr,
    reduced_model,
    tracking_error,
)


def c2d(A, B, dt):
    """Zero-order-hold discretisation."""
    n, m = B.shape
    M = expm(np.block([[A, B], [np.zeros((m, n + m))]]) * dt)
    return M[:n, :n], M[:n, n:]


class MPCController:
    def __init__(
        self,
        model: SixDOF,
        tr: TrimResult,
        dt: float = 0.05,
        horizon: int = 30,
        Q: np.ndarray | None = None,
        Qi: np.ndarray | None = None,
        R: np.ndarray | None = None,
        max_h_error: float = 10.0,
        max_psi_error: float = np.radians(20),
    ):
        self.tr = tr
        self.dt = dt
        self.N = horizon
        A, B = reduced_model(model, tr)
        self.Ad, self.Bd = c2d(*augmented_model(A, B), dt)
        n, m = self.Bd.shape
        self.n, self.m = n, m
        Q = augmented_weights(Q, Qi)
        R = DEFAULT_R if R is None else R
        _, P = dlqr(self.Ad, self.Bd, Q * dt, R * dt)

        # Prediction: X = Phi x0 + Gamma U  with X = [x1; ...; xN]
        N = horizon
        Phi = np.zeros((N * n, n))
        Gamma = np.zeros((N * n, N * m))
        powers = [np.eye(n)]
        for _ in range(N):
            powers.append(self.Ad @ powers[-1])
        for k in range(N):
            Phi[k * n:(k + 1) * n] = powers[k + 1]
            for j in range(k + 1):
                Gamma[k * n:(k + 1) * n, j * m:(j + 1) * m] = powers[k - j] @ self.Bd
        Qbar = np.kron(np.eye(N), Q * dt)
        Qbar[-n:, -n:] = P
        Rbar = np.kron(np.eye(N), R * dt)
        self.Phi, self.Gamma = Phi, Gamma
        self.H = Gamma.T @ Qbar @ Gamma + Rbar
        self.G = Gamma.T @ Qbar

        lo = np.asarray(model.P.delta_min) - tr.delta
        hi = np.asarray(model.P.delta_max) - tr.delta
        self.lo, self.hi = lo, hi
        self.bounds = list(zip(np.tile(lo, N), np.tile(hi, N)))
        self.U = np.zeros(N * m)
        self.x_trim = tr.x_euler[REDUCED_IDX]
        self.z = np.zeros(3)
        self.max_h_error = max_h_error
        self.max_psi_error = max_psi_error
        self.last_cost = np.nan
        self.last_iterations = 0

    def error(self, x_euler, h_ref, Va_ref, psi_ref):
        return tracking_error(
            x_euler, self.x_trim, self.tr.Va, h_ref, Va_ref, psi_ref, self.max_h_error, self.max_psi_error
        )

    def solve(self, x0: np.ndarray) -> np.ndarray:
        """Solve the constrained QP from augmented state ``x0``; returns du_0."""
        f = self.G @ (self.Phi @ x0)
        H = self.H

        def cost(Uv):
            HU = H @ Uv
            return 0.5 * Uv @ HU + f @ Uv, HU + f

        U0 = np.concatenate([self.U[self.m:], self.U[-self.m:]])  # shifted warm start
        res = minimize(cost, U0, jac=True, method="L-BFGS-B", bounds=self.bounds, options={"maxiter": 300})
        self.U = res.x
        self.last_cost = float(res.fun)
        self.last_iterations = int(res.nit)
        return np.clip(res.x[: self.m], self.lo, self.hi)

    def update(self, x_euler, h_ref, Va_ref=None, psi_ref=0.0) -> np.ndarray:
        """Compute the next command; call once every ``self.dt`` seconds."""
        Va_ref = self.tr.Va if Va_ref is None else Va_ref
        e, e_int = self.error(x_euler, h_ref, Va_ref, psi_ref)
        du = self.solve(np.concatenate([e, self.z]))
        self.z = np.clip(self.z + self.dt * e_int, -50.0, 50.0)
        return self.tr.delta + du
