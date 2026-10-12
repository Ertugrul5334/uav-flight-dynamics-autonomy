"""Aircraft parameter sets.

The default airframe is an Aerosonde-class small fixed-wing UAV. Mass,
geometry and stability derivatives are adapted from Beard & McLain,
*Small Unmanned Aircraft: Theory and Practice* (Princeton, 2012), App. E.
The propeller constant ``k_motor`` is reduced so the static thrust is
realistic (~125 N) for a 13.5 kg aircraft.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class AircraftParams:
    # ---- mass and inertia -------------------------------------------------
    mass: float = 13.5  # kg
    Jx: float = 0.8244  # kg m^2
    Jy: float = 1.135
    Jz: float = 1.759
    Jxz: float = 0.1204
    gravity: float = 9.80665

    # ---- geometry -----------------------------------------------------------
    S: float = 0.55  # wing area, m^2
    b: float = 2.8956  # span, m
    c: float = 0.18994  # mean aerodynamic chord, m
    e: float = 0.9  # Oswald efficiency

    # ---- propulsion -------------------------------------------------------
    S_prop: float = 0.2027
    C_prop: float = 1.0
    k_motor: float = 40.0

    # ---- longitudinal stability derivatives ------------------------------
    C_L_0: float = 0.28
    C_L_alpha: float = 3.45
    C_L_q: float = 0.0
    C_L_delta_e: float = 0.36
    C_D_p: float = 0.0437  # parasitic drag
    C_D_q: float = 0.0
    C_D_delta_e: float = 0.0
    C_m_0: float = -0.02338
    C_m_alpha: float = -0.38
    C_m_q: float = -3.6
    C_m_delta_e: float = -0.5
    M_stall: float = 50.0  # stall blending steepness
    alpha0_stall: float = 0.4712  # stall angle, rad

    # ---- lateral-directional stability derivatives -----------------------
    C_Y_0: float = 0.0
    C_Y_beta: float = -0.98
    C_Y_p: float = 0.0
    C_Y_r: float = 0.0
    C_Y_delta_a: float = 0.0
    C_Y_delta_r: float = -0.17
    C_ell_0: float = 0.0
    C_ell_beta: float = -0.12
    C_ell_p: float = -0.26
    C_ell_r: float = 0.14
    C_ell_delta_a: float = 0.08
    C_ell_delta_r: float = 0.105
    C_n_0: float = 0.0
    C_n_beta: float = 0.25
    C_n_p: float = 0.022
    C_n_r: float = -0.35
    C_n_delta_a: float = 0.06
    C_n_delta_r: float = -0.032

    # ---- actuator limits [elevator, aileron, rudder, throttle] -----------
    delta_min: np.ndarray = field(default_factory=lambda: np.array([-0.5, -0.5, -0.5, 0.0]))
    delta_max: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))

    @property
    def AR(self) -> float:
        return self.b**2 / self.S

    @property
    def J(self) -> np.ndarray:
        return np.array(
            [[self.Jx, 0.0, -self.Jxz], [0.0, self.Jy, 0.0], [-self.Jxz, 0.0, self.Jz]]
        )

    def stability_derivatives(self) -> dict[str, float]:
        """All aerodynamic coefficients and derivatives, keyed by name."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name.startswith("C_") and f.name not in ("C_prop",)
        }


def aerosonde() -> AircraftParams:
    return AircraftParams()
