"""Trim the 6DOF model in several flight conditions and analyse its modes."""

import numpy as np

from uavsim import SixDOF, linearize, trim
from uavsim.trim import modes

model = SixDOF()
print("Stability derivatives used by the aerodynamic model:", len(model.P.stability_derivatives()))

conditions = [
    ("level, 25 m/s", 25.0, 0.0, np.inf),
    ("level, 20 m/s", 20.0, 0.0, np.inf),
    ("climb 5 deg", 25.0, np.radians(5), np.inf),
    ("right turn R=150 m", 25.0, 0.0, 150.0),
    ("climbing left turn", 25.0, np.radians(3), -200.0),
]
print(f"\n{'condition':22s} {'alpha':>7s} {'theta':>7s} {'phi':>7s} {'elev':>7s} {'ail':>7s} {'rud':>7s} {'thr':>6s} {'resid':>9s}")
for name, Va, gamma, R in conditions:
    t = trim(model, Va, gamma, R)
    d = np.degrees(t.delta[:3])
    print(
        f"{name:22s} {np.degrees(t.alpha):7.2f} {np.degrees(t.theta):7.2f} {np.degrees(t.phi):7.2f} "
        f"{d[0]:7.2f} {d[1]:7.2f} {d[2]:7.2f} {t.delta[3]:6.3f} {t.residual:9.1e}"
    )

t = trim(model, 25.0)
A, B = linearize(model, t.x_euler, t.delta)
for group, eig in modes(A).items():
    print(f"\n{group} eigenvalues:")
    for z in eig:
        if abs(z.imag) > 1e-6:
            wn = abs(z)
            print(f"  {z.real:+.3f} {z.imag:+.3f}j   wn={wn:.2f} rad/s  zeta={-z.real / wn:.2f}")
        else:
            print(f"  {z.real:+.4f}  (time constant {abs(1 / z.real) if z.real else np.inf:.2f} s)")
