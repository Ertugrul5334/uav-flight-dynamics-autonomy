"""PID autopilot on the full 6DOF model: roll, pitch, altitude and heading loops.

Each loop is stepped in isolation and its step-response metrics are printed,
including the number of control iterations (100 Hz) needed to settle.
"""

import numpy as np
from _plot import save_figure

from uavsim import Simulation, SixDOF, trim
from uavsim.control import Autopilot, Commands
from uavsim.metrics import step_info

DT = 0.01
model = SixDOF(attitude="quaternion")
tr = trim(model, Va=25.0, altitude=100.0)

cases = [
    ("roll 0 -> 20 deg", Commands(phi=np.radians(20)), 5, "phi", np.radians(20)),
    ("pitch +5 deg", Commands(theta=tr.theta + np.radians(5)), 10, "theta", tr.theta + np.radians(5)),
    ("altitude 100 -> 120 m", Commands(h=120), 40, "h", 120.0),
    ("heading 0 -> 45 deg", Commands(chi=np.radians(45)), 40, "psi", np.radians(45)),
]

panels = []
print(f"{'loop':24s} {'rise [s]':>9s} {'settle [s]':>11s} {'iterations':>11s} {'overshoot':>10s}")
for name, cmd, T, key, target in cases:
    ap = Autopilot(DT, tr.delta, tr.theta, heading_source="heading")
    sim = Simulation(model, tr.x, DT)
    a = sim.run(T, lambda t, s: ap.update(cmd, s.state)).arrays()
    info = step_info(a["t"], a[key], target)
    print(
        f"{name:24s} {info['rise_time']:9.2f} {info['settling_time']:11.2f} "
        f"{info['settling_steps']:11d} {info['overshoot_pct']:9.1f}%"
    )
    scale = 1.0 if key == "h" else 180 / np.pi
    panels.append((name, a["t"], [(a[key] * scale, key), (np.full_like(a["t"], target * scale), "command")], ""))

save_figure("02_pid_autopilot", panels)
