"""Compare PID, LQR and constrained MPC on the same manoeuvre.

Manoeuvre: climb 100 -> 130 m, turn to heading 90 deg, then speed up to 28 m/s.
"""

import time

import numpy as np
from _plot import save_figure

from uavsim import Simulation, SixDOF, trim
from uavsim.control import Autopilot, Commands, LQRController, MPCController

DT = 0.01
T = 60.0
model = SixDOF()
tr = trim(model, Va=25.0, altitude=100.0)


def ref(t):
    h = 100.0 if t < 2 else 130.0
    psi = 0.0 if t < 2 else np.radians(90)
    Va = 25.0 if t < 30 else 28.0
    return h, Va, psi


def run(name):
    sim = Simulation(model, tr.x, DT)
    if name == "PID":
        ap = Autopilot(DT, tr.delta, tr.theta, heading_source="heading")

        def ctrl(t, s):
            h, Va, psi = ref(t)
            return ap.update(Commands(Va=Va, h=h, chi=psi), s.state)

    elif name == "LQR":
        c = LQRController(model, tr, DT)

        def ctrl(t, s):
            return c.update(model.to_euler_state(s.x), *ref(t))

    else:
        c = MPCController(model, tr, dt=0.05, horizon=30)
        held = {"u": tr.delta}

        def ctrl(t, s):
            if round(t / DT) % 5 == 0:  # MPC runs at 20 Hz, zero-order hold in between
                held["u"] = c.update(model.to_euler_state(s.x), *ref(t))
            return held["u"]

    t0 = time.perf_counter()
    a = sim.run(T, ctrl).arrays()
    return a, time.perf_counter() - t0


results = {}
print(f"{'controller':10s} {'h err RMS':>10s} {'psi err RMS':>12s} {'Va err RMS':>11s} {'max |phi|':>10s} {'wall [s]':>9s}")
for name in ("PID", "LQR", "MPC"):
    a, wall = run(name)
    results[name] = a
    refs = np.array([ref(t) for t in a["t"]])
    eh = a["h"] - refs[:, 0]
    eVa = a["Va"] - refs[:, 1]
    epsi = np.degrees(np.angle(np.exp(1j * (a["psi"] - refs[:, 2]))))
    rms = lambda e: np.sqrt(np.mean(e**2))
    print(
        f"{name:10s} {rms(eh):9.2f}m {rms(epsi):10.2f}deg {rms(eVa):9.3f}m/s "
        f"{np.degrees(np.abs(a['phi']).max()):8.1f}deg {wall:9.2f}"
    )

t = results["PID"]["t"]
save_figure(
    "04_lqr_vs_mpc",
    [
        ("Altitude", t, [(results[k]["h"], k) for k in results], "m"),
        ("Heading", t, [(np.degrees(results[k]["psi"]), k) for k in results], "deg"),
        ("Airspeed", t, [(results[k]["Va"], k) for k in results], "m/s"),
        ("Elevator", t, [(np.degrees(results[k]["delta"][:, 0]), k) for k in results], "deg"),
    ],
)
