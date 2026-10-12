"""3D waypoint mission flown on noisy sensors in gusty wind.

Truth model (6DOF, quaternion) -> sensors (IMU, magnetometer, baro, pitot,
GPS) -> complementary estimator -> waypoint guidance -> PID autopilot.
After the last waypoint the aircraft loiters on a 150 m orbit.
"""

import numpy as np
from _plot import save_figure

from uavsim import Simulation, SixDOF, Wind, trim
from uavsim.control import Autopilot, Commands
from uavsim.estimation import ComplementaryEstimator
from uavsim.guidance import WaypointNavigator
from uavsim.sensors import SensorSuite

DT = 0.01
model = SixDOF()
tr = trim(model, Va=25.0, altitude=100.0)

waypoints = [
    [0, 0, 100],
    [800, 0, 120],
    [800, 600, 160],
    [200, 900, 140],
    [-200, 300, 110],
    [0, 0, 100],
]
nav = WaypointNavigator(waypoints, Va=25.0)
autopilot = Autopilot(DT, tr.delta, tr.theta)
sim = Simulation(model, tr.x, DT, wind=Wind(steady_ned=(4.0, -3.0, 0.0), sigma=(1.0, 1.0, 0.5), seed=1))
sensors = SensorSuite(model, seed=2)
estimator = ComplementaryEstimator(DT, sim.state)
last = {"delta": tr.delta}
est_log = []


def controller(t, s):
    meas = sensors.update(t, s.x, last["delta"], s.wind_ned)
    est = estimator.update(meas)
    cmd = nav.update(est.pn, est.pe, est.chi)
    last["delta"] = autopilot.update(Commands(Va=cmd.Va, h=cmd.h, chi=cmd.chi, phi_ff=cmd.phi_ff), est)
    est_log.append((est.phi, est.theta, est.psi, est.h, est.Va, s.state.phi, s.state.theta, s.state.psi, s.state.h, s.state.Va))
    return last["delta"]


log = sim.run(300.0, controller, log_every=10)
a = log.arrays()
E = np.array(est_log)
err = E[:, :5] - E[:, 5:]
err[:, 2] = np.angle(np.exp(1j * err[:, 2]))
print(f"waypoints reached: {nav.reached}  mission complete: {nav.done}")
print(f"simulated {a['t'][-1]:.0f} s at {log.speedup:.1f}x real time")
print(
    "estimation RMS error: roll {:.2f} deg, pitch {:.2f} deg, heading {:.2f} deg, altitude {:.2f} m, airspeed {:.2f} m/s".format(
        *np.degrees(np.sqrt(np.mean(err[:, :3] ** 2, 0))), *np.sqrt(np.mean(err[:, 3:] ** 2, 0))
    )
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from _plot import OUT

    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    wp = np.array(waypoints)
    ax.plot(a["pe"], a["pn"], a["h"], label="flown path")
    ax.plot(wp[:, 1], wp[:, 0], wp[:, 2], "o--", label="waypoints")
    ax.set_xlabel("east [m]")
    ax.set_ylabel("north [m]")
    ax.set_zlabel("altitude [m]")
    ax.legend()
    span = max(np.ptp(a["pn"]), np.ptp(a["pe"])) / 2
    cn, ce = (a["pn"].max() + a["pn"].min()) / 2, (a["pe"].max() + a["pe"].min()) / 2
    ax.set_xlim(ce - span, ce + span)
    ax.set_ylim(cn - span, cn + span)
    fig.savefig(OUT / "05_waypoint_mission_3d.png", dpi=110)
    plt.close(fig)
    print(f"saved {OUT / '05_waypoint_mission_3d.png'}")
except ImportError:
    pass

save_figure(
    "05_waypoint_mission",
    [
        ("Altitude", a["t"], [(a["h"], "h")], "m"),
        ("Roll", a["t"], [(np.degrees(a["phi"]), "phi")], "deg"),
        ("Airspeed", a["t"], [(a["Va"], "Va")], "m/s"),
    ],
)
