"""2D longitudinal simulation with cascaded PID loops.

Runs the same altitude-change scenario twice: unthrottled (as fast as the CPU
allows) and paced at a chosen real-time factor.

    python examples/01_sim2d_realtime.py            # unthrottled + 10x real time
    python examples/01_sim2d_realtime.py --rt 1     # watch it in real time
"""

import argparse

import numpy as np
from _plot import save_figure

from uavsim.metrics import step_info
from uavsim.sim2d import Sim2D

parser = argparse.ArgumentParser()
parser.add_argument("--rt", type=float, default=10.0, help="real-time factor for the paced run")
parser.add_argument("--duration", type=float, default=40.0)
args = parser.parse_args()


def schedule(t, sim):
    sim.h_cmd = 100.0 if t < 2.0 else 130.0
    sim.Va_cmd = 25.0 if t < 20.0 else 28.0


fast = Sim2D(Va=25.0, h0=100.0).run(args.duration, schedule=schedule)
print(f"unthrottled: {args.duration:.0f} s simulated in {fast.wall_time:.2f} s -> {fast.speedup:.0f}x real time")

paced = Sim2D(Va=25.0, h0=100.0).run(args.duration, realtime_factor=args.rt, schedule=schedule)
print(f"paced at {args.rt:g}x: wall time {paced.wall_time:.2f} s (target {args.duration / args.rt:.2f} s)")

a = fast.arrays()
i = (a["t"] >= 2.0) & (a["t"] < 20.0)  # before the airspeed step
info = step_info(a["t"][i] - 2.0, a["h"][i], 130.0, y_initial=100.0)
print("altitude step 100 -> 130 m: rise {rise_time:.1f} s, settle {settling_time:.1f} s, overshoot {overshoot_pct:.1f} %".format(**info))

save_figure(
    "01_sim2d",
    [
        ("Altitude", a["t"], [(a["h"], "h")], "m"),
        ("Pitch", a["t"], [(np.degrees(a["theta"]), "theta"), (np.degrees(a["theta_c"]), "command")], "deg"),
        ("Airspeed", a["t"], [(a["Va"], "Va")], "m/s"),
    ],
)
