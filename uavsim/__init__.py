"""uavsim: fixed-wing UAV flight dynamics, control and autonomy in Python.

Main entry points
-----------------
* :class:`uavsim.dynamics.SixDOF`              nonlinear 6DOF model (Euler or quaternion)
* :class:`uavsim.sim2d.Sim2D`                  2D longitudinal simulator with PID loops
* :func:`uavsim.trim.trim` / :func:`uavsim.trim.linearize`
* :class:`uavsim.control.Autopilot`            PID roll / pitch / altitude / heading autopilot
* :class:`uavsim.control.LQRController`        LQR (LQI) tracking controller
* :class:`uavsim.control.MPCController`        constrained linear MPC
* :class:`uavsim.guidance.WaypointNavigator`   3D waypoint path following
* :class:`uavsim.sensors.SensorSuite`          noisy IMU / mag / baro / pitot / GPS
* :class:`uavsim.estimation.ComplementaryEstimator`
* :class:`uavsim.simulation.Simulation`        real-time or faster-than-real-time runner
"""

from .dynamics import FlightState, SixDOF
from .params import AircraftParams, aerosonde
from .simulation import SimLog, Simulation, Wind
from .trim import TrimResult, linearize, trim

__version__ = "0.2.0"

__all__ = [
    "AircraftParams",
    "FlightState",
    "SimLog",
    "Simulation",
    "SixDOF",
    "TrimResult",
    "Wind",
    "aerosonde",
    "linearize",
    "trim",
]
