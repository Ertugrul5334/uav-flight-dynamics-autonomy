import numpy as np
import pytest

from uavsim import Simulation, linearize, trim
from uavsim.trim import modes


@pytest.mark.parametrize(
    "Va,gamma,R",
    [(25.0, 0.0, np.inf), (20.0, 0.0, np.inf), (25.0, np.radians(5), np.inf), (25.0, 0.0, 200.0), (25.0, 0.0, -200.0)],
)
def test_trim_converges(model, Va, gamma, R):
    tr = trim(model, Va, gamma, R)
    assert tr.success
    assert abs(tr.beta) < 1e-6
    xdot = model.derivatives(tr.x, tr.delta)
    assert np.allclose(xdot[3:6], 0, atol=1e-6)  # no translational acceleration
    assert np.isclose(-xdot[2], Va * np.sin(gamma), atol=1e-6)
    if np.isfinite(R):
        assert np.isclose(np.tan(tr.phi), Va**2 / (model.P.gravity * R), rtol=0.05)


def test_trimmed_aircraft_stays_trimmed(model, level_trim):
    sim = Simulation(model, level_trim.x, dt=0.01)
    sim.run(20.0, lambda t, s: level_trim.delta)
    fs = sim.state
    assert abs(fs.h - 100.0) < 0.05
    assert abs(fs.Va - 25.0) < 0.01
    assert abs(fs.theta - level_trim.theta) < 1e-3


def test_linearisation_modes_are_physical(model, level_trim):
    A, B = linearize(model, level_trim.x_euler, level_trim.delta)
    lon = modes(A)["longitudinal"]
    # phugoid (slow, lightly damped) and short period (fast, well damped), both stable
    assert all(np.real(z) < 0 for z in lon)
    assert abs(lon[0]) < 1.0 < abs(lon[-1])
    # elevator mainly drives pitch acceleration, aileron mainly roll acceleration
    assert abs(B[10, 0]) > abs(B[9, 0]) and B[10, 0] < 0
    assert B[9, 1] > 0
