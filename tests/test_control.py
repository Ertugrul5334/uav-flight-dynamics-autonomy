import numpy as np
import pytest

from uavsim import Simulation
from uavsim.control import PID, Autopilot, Commands, LQRController, MPCController
from uavsim.metrics import step_info
from uavsim.sim2d import Sim2D

DT = 0.01


def test_pid_first_order_plant_and_anti_windup():
    pid = PID(2.0, 1.0, 0.0, dt=DT, limits=(-1, 1))
    y = 0.0
    for _ in range(2000):
        u = pid.update(5.0, y)  # unreachable setpoint: output saturates
        y += DT * (-y + u)
    assert np.isclose(u, 1.0)
    assert pid.integrator < 20.0  # bounded, not ~100 as without anti-windup
    pid2 = PID(2.0, 1.0, 0.0, dt=DT)
    y = 0.0
    for _ in range(3000):
        y += DT * (-y + pid2.update(0.5, y))
    assert abs(y - 0.5) < 1e-3


def test_pid_wraps_angles():
    pid = PID(1.0, wrap=True)
    assert np.isclose(pid.update(np.radians(179), np.radians(-179)), np.radians(-2))


def _ap_run(model, tr, cmd, T):
    ap = Autopilot(DT, tr.delta, tr.theta, heading_source="heading")
    sim = Simulation(model, tr.x, DT)
    return sim.run(T, lambda t, s: ap.update(cmd, s.state)).arrays()


def test_autopilot_roll_and_pitch_loops(model, level_trim):
    a = _ap_run(model, level_trim, Commands(phi=np.radians(20)), 5)
    info = step_info(a["t"], a["phi"], np.radians(20))
    assert info["settling_time"] < 1.5 and info["overshoot_pct"] < 10
    th = level_trim.theta + np.radians(5)
    a = _ap_run(model, level_trim, Commands(theta=th), 10)
    assert abs(a["theta"][-1] - th) < np.radians(0.2)


def test_autopilot_altitude_heading_airspeed(model, level_trim):
    a = _ap_run(model, level_trim, Commands(Va=28, h=130, chi=np.radians(90)), 60)
    assert abs(a["h"][-1] - 130) < 0.5
    assert abs(np.degrees(a["psi"][-1]) - 90) < 1.0
    assert abs(a["Va"][-1] - 28) < 0.1
    assert np.max(np.abs(a["phi"])) <= np.radians(40)


def test_sim2d_altitude_hold_and_faster_than_realtime():
    sim = Sim2D(Va=25, h0=100)
    log = sim.run(40.0, schedule=lambda t, s: setattr(s, "h_cmd", 120.0 if t > 1 else 100.0))
    a = log.arrays()
    assert abs(a["h"][-1] - 120) < 0.5
    assert abs(a["Va"][-1] - 25) < 0.2
    assert log.speedup > 5.0


def test_sim2d_realtime_pacing():
    log = Sim2D().run(0.5, realtime_factor=1.0)
    assert 0.45 < log.wall_time < 0.8
    log = Sim2D().run(1.0, realtime_factor=4.0)
    assert 0.2 < log.wall_time < 0.5


def test_lqr_closed_loop_stable_and_tracks(model, level_trim):
    c = LQRController(model, level_trim, DT)
    assert np.all(np.real(c.closed_loop_poles()) < 0)
    sim = Simulation(model, level_trim.x, DT)
    sim.run(50, lambda t, s: c.update(model.to_euler_state(s.x), 120.0, 27.0, np.radians(60)))
    fs = sim.state
    assert abs(fs.h - 120) < 0.5
    assert abs(fs.Va - 27) < 0.1
    assert abs(np.degrees(fs.psi) - 60) < 1.0


def test_lqr_rejects_disturbance(model, level_trim):
    c = LQRController(model, level_trim, DT)
    x = level_trim.x_euler.copy()
    x[9:12] += [0.5, 0.3, -0.3]  # rate kick
    sim = Simulation(model, model.from_euler_state(x), DT)
    sim.run(20, lambda t, s: c.update(model.to_euler_state(s.x), 100.0, 25.0, 0.0))
    fs = sim.state
    assert abs(fs.h - 100) < 0.5 and abs(fs.phi) < np.radians(0.5)


@pytest.mark.parametrize("horizon", [20])
def test_mpc_tracks_and_respects_limits(model, level_trim, horizon):
    c = MPCController(model, level_trim, dt=0.05, horizon=horizon)
    sim = Simulation(model, level_trim.x, DT)
    held = {"u": level_trim.delta}
    cmds = []

    def ctrl(t, s):
        if round(t / DT) % 5 == 0:
            held["u"] = c.update(model.to_euler_state(s.x), 115.0, 25.0, np.radians(45))
            cmds.append(held["u"])
        return held["u"]

    sim.run(40, ctrl)
    cmds = np.array(cmds)
    P = model.P
    assert np.all(cmds >= P.delta_min - 1e-9) and np.all(cmds <= P.delta_max + 1e-9)
    fs = sim.state
    assert abs(fs.h - 115) < 0.5
    assert abs(np.degrees(fs.psi) - 45) < 1.0
