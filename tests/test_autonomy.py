import numpy as np

from uavsim import Simulation, Wind
from uavsim.control import Autopilot, Commands
from uavsim.estimation import ComplementaryEstimator
from uavsim.guidance import WaypointNavigator
from uavsim.sensors import SensorParams, SensorSuite

DT = 0.01


def test_guidance_straight_line_law():
    nav = WaypointNavigator([[0, 0, 100], [1000, 0, 200]], k_path=0.05)
    on = nav.update(100.0, 0.0, 0.0)
    assert np.isclose(on.chi, 0.0) and np.isclose(on.cross_track, 0.0)
    assert np.isclose(on.h, 110.0)  # altitude interpolated along the leg
    right = nav.update(100.0, 50.0, 0.0)
    assert right.cross_track > 0 and right.chi < 0  # turn back left toward the path
    far = nav.update(100.0, 1e6, 0.0)
    assert np.isclose(far.chi, -nav.chi_inf, atol=1e-3)


def test_half_plane_switching():
    nav = WaypointNavigator([[0, 0, 100], [100, 0, 100], [100, 100, 100]])
    nav.update(50, 0, 0)
    assert nav.leg == 0
    nav.update(101, 1, 0)
    assert nav.leg == 1 and nav.reached == [1]
    nav.update(100, 101, np.pi / 2)
    assert nav.done


def test_loiter_orbit_after_last_waypoint():
    nav = WaypointNavigator([[0, 0, 100], [100, 0, 120]], loiter_radius=100.0)
    cmd = nav.update(150, 0, 0)  # past the final waypoint
    assert nav.done and np.isclose(cmd.h, 120)
    on_circle = nav.update(100, 100, np.pi)  # due east of centre on the circle
    assert np.isclose(on_circle.cross_track, 0.0)
    assert np.isclose(abs(on_circle.chi), np.pi)  # clockwise: heading south
    assert on_circle.phi_ff > 0


def test_sensor_noise_statistics(model, level_trim):
    sp = SensorParams(gyro_bias_walk=0.0)
    s = SensorSuite(model, sp, seed=0)
    gyro, accel, gps = [], [], 0
    for k in range(2000):
        m = s.update(k * DT, level_trim.x, level_trim.delta)
        gyro.append(m.gyro)
        accel.append(m.accel)
        gps += m.gps_new
    gyro, accel = np.array(gyro), np.array(accel)
    assert np.allclose(gyro.std(0), sp.gyro_sigma, rtol=0.15)
    assert np.allclose(gyro.mean(0), 0.0, atol=3 * sp.gyro_sigma / np.sqrt(len(gyro)) + 1e-12)
    # level trimmed flight: accelerometer reads -g rotated into body axes
    g_body = model.rotation(level_trim.x).T @ [0, 0, -model.P.gravity]
    assert np.allclose(accel.mean(0), g_body, atol=0.01)
    assert gps == 100  # 5 Hz for 20 s


def _mission(model, tr, use_estimator, seed=0):
    wps = [[0, 0, 100], [500, 0, 120], [500, 500, 140], [0, 500, 110], [0, 0, 100]]
    nav = WaypointNavigator(wps, Va=25.0)
    ap = Autopilot(DT, tr.delta, tr.theta)
    sim = Simulation(model, tr.x, DT, Wind((3.0, -2.0, 0.0), (0.5, 0.5, 0.2), seed=seed))
    sens = SensorSuite(model, seed=seed + 1)
    est = ComplementaryEstimator(DT, sim.state)
    last = {"d": tr.delta}

    def ctrl(t, s):
        if use_estimator:
            fs = est.update(sens.update(t, s.x, last["d"], s.wind_ned))
        else:
            fs = s.state
        g = nav.update(fs.pn, fs.pe, fs.chi)
        last["d"] = ap.update(Commands(Va=g.Va, h=g.h, chi=g.chi), fs)
        return last["d"]

    log = sim.run(150, ctrl, log_every=10)
    return nav, log.arrays()


def test_waypoint_mission_truth_feedback(model, level_trim):
    nav, _ = _mission(model, level_trim, use_estimator=False)
    assert nav.done and nav.reached == [1, 2, 3, 4]


def test_waypoint_mission_with_noisy_sensors(model, level_trim):
    nav, a = _mission(model, level_trim, use_estimator=True)
    assert nav.done and nav.reached == [1, 2, 3, 4]
    assert np.all(np.abs(a["phi"]) < np.radians(45))
    assert a["h"].min() > 80
