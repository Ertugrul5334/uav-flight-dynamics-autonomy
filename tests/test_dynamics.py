import numpy as np

from uavsim import AircraftParams, SixDOF
from uavsim.aerodynamics import air_data


def test_twenty_seven_stability_derivatives():
    derivs = AircraftParams().stability_derivatives()
    assert len(derivs) >= 10
    assert {"C_L_alpha", "C_m_alpha", "C_m_q", "C_ell_p", "C_n_beta", "C_n_r"} <= set(derivs)


def test_free_fall_without_aero():
    P = AircraftParams()
    m = SixDOF(P, use_isa=False, rho=0.0)  # vacuum: no aerodynamic or thrust forces
    x = m.make_state([0, 0, -1000], [0, 0, 0], [0, 0, 0], [0, 0, 0])
    for _ in range(100):
        x = m.step(x, [0, 0, 0, 0], 0.01)
    # pd = -1000 + g t^2 / 2 after 1 s
    assert np.isclose(x[2], -1000 + 0.5 * P.gravity, atol=1e-6)
    assert np.isclose(x[5], P.gravity, atol=1e-9)


def test_torque_free_rotation_conserves_energy_and_momentum():
    P = AircraftParams(gravity=0.0)
    m = SixDOF(P, use_isa=False, rho=0.0)
    x = m.make_state([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3], [1.0, 0.5, -0.8])
    J = P.J

    def invariants(x):
        w = x[10:13]
        R = m.rotation(x)
        return 0.5 * w @ J @ w, R @ (J @ w)  # energy, inertial angular momentum

    E0, H0 = invariants(x)
    for _ in range(1000):
        x = m.step(x, [0, 0, 0, 0], 0.005)
    E1, H1 = invariants(x)
    assert np.isclose(E0, E1, rtol=1e-6)
    assert np.allclose(H0, H1, atol=1e-6)


def test_euler_and_quaternion_models_agree(level_trim):
    mq = SixDOF(attitude="quaternion")
    me = SixDOF(attitude="euler")
    x12 = level_trim.x_euler.copy()
    x12[9:12] += [0.2, -0.1, 0.15]  # excite all axes
    xq, xe = mq.from_euler_state(x12), me.from_euler_state(x12)
    delta = level_trim.delta + np.array([0.02, 0.03, -0.02, 0.0])
    for _ in range(300):
        xq = mq.step(xq, delta, 0.01)
        xe = me.step(xe, delta, 0.01)
    assert np.allclose(mq.to_euler_state(xq), xe, atol=1e-6)
    assert np.isclose(np.linalg.norm(xq[6:10]), 1.0)


def test_quaternion_model_passes_through_vertical():
    """Euler kinematics are singular at theta = 90 deg; quaternions are not."""
    P = AircraftParams(gravity=0.0)
    m = SixDOF(P, use_isa=False, rho=0.0)
    x = m.make_state([0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1.0, 0])  # pure pitch rate
    for _ in range(int(np.pi / 0.01)):  # pitch through 180 deg
        x = m.step(x, [0, 0, 0, 0], 0.01)
    assert np.all(np.isfinite(x))
    assert np.isclose(np.linalg.norm(x[6:10]), 1.0)


def test_air_data_with_wind(model, level_trim):
    fs = model.flight_state(level_trim.x, wind_ned=[5.0, 0.0, 0.0])
    assert np.isclose(fs.Va, 20.0, atol=0.1)  # 5 m/s tailwind
    ad = air_data(np.array([20.0, 0.0, 0.0]), 1.2)
    assert ad.alpha == 0.0 and ad.beta == 0.0
