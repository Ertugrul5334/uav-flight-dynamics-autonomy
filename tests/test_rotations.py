import numpy as np
import pytest

from uavsim.rotations import (
    euler_rates_matrix,
    euler_to_quat,
    euler_to_rotation,
    quat_kinematics,
    quat_multiply,
    quat_rotate,
    quat_to_euler,
    quat_to_rotation,
    rotation_to_euler,
    rotation_to_quat,
    wrap_angle,
)

rng = np.random.default_rng(0)
ANGLES = [
    (rng.uniform(-np.pi, np.pi), rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)) for _ in range(50)
]


@pytest.mark.parametrize("phi,theta,psi", ANGLES)
def test_euler_quat_dcm_roundtrips(phi, theta, psi):
    R = euler_to_rotation(phi, theta, psi)
    q = euler_to_quat(phi, theta, psi)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(quat_to_rotation(q), R, atol=1e-12)
    assert np.allclose(quat_to_euler(q), [phi, theta, psi], atol=1e-9)
    assert np.allclose(rotation_to_euler(R), [phi, theta, psi], atol=1e-9)
    q2 = rotation_to_quat(R)
    assert np.allclose(q2 * np.sign(q2[0]), q * np.sign(q[0]), atol=1e-9)


def test_quat_rotate_matches_dcm():
    q = euler_to_quat(0.3, -0.2, 1.1)
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(quat_rotate(q, v), quat_to_rotation(q) @ v)


def test_quat_multiply_composes_rotations():
    q1, q2 = euler_to_quat(0.1, 0.2, 0.3), euler_to_quat(-0.4, 0.1, 0.9)
    assert np.allclose(quat_to_rotation(quat_multiply(q1, q2)), quat_to_rotation(q1) @ quat_to_rotation(q2))


def test_quaternion_and_euler_kinematics_agree():
    phi, theta, psi = 0.2, 0.1, -0.5
    omega = np.array([0.3, -0.2, 0.5])
    q = euler_to_quat(phi, theta, psi)
    dt = 1e-6
    q_next = q + dt * quat_kinematics(q, omega)
    euler_dot_num = (quat_to_euler(q_next) - np.array([phi, theta, psi])) / dt
    assert np.allclose(euler_dot_num, euler_rates_matrix(phi, theta) @ omega, atol=1e-4)


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(-3 * np.pi / 2), np.pi / 2)
