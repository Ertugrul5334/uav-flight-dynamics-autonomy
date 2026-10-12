"""Attitude representations: Euler angles (3-2-1), rotation matrices and quaternions.

Conventions
-----------
* Inertial frame is North-East-Down (NED); body frame is x forward, y right, z down.
* Euler angles are the aerospace 3-2-1 sequence (yaw psi, pitch theta, roll phi).
* Rotation matrices returned here map body-frame vectors into the NED frame:
  ``v_ned = R @ v_body``.
* Quaternions are scalar-first ``[e0, e1, e2, e3]`` and represent the same
  body-to-NED rotation.
"""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------- #
# Euler angles
# --------------------------------------------------------------------------- #
def euler_to_rotation(phi: float, theta: float, psi: float) -> np.ndarray:
    """Body-to-NED rotation matrix from 3-2-1 Euler angles."""
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)
    return np.array(
        [
            [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
            [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
            [-sth, sph * cth, cph * cth],
        ]
    )


def rotation_to_euler(R: np.ndarray) -> np.ndarray:
    """3-2-1 Euler angles ``[phi, theta, psi]`` from a body-to-NED rotation matrix."""
    theta = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    phi = np.arctan2(R[2, 1], R[2, 2])
    psi = np.arctan2(R[1, 0], R[0, 0])
    return np.array([phi, theta, psi])


def euler_rates_matrix(phi: float, theta: float) -> np.ndarray:
    """Matrix ``W`` with ``[phi_dot, theta_dot, psi_dot] = W @ [p, q, r]``.

    Singular at theta = +/-90 deg (gimbal lock), which is why the quaternion
    formulation exists.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, tth = np.cos(theta), np.tan(theta)
    return np.array(
        [
            [1.0, sph * tth, cph * tth],
            [0.0, cph, -sph],
            [0.0, sph / cth, cph / cth],
        ]
    )


# --------------------------------------------------------------------------- #
# Quaternions
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` (scalar first)."""
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return np.array(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a body-frame vector into NED using quaternion ``q``."""
    qv = np.concatenate(([0.0], v))
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[1:]


def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    cph, sph = np.cos(phi / 2), np.sin(phi / 2)
    cth, sth = np.cos(theta / 2), np.sin(theta / 2)
    cps, sps = np.cos(psi / 2), np.sin(psi / 2)
    return np.array(
        [
            cps * cth * cph + sps * sth * sph,
            cps * cth * sph - sps * sth * cph,
            cps * sth * cph + sps * cth * sph,
            sps * cth * cph - cps * sth * sph,
        ]
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    e0, e1, e2, e3 = q
    phi = np.arctan2(2.0 * (e0 * e1 + e2 * e3), e0**2 + e3**2 - e1**2 - e2**2)
    theta = np.arcsin(np.clip(2.0 * (e0 * e2 - e1 * e3), -1.0, 1.0))
    psi = np.arctan2(2.0 * (e0 * e3 + e1 * e2), e0**2 + e1**2 - e2**2 - e3**2)
    return np.array([phi, theta, psi])


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Body-to-NED rotation matrix from a unit quaternion."""
    e0, e1, e2, e3 = q
    return np.array(
        [
            [e0**2 + e1**2 - e2**2 - e3**2, 2 * (e1 * e2 - e3 * e0), 2 * (e1 * e3 + e2 * e0)],
            [2 * (e1 * e2 + e3 * e0), e0**2 - e1**2 + e2**2 - e3**2, 2 * (e2 * e3 - e1 * e0)],
            [2 * (e1 * e3 - e2 * e0), 2 * (e2 * e3 + e1 * e0), e0**2 - e1**2 - e2**2 + e3**2],
        ]
    )


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion from a rotation matrix (Shepperd's method, numerically robust)."""
    tr = np.trace(R)
    candidates = np.array([tr, R[0, 0], R[1, 1], R[2, 2]])
    i = int(np.argmax(candidates))
    if i == 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 2:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q if q[0] >= 0 else -q


def quat_kinematics(q: np.ndarray, omega: np.ndarray, gain: float = 100.0) -> np.ndarray:
    """Quaternion derivative ``q_dot = 0.5 * Omega(omega) q``.

    A small constraint-stabilisation term ``gain * (1 - |q|^2) q`` pulls the
    quaternion back to unit norm so numerical drift does not accumulate.
    """
    p, qq, r = omega
    Omega = np.array(
        [
            [0.0, -p, -qq, -r],
            [p, 0.0, r, -qq],
            [qq, -r, 0.0, p],
            [r, qq, -p, 0.0],
        ]
    )
    return 0.5 * Omega @ q + gain * (1.0 - q @ q) * q
