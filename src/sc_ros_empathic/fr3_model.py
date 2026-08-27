"""
fr3_model.py
------------
Self-contained FR3 forward kinematics and geometric Jacobian in pure
numpy, from the nominal Denavit-Hartenberg parameters (modified /
Craig convention, as published by Franka). Used ONLY by the
manipulability / singularity-avoidance factor, which needs J(q) at an
ARBITRARY q: the current configuration AND the lookahead
q + qdot_candidate * dt for every candidate command.

Computed here **independently of the velocity controller and without
PyKDL**. The only input is the joint vector q, which the node already
receives from franka_states.

Nominal FR3 modified-DH (a_{i-1}, d_i, alpha_{i-1}); every joint is
revolute:
    i=1:   0        0.333     0
    i=2:   0        0        -pi/2
    i=3:   0        0.316     pi/2
    i=4:   0.0825   0         pi/2
    i=5:  -0.0825   0.384    -pi/2
    i=6:   0        0         pi/2
    i=7:   0.088    0         pi/2
    flange 0        0.107     0        (fixed, no joint -> fr3_link8)

Manufacturer nominal values, no per-robot calibration: fine for a
manipulability index, which needs the kinematic structure, not
millimetre accuracy.
"""

import numpy as np

N_JOINTS = 7

# (a_{i-1}, d_i, alpha_{i-1}) for i = 1..7
_DH = np.array([
    [0.0,     0.333,  0.0],
    [0.0,     0.0,   -np.pi / 2.0],
    [0.0,     0.316,  np.pi / 2.0],
    [0.0825,  0.0,    np.pi / 2.0],
    [-0.0825, 0.384, -np.pi / 2.0],
    [0.0,     0.0,    np.pi / 2.0],
    [0.088,   0.0,    np.pi / 2.0],
])
_FLANGE_D = 0.107   # fixed translation along z7 to fr3_link8


def _rx(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[1.0, 0.0, 0.0, 0.0],
                     [0.0,   c,  -s, 0.0],
                     [0.0,   s,   c, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def _rz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,  -s, 0.0, 0.0],
                     [s,   c, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_frames(q):
    """Return (T0, axes, origins):
      T0         4x4 base -> flange (fr3_link8) transform
      axes[i]    joint i rotation axis (unit, shape (3,)) in the base frame
      origins[i] a point on joint i's axis (shape (3,)) in the base frame
    for i = 0..6.
    """
    q = np.asarray(q, dtype=float)
    if q.shape[0] < N_JOINTS:
        raise ValueError("fr3_model: need 7 joint angles, got %d" % q.shape[0])
    T = np.eye(4)
    axes = np.zeros((N_JOINTS, 3))
    origins = np.zeros((N_JOINTS, 3))
    for i in range(N_JOINTS):
        a, d, alpha = _DH[i]
        # frame in which theta_i acts: after Rx(alpha_{i-1}) Dx(a_{i-1})
        M = T @ _rx(alpha)
        M[0, 3] += a * M[0, 0]
        M[1, 3] += a * M[1, 0]
        M[2, 3] += a * M[2, 0]
        axes[i] = M[:3, 2]
        origins[i] = M[:3, 3]
        Tz = _rz(q[i])
        Tz[2, 3] = d
        T = M @ Tz
    T = T.copy()
    T[:3, 3] += _FLANGE_D * T[:3, 2]
    return T, axes, origins


def fk(q):
    """4x4 base -> flange (fr3_link8) pose."""
    return fk_frames(q)[0]


def jacobian(q):
    """6x7 geometric Jacobian [linear; angular] of the flange w.r.t. q,
    expressed in the base frame."""
    T0, axes, origins = fk_frames(q)
    p_e = T0[:3, 3]
    J = np.zeros((6, N_JOINTS))
    for i in range(N_JOINTS):
        z = axes[i]
        J[:3, i] = np.cross(z, p_e - origins[i])
        J[3:, i] = z
    return J
