"""
dh_utils.py
-----------
Pure-numpy forward kinematics and numerical Jacobian for the 4-DoF
human-arm kinematic model described in the visuo-tactile perception
paper (shoulder flex/ext, shoulder abd/add, shoulder int/ext rotation,
elbow flex/ext), following the Denavit-Hartenberg parameters:

Joint (i) | a_i | alpha_i |  d_i | theta_i
    1     |  0  |  pi/2   |   0  |   q1
    2     |  0  |  pi/2   |   0  |   q2
    3     |  0  | -pi/2   |  l1  |   q3
    4     |  l2 |   0     |   0  |   pi/2 - q4

q4 is the ELBOW FLEXION angle: q4 = 0 is the fully extended arm
(forearm colinear with the upper arm), q4 > 0 flexes it (q4 ~ 145 deg
fully flexed), matching DEFAULT_JOINT_LIMITS q4 = [0, 2.53]. The
pi/2 - q4 in theta_4 is what makes q4 = 0 the straight arm -- joint 3's
-pi/2 twist otherwise puts the forearm perpendicular to the upper arm
at theta_4 = 0. NOTE: the visuo-tactile pipeline reports the elbow
INTERIOR angle instead (pi = extended); shared_control_node converts it
(q4 <- pi - right_arm_q4) before anything here sees it.

No ROS/KDL dependency: this module is only used to relate a candidate
Cartesian velocity at the wrist to the corresponding human joint
velocities (q1_dot .. q4_dot), which is what the joint-limit safety
factor needs. l1 (upper-arm length) and l2 (forearm length) are
time-varying estimates coming from the visuo-tactile pipeline (bone
length estimator).

Ported from the offline simulation/verification package (the elbow
zero was corrected here: theta_4 = pi/2 - q4, not q4).
"""

import numpy as np

N_HUMAN_JOINTS = 4


def dh_transform(a, alpha, d, theta):
    """Standard Denavit-Hartenberg homogeneous transform."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,       ca,      d],
        [0,         0,        0,      1],
    ])


def human_arm_dh_params(q, l1, l2):
    """Return the list of (a, alpha, d, theta) tuples for q = [q1..q4].

    theta_4 = pi/2 - q4 so that q4 is elbow flexion measured from the
    extended arm (q4 = 0 -> straight); see the module docstring.
    """
    q1, q2, q3, q4 = q
    return [
        (0.0, np.pi / 2, 0.0, q1),
        (0.0, np.pi / 2, 0.0, q2),
        (0.0, -np.pi / 2, l1, q3),
        (l2, 0.0, 0.0, np.pi / 2 - q4),
    ]


def human_arm_fk(q, l1, l2):
    """Forward kinematics of the human arm chain, rooted at the shoulder
    frame {0}. Returns the 4x4 pose of the wrist frame {4} w.r.t. {0}."""
    T = np.eye(4)
    for (a, alpha, d, theta) in human_arm_dh_params(q, l1, l2):
        T = T @ dh_transform(a, alpha, d, theta)
    return T


def human_arm_wrist_position(q, l1, l2):
    return human_arm_fk(q, l1, l2)[:3, 3]


def human_arm_points(q, l1, l2):
    """(shoulder, elbow, wrist) Cartesian positions, each (3,), IN THE
    SHOULDER FRAME {0} -- so the shoulder is the origin (0, 0, 0) by
    construction. For logging and for cross-checking the visuo-tactile
    pipeline's own q against its own Cartesian estimates.

    This does NOT capture how the shoulder itself moves in the room:
    for that, record the pipeline's shoulder/elbow/wrist points in its
    fixed (camera or robot-base) frame, which is a separate signal.
    """
    q = np.asarray(q, dtype=float)
    T = np.eye(4)
    elbow = None
    for i, (a, alpha, d, theta) in enumerate(human_arm_dh_params(q, l1, l2)):
        T = T @ dh_transform(a, alpha, d, theta)
        if i == 2:                       # after the 3rd joint -> elbow
            elbow = T[:3, 3].copy()
    return np.zeros(3), elbow, T[:3, 3].copy()


def human_arm_jacobian(q, l1, l2, eps=1e-6):
    """Numerical (finite-difference) 3x4 positional Jacobian of the wrist
    position w.r.t. the 4 human joint angles. A numerical Jacobian is used
    on purpose: it keeps this module independent from a symbolic/KDL
    dependency and is cheap enough (4 extra FK evaluations) to run inside
    a real-time control loop at typical admittance-control rates.
    """
    q = np.asarray(q, dtype=float)
    p0 = human_arm_wrist_position(q, l1, l2)
    J = np.zeros((3, N_HUMAN_JOINTS))
    for i in range(N_HUMAN_JOINTS):
        dq = np.zeros(N_HUMAN_JOINTS)
        dq[i] = eps
        p1 = human_arm_wrist_position(q + dq, l1, l2)
        J[:, i] = (p1 - p0) / eps
    return J


def cartesian_to_human_joint_velocity(q, l1, l2, v_cartesian, damping=1e-3, J=None,
                                       joint_limits=None, center_gain=0.0):
    """Map a candidate Cartesian velocity (3,) at the wrist to the
    corresponding human joint velocities using the damped
    least-squares pseudo-inverse of the arm Jacobian (robust near the
    fully-extended/flexed postures where the Jacobian may lose rank).

    Redundancy: the arm has 4 joints for 3 Cartesian position
    constraints, so J has a 1-dimensional null space in general
    configurations (the "elbow swivel" self-motion familiar from human-arm
    biomechanics) -- the same wrist position/velocity is reachable by a
    whole family of joint configurations/velocities, e.g. q3 (shoulder
    internal/external rotation) can sit near or far from its limit for
    the SAME wrist target, depending purely on which point of that
    family the arm is at.

    By default (`center_gain=0`, `joint_limits=None`), this function
    returns the MINIMUM-NORM solution (Levenberg-Marquardt / damped
    Jacobian-transpose), which implicitly sets the null-space component
    to zero rather than exploiting it for any secondary objective. This
    is the mode used by performance.joint_safety_factor() to evaluate a
    CANDIDATE command's effect: it deliberately does not assume any
    particular redundancy-resolution strategy for the real human (whose
    actual neuromuscular resolution of this redundancy is not modeled),
    so minimum-norm is used as a neutral, non-committal estimate.

    If `joint_limits` is given and `center_gain>0`, a secondary,
    null-space-projected task is added on top of the minimum-norm
    solution, biasing the redundant degree of freedom towards the
    center of each joint's range (Liegeois' gradient-projection method):

        qdot = qdot_particular + N @ qdot_secondary
        N                = I - J^+ J                  (null-space projector)
        qdot_secondary_i = -center_gain * rho_i        (pulls q_i towards mid-range)

    where rho_i in [-1,1] is joint i's normalized position relative to
    its limits. This does not change the resulting Cartesian velocity
    (N @ qdot_secondary lies in the null space of J by construction) --
    it only decides where the redundant degree of freedom is spent. Not
    used for the real safety-factor candidate evaluation, for the reason
    above; kept for parity with the offline simulator's synthetic-human
    model, in case this package is later reused to drive one.

    Performance note: the finite-difference human_arm_jacobian(q, l1, l2)
    call is the single most expensive operation in the shared-control
    loop (~70 us). It depends only on (q, l1, l2), NOT on v_cartesian,
    so a caller evaluating several candidate commands for the SAME
    joint state within one control cycle (as shared_control_core.py
    does, for v_h, v_r, and the raw blend) should compute it ONCE and
    pass it in via `J=` instead of letting each call recompute it.
    """
    if J is None:
        J = human_arm_jacobian(q, l1, l2)
    JJt = J @ J.T
    damped_pinv = J.T @ np.linalg.inv(JJt + (damping ** 2) * np.eye(3))  # (4,3)
    qdot = damped_pinv @ np.asarray(v_cartesian, dtype=float)

    if joint_limits is not None and center_gain > 0.0:
        q = np.asarray(q, dtype=float)
        q_mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
        q_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
        rho = (q - q_mid) / q_half_range
        qdot_secondary = -center_gain * rho
        N = np.eye(N_HUMAN_JOINTS) - damped_pinv @ J
        qdot = qdot + N @ qdot_secondary

    return qdot
