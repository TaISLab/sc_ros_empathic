"""
performance.py
--------------
Local performance factors for the reactive shared-control law.

Ported verbatim (logic unchanged) from the offline simulation/
verification package used to develop and tune this law; only the
import of dh_utils was made package-relative for ROS packaging.

Extends the two factors used in the original reactive AAN framework
(smoothness, directness) with two NEW factors that are the core
novelty of this work:

  * Human joint-limit safety factor (eta_safety): penalizes candidate
    commands that would drive the human's shoulder/elbow joints
    (q1..q4, estimated in real time by the visuo-tactile pipeline)
    closer to their physiological limits.

  * Robot manipulability / singularity-avoidance factor (eta_manip):
    penalizes candidate commands that would reduce the FR3
    manipulability index (Yoshikawa, 1985), i.e. that steer the robot
    towards a kinematic singularity.

Every candidate command k in {h, r, s} (human, robot, blend) receives
its own performance eta_k in [0, 1], obtained as a weighted average of
the four partial factors:

    eta_k = sum_i(w_i * eta_k_i) / sum_i(w_i)

which is then used to combine vh and vr into the shared velocity
command (unchanged from the original framework):

    v_hat_s = eta_r * v_r + eta_h * v_h
    v_s     = eta_s * v_hat_s
"""

import numpy as np

from .dh_utils import cartesian_to_human_joint_velocity


# ---------------------------------------------------------------------
# Factor 1 & 2: smoothness and directness (as in the reactive AAN paper)
# ---------------------------------------------------------------------

def smoothness_factor(v_candidate, v_prev, C1=1.0, eps=1e-9):
    """eta_k1 = exp(-C1 * |angle(v_candidate, v_prev)|)"""
    angle = _angle_between(v_candidate, v_prev, eps)
    return float(np.exp(-C1 * abs(angle)))


def directness_factor(v_candidate, tangent_vec, C2=1.0, eps=1e-9):
    """eta_k2 = exp(-C2 * |angle(v_candidate, path_tangent)|)"""
    angle = _angle_between(v_candidate, tangent_vec, eps)
    return float(np.exp(-C2 * abs(angle)))


def _angle_between(v1, v2, eps=1e-9):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_a)


# ---------------------------------------------------------------------
# Factor 3 (NEW): human joint-limit safety
# ---------------------------------------------------------------------

# Default anatomical ranges (radians), configurable via ROS params.
# Values are illustrative mid-range goniometric limits for q1..q4:
#   q1: shoulder flexion/extension, q2: shoulder abd/add,
#   q3: shoulder internal/external rotation, q4: elbow flexion/extension.
DEFAULT_JOINT_LIMITS = np.array([
    [-1.05, 3.14],   # q1 [-60 deg, 180 deg]
    [0.0, 3.14],     # q2 [0 deg, 180 deg]
    [-1.57, 1.57],   # q3 [-90 deg, 90 deg]
    [0.0, 2.53],     # q4 [0 deg, 145 deg]
])


def joint_safety_factor(q_human, v_candidate, l1, l2,
                         joint_limits=None, Cs=1.0, margin_floor=0.05,
                         static_weight=0.5, dynamic_weight=0.5,
                         proximity_threshold=0.3, J_arm=None):
    """eta_k3: penalizes candidate Cartesian commands that push q1..q4
    towards their limits, weighted by how close each joint already is.

    The penalty combines two terms, since the 4-DoF human-arm Jacobian
    can have near-zero-norm columns for some joints at some postures
    (e.g. a pure rotation about a link's own axis does not instantly
    translate the wrist), which would otherwise make the safety factor
    blind to a joint that is already at risk:

      (a) STATIC proximity term: grows as any joint's margin to its
          limit shrinks below `proximity_threshold`, regardless of the
          candidate command (an artificial-potential-style term).
      (b) DYNAMIC closing-rate term: grows when the candidate command,
          mapped to joint velocities via the arm Jacobian, actively
          drives a joint further towards the limit it is closest to --
          BUT only once that joint is inside the proximity band. A joint
          sitting comfortably mid-range is not a joint-limit-safety
          concern however briskly the command moves it, so both terms
          are gated by the same `prox_w` (without this gate the dynamic
          term fires on any motion, anywhere in the range, and near a
          Jacobian singularity it drives eta_k3 to 0 far from any limit).

    For every joint i:
      rho_i    = normalized position in [-1, 1] (0 = mid-range)
      margin_i = 1 - |rho_i|                      (shrinks near a limit)
      prox_w_i = clip((proximity_threshold - margin_i)/proximity_threshold, 0, 1)
      qdot_i   = joint velocity induced by v_candidate (via arm Jacobian)
      closing_rate_i = qdot_i if moving towards the limit else 0

    penalty = static_weight  * sum_i( proximity_threshold * prox_w_i )
            + dynamic_weight * sum_i( prox_w_i * max(0, closing_rate_i)
                                      / max(margin_i, margin_floor) )
    eta_k3  = exp(-Cs * penalty)
    """
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS

    q_human = np.asarray(q_human, dtype=float)
    qdot = cartesian_to_human_joint_velocity(q_human, l1, l2, v_candidate, J=J_arm)

    static_penalty = 0.0
    dynamic_penalty = 0.0
    for i, (q_min, q_max) in enumerate(joint_limits):
        q_mid = 0.5 * (q_max + q_min)
        q_half_range = 0.5 * (q_max - q_min)
        rho = (q_human[i] - q_mid) / q_half_range
        margin = 1.0 - abs(rho)

        # 0 while the joint is within (1 - tau) of mid-range, ramping to
        # 1 at the limit. Gates BOTH terms.
        prox_w = min(max((proximity_threshold - margin) / proximity_threshold,
                         0.0), 1.0)
        static_penalty += proximity_threshold * prox_w

        # Positive when qdot moves the joint further in the direction
        # it is already displaced from the mid-range (i.e. towards
        # whichever limit it is closest to).
        closing_rate = qdot[i] * np.sign(rho) if rho != 0 else 0.0
        dynamic_penalty += (prox_w * max(closing_rate, 0.0)
                            / max(margin, margin_floor))

    penalty = static_weight * static_penalty + dynamic_weight * dynamic_penalty
    return float(np.exp(-Cs * penalty))


# ---------------------------------------------------------------------
# Factor 4 (NEW): robot manipulability / singularity avoidance
# ---------------------------------------------------------------------

def manipulability_index(J):
    """Yoshikawa manipulability index w(q) = sqrt(det(J J^T))."""
    JJt = J @ J.T
    det = np.linalg.det(JJt)
    return float(np.sqrt(max(det, 0.0)))


def manipulability_factor(J_current, J_predicted, Cm=1.0):
    """eta_k4: rewards candidate commands that increase (or do not
    reduce) the robot's manipulability index, penalizing commands that
    drive the arm towards a singular configuration.

    J_predicted should be the robot Jacobian evaluated at the
    configuration q + qdot_candidate * dt (obtained externally, via
    PyKDL/URDF -- see robot_model.RobotModel).
    """
    w_now = manipulability_index(J_current)
    w_next = manipulability_index(J_predicted)
    delta_w = w_next - w_now
    # Only penalize degradation; do not reward moving away from
    # singularities beyond what is needed (keeps the factor in (0,1]).
    penalty = max(-delta_w, 0.0) / max(w_now, 1e-6)
    return float(np.exp(-Cm * penalty))


# ---------------------------------------------------------------------
# Weighted combination (generalized to m=4)
# ---------------------------------------------------------------------

def total_performance(factors, weights=None):
    """factors: dict with keys a subset of
       {'smoothness', 'directness', 'joint_safety', 'manipulability'}.
       weights: optional dict with the same keys (defaults to 1.0 each)."""
    if weights is None:
        weights = {k: 1.0 for k in factors}
    num = sum(weights.get(k, 1.0) * v for k, v in factors.items())
    den = sum(weights.get(k, 1.0) for k in factors)
    return num / den if den > 0 else 0.0
