"""
shared_control_core.py
------------------------
ROS-independent core of the shared-control law: candidate performance
evaluation + combination into the emergent velocity command. This is
the SAME code (module-for-module port, logic unchanged) used in the
offline Monte-Carlo simulation/verification package this control law
was tuned and validated in, so that anything validated there is
guaranteed to behave identically once deployed here.

Only the imports were made package-relative for ROS packaging; no
control-law logic was changed.
"""

import numpy as np

from . import performance as perf
from .dh_utils import human_arm_jacobian
from .robot_model import RobotModel  # noqa: F401  (re-exported for callers)


class SharedControlCore(object):
    def __init__(self, dt_lookahead=0.05, v_max=0.15, lpf_alpha=0.2,
                 weights=None, C1=1.0, C2=1.0, Cs=1.0, Cm=1.0,
                 robot_model=None, joint_limits=None,
                 proximity_threshold=0.3):
        self.dt_lookahead = dt_lookahead
        self.v_max = v_max
        self.alpha_lpf = lpf_alpha
        self.weights = weights or {"smoothness": 1.0, "directness": 1.0,
                                    "joint_safety": 1.0, "manipulability": 1.0}
        self.C1, self.C2, self.Cs, self.Cm = C1, C2, Cs, Cm
        self.robot_model = robot_model  # optional; None disables eta_manipulability
        # Pre-registered per-participant human joint limits [q_min, q_max]
        # (4x2) and proximity threshold tau for the joint-safety factor.
        # None -> performance.joint_safety_factor uses its documented
        # defaults; passing them here does not change the factor's logic,
        # only which limits/threshold it scores against.
        self.joint_limits = joint_limits
        self.proximity_threshold = proximity_threshold

        self.v_prev = np.zeros(3)
        self.v_s_filtered = np.zeros(3)

    def candidate_performance(self, v_candidate, tangent, q_human=None, l1=None,
                               l2=None, q_robot=None, J_robot=None,
                               active_factors=("smoothness", "directness",
                                               "joint_safety", "manipulability"),
                               J_arm_human=None):
        factors = {}
        if "smoothness" in active_factors:
            factors["smoothness"] = perf.smoothness_factor(v_candidate, self.v_prev, self.C1)
        if "directness" in active_factors:
            factors["directness"] = perf.directness_factor(v_candidate, tangent, self.C2)
        if "joint_safety" in active_factors and q_human is not None:
            factors["joint_safety"] = perf.joint_safety_factor(
                q_human, v_candidate, l1, l2, Cs=self.Cs, J_arm=J_arm_human,
                joint_limits=self.joint_limits,
                proximity_threshold=self.proximity_threshold)
        if ("manipulability" in active_factors and J_robot is not None
                and q_robot is not None and self.robot_model is not None):
            qdot = np.linalg.pinv(J_robot[:3, :]) @ v_candidate
            J_pred = self.robot_model.predict_jacobian(q_robot, qdot, self.dt_lookahead)
            factors["manipulability"] = perf.manipulability_factor(
                J_robot, J_pred, Cm=self.Cm)
        return perf.total_performance(factors, self.weights), factors

    def step(self, v_h, v_r, tangent, q_human=None, l1=None, l2=None,
              q_robot=None, J_robot=None, active_factors=None):
        active_factors = active_factors or (
            "smoothness", "directness", "joint_safety", "manipulability")

        # The human-arm Jacobian depends only on (q_human, l1, l2), not on
        # the candidate command, so it is computed ONCE per control cycle
        # and reused across the three candidate evaluations below (v_h,
        # v_r, and the raw blend) instead of being recomputed three times.
        # This is the single most expensive operation in the loop
        # (~70 us); reusing it roughly cuts the extended controller's
        # per-cycle cost by half (see the offline benchmark this was
        # profiled with).
        J_arm_human = None
        if "joint_safety" in active_factors and q_human is not None:
            J_arm_human = human_arm_jacobian(q_human, l1, l2)

        eta_h, factors_h = self.candidate_performance(
            v_h, tangent, q_human, l1, l2, q_robot, J_robot, active_factors, J_arm_human)
        eta_r, factors_r = self.candidate_performance(
            v_r, tangent, q_human, l1, l2, q_robot, J_robot, active_factors, J_arm_human)

        v_hat_s = eta_r * v_r + eta_h * v_h
        eta_s, factors_s = self.candidate_performance(
            v_hat_s, tangent, q_human, l1, l2, q_robot, J_robot, active_factors, J_arm_human)
        v_s = eta_s * v_hat_s

        self.v_s_filtered = (self.alpha_lpf * v_s
                              + (1 - self.alpha_lpf) * self.v_s_filtered)
        speed = np.linalg.norm(self.v_s_filtered)
        v_out = self.v_s_filtered if speed <= self.v_max else (
            self.v_s_filtered * self.v_max / max(speed, 1e-9))

        self.v_prev = v_out
        return v_out, {"eta_h": eta_h, "eta_r": eta_r, "eta_s": eta_s,
                        "factors_h": factors_h, "factors_r": factors_r,
                        "v_hat_s": v_hat_s}
