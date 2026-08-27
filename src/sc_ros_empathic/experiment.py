"""
experiment.py
-------------
ROS-independent helpers for running the human-subject experimental
protocol of the paper (Section V) on top of the shared-control law.

Nothing here changes the control law. It only provides:

  * CONDITIONS: the mapping from an experimental control mode
    (paper Sec. V-B: A standalone, B baseline m=2, C joint-safety m=3,
    D manipulability m=3, E extended m=4) to the set of local
    performance factors that must be active for that condition and
    whether the robot path-following command is used at all.
    Condition F (the impedance-control AAN baseline of Zhang et al.
    [9]) is a different controller and is NOT produced by this law --
    see scripts/baseline_aan_node.py.

  * LapCounter: turns the monotone-ish path parameter s in [0, 1)
    reported by the reactive path follower into an integer lap index,
    so offline analysis can drop the first loop (used as training) and
    segment the remaining loops per trial, exactly as specified in the
    paper ("each trial consists of four loops, the first discarded as
    training").

  * joint_margins: the per-joint normalized margin m_i = 1 - |rho_i|
    that the paper reports as a metric ("minimum joint margin
    min_i m_i(t)" and "time-below-threshold"). This is the SAME rho_i
    definition used inside performance.joint_safety_factor; it is
    re-exposed here as a plain observable so the node can log it
    without reaching into the factor's internals.
"""

import numpy as np

from .performance import DEFAULT_JOINT_LIMITS


# ---------------------------------------------------------------------
# Experimental conditions (paper Sec. V-B)
# ---------------------------------------------------------------------
#
# use_robot_command:
#   True  -> the reactive path-following command v_r participates in the
#            performance-weighted blend (v_hat_s = eta_r v_r + eta_h v_h).
#   False -> "standalone" / no assistance: only the human's
#            admittance-derived command is shaped and sent; the robot
#            does not drive the limb along the path.
#
# factors:
#   the local performance factors that must be active for this
#   condition. smoothness + directness are always present for any
#   assisted condition (they define the baseline reactive law of [1]);
#   joint_safety and manipulability are the two factors this paper adds.
#
# NOTE: at run time the node intersects `factors` with what the sensors
# can actually support that cycle (fresh human-arm state for
# joint_safety, a working KDL model for manipulability) and warns on a
# downgrade -- a condition never runs with a stale q_h or a missing
# Jacobian silently substituted.
CONDITIONS = {
    "A_standalone": dict(
        use_robot_command=False,
        factors=(),
        description="No assistance: shaped human admittance command only.",
    ),
    "B_baseline_m2": dict(
        use_robot_command=True,
        factors=("smoothness", "directness"),
        description="Baseline reactive shared control of Ruiz-Ruiz et al. [1] (m=2).",
    ),
    "C_jointsafety_m3": dict(
        use_robot_command=True,
        factors=("smoothness", "directness", "joint_safety"),
        description="Ablation: baseline + human joint-limit safety factor only (m=3).",
    ),
    "D_manip_m3": dict(
        use_robot_command=True,
        factors=("smoothness", "directness", "manipulability"),
        description="Ablation: baseline + robot manipulability factor only (m=3).",
    ),
    "E_extended_m4": dict(
        use_robot_command=True,
        factors=("smoothness", "directness", "joint_safety", "manipulability"),
        description="Proposed extended shared control: all four factors (m=4).",
    ),
}

# Condition F is intentionally absent: it is the impedance-control
# assist-as-needed baseline of Zhang et al. [9], a separate controller
# used only in the nominal placement to anchor comparability. It is not
# an instance of this performance-weighted law. See
# scripts/baseline_aan_node.py.
CONDITION_F_ID = "F_impedance_aan"


def resolve_condition(condition_id):
    """Return the CONDITIONS entry for `condition_id`, raising a clear
    error (listing the valid ids) on a typo -- the experimental
    condition must never silently fall back to a default."""
    try:
        return CONDITIONS[condition_id]
    except KeyError:
        valid = ", ".join(sorted(CONDITIONS)) + ", " + CONDITION_F_ID
        raise ValueError(
            "unknown experimental condition %r; valid ids: %s"
            % (condition_id, valid))


# ---------------------------------------------------------------------
# Lap counting for trial segmentation (paper Sec. V-B)
# ---------------------------------------------------------------------

class LapCounter(object):
    """Accumulate an integer lap index from the closed-path parameter
    s in [0, 1) reported each control cycle.

    The circle-tracing task is a closed path, so s wraps 1 -> 0 once
    per completed loop. A forward wrap (s jumps down by more than
    `wrap_threshold`, e.g. 0.98 -> 0.03) increments the lap; a backward
    wrap increments it back down, so brief reversals near the seam do
    not inflate the count. Small forward/backward jitter that does not
    cross the seam never changes the lap.

    Usage:
        lc = LapCounter()
        lc.update(s)            # every cycle
        lc.lap                  # 0 during the first loop, 1 during the
                                # second, ... -> offline analysis keeps
                                # laps >= 1 (lap 0 is the training loop)
        lc.total_progress       # lap + s, a continuous progress signal
    """

    def __init__(self, wrap_threshold=0.5):
        self.wrap_threshold = float(wrap_threshold)
        self.lap = 0
        self._s_prev = None

    def reset(self):
        self.lap = 0
        self._s_prev = None

    def update(self, s):
        s = float(s) % 1.0
        if self._s_prev is not None:
            delta = s - self._s_prev
            if delta < -self.wrap_threshold:
                self.lap += 1
            elif delta > self.wrap_threshold:
                self.lap -= 1
        self._s_prev = s
        return self.lap

    @property
    def total_progress(self):
        if self._s_prev is None:
            return float(self.lap)
        return self.lap + self._s_prev


# ---------------------------------------------------------------------
# Joint-margin observable (paper Sec. V-D metric)
# ---------------------------------------------------------------------

def joint_margins(q_human, joint_limits=None):
    """Per-joint normalized margin m_i = 1 - |rho_i|, with
    rho_i = (q_i - q_mid_i) / (0.5 (q_max_i - q_min_i)) in [-1, 1]
    (0 at mid-range, +-1 at a limit).

    Returns (margins, min_margin) where `margins` is a length-4 array.
    m_i can go slightly negative if a joint estimate is momentarily
    outside its registered range; this is reported as-is (not clipped)
    so the "time-below-threshold" metric stays honest.
    """
    if joint_limits is None:
        joint_limits = DEFAULT_JOINT_LIMITS
    joint_limits = np.asarray(joint_limits, dtype=float)
    q = np.asarray(q_human, dtype=float)

    q_mid = 0.5 * (joint_limits[:, 0] + joint_limits[:, 1])
    q_half_range = 0.5 * (joint_limits[:, 1] - joint_limits[:, 0])
    rho = (q - q_mid) / q_half_range
    margins = 1.0 - np.abs(rho)
    return margins, float(np.min(margins))
