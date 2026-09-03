#!/usr/bin/env python3
"""
shared_control_node.py
-----------------------
ROS1 node implementing the reactive, performance-weighted shared
control law (human joint-limit safety + robot manipulability factors)
on top of the FR3's external Cartesian velocity controller, wired for
the human-subject experimental protocol of the paper (Section V).

Reads:
  * /franka_state_controller/franka_states (franka_msgs/FrankaState)
        -> O_F_ext_hat_K (interaction force/torque at the EE, base
           frame) drives the admittance model that produces v_h;
           O_T_EE (EE pose) and q (joint angles) give the current
           Cartesian position x and robot configuration q_robot.
  * human arm joint state topic (q_h = q1..q4), segment lengths
    (l1, l2), and shoulder/elbow/wrist Cartesian points (in the
    pipeline's fixed frame -- the signal that captures how the
    shoulder itself moves), all published by the visuo-tactile
    pipeline.
        !!! TOPIC NAMES/MESSAGE TYPES BELOW ARE PLACEHOLDERS !!!
        Confirm the real ones with the visuo-tactile pipeline owner
        and override via the ROS params below -- no code change
        needed once confirmed, just launch-file arguments.
The robot Jacobian for the manipulability factor is computed
analytically from q by fr3_model.py (~jacobian_source=analytic, the
default) -- self-contained, no PyKDL, independent of the velocity
controller. ~jacobian_source=kdl (PyKDL) or =topic (a controller that
publishes its own 6xN Jacobian as std_msgs/Float64MultiArray on
~robot_jacobian_topic) are alternatives; =none disables the factor.

Experimental condition (~condition, paper Sec. V-B):
  A_standalone      no assistance -- shaped human admittance command only
  B_baseline_m2     baseline reactive SC of [1] (smoothness + directness)
  C_jointsafety_m3  baseline + human joint-limit safety factor  (ablation)
  D_manip_m3        baseline + robot manipulability factor       (ablation)
  E_extended_m4     proposed extended SC: all four factors
  F_impedance_aan   NOT this node -- see scripts/baseline_aan_node.py
The requested factor set is intersected each cycle with what the
sensors can support (fresh q_h for joint_safety, a working KDL model
for manipulability); a downgrade is logged, never silent.

Publishes:
  * <cmd_topic> (geometry_msgs/TwistStamped): the emergent Cartesian
    velocity v_s, consumed by the FR3's external Cartesian velocity
    controller (taislab_franka_controllers /
    cartesian_velocity_external_controller).
  * ~eta (std_msgs/Float64MultiArray): [eta_h, eta_r, eta_s].
  * ~diag/* : per-cycle observables for the offline metric analysis of
    Sec. V-D (per-factor eta contributions, per-joint margins m_i,
    FR3 manipulability w(q_r), path progress / lap index / cross-track
    error, v_h, v_r, v_s, filtered interaction force, and the
    shoulder/elbow/wrist points -- ~diag/arm_points from the pipeline
    and ~diag/arm_points_fk from FK on (q_h, l1, l2), poses[0..2] =
    shoulder, elbow, wrist). All plain std_msgs / geometry_msgs types
    -- record with `rosbag record`.

IMPORTANT -- rate honesty: this node runs its Python loop at
~`~rate_hz` (default 200 Hz), NOT the 1 kHz used in the offline
per-cycle cost benchmark. That benchmark measures the cost of ONE
call to SharedControlCore.step() (the actual control-law computation,
which comfortably fits inside a 1 kHz budget on the target hardware),
not the achievable rate of a plain rospy publisher loop, which is
subject to the Python GIL, OS scheduling jitter and non-realtime
message passing and should not be assumed to hold 1 kHz reliably. The
real 1 kHz realtime loop lives in the C++ external velocity
controller, which zero-order-holds the last command received from
this node between updates -- exactly the architecture
cartesian_velocity_external_controller is built for. Re-benchmark
`~rate_hz` achievable on your control PC before trusting it blindly;
do not assume it silently matches the paper's per-cycle timing figures.
"""

import csv
import os
import time

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Pose, PoseArray, TwistStamped, Vector3Stamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Float64MultiArray, String
from std_srvs.srv import Empty, EmptyResponse

from sc_ros_empathic.shared_control_core import SharedControlCore
from sc_ros_empathic.robot_model import RobotModel
from sc_ros_empathic.path_follower import CirclePath, ReactivePathFollower
from sc_ros_empathic.performance import manipulability_index
from sc_ros_empathic.dh_utils import (human_arm_points, human_arm_jacobian,
                                       cartesian_to_human_joint_velocity)
from sc_ros_empathic.experiment import (
    resolve_condition, LapCounter, joint_margins, joint_rho, CONDITION_F_ID)
from sc_ros_empathic.performance import DEFAULT_JOINT_LIMITS
from sc_ros_empathic.subject_config import SubjectConfig, SubjectConfigError


# Fixed slot order for the ~diag/factors_* arrays (NaN = factor not
# active this cycle / for this candidate).
FACTOR_SLOTS = ("smoothness", "directness", "joint_safety", "manipulability")


def _factors_to_array(factors):
    return [float(factors.get(k, np.nan)) for k in FACTOR_SLOTS]


class SharedControlNode(object):
    def __init__(self):
        rospy.init_node('shared_control_node', anonymous=False)

        # ------------------------------------------------------------
        # Experimental condition (paper Sec. V-B). No silent default:
        # resolve_condition() raises with the valid id list on a typo.
        # ------------------------------------------------------------
        self.condition_id = rospy.get_param('~condition', 'E_extended_m4')
        if self.condition_id == CONDITION_F_ID:
            rospy.logfatal(
                'shared_control_node: condition %s (impedance-control AAN '
                'baseline of Zhang et al. [9]) is a different controller '
                'and is not produced by this node. Run '
                'scripts/baseline_aan_node.py instead.', CONDITION_F_ID)
            raise rospy.ROSInitException('condition F is not this node')
        try:
            cond = resolve_condition(self.condition_id)
        except ValueError as e:
            rospy.logfatal('shared_control_node: %s', e)
            raise rospy.ROSInitException(str(e))
        self.cond_use_robot = cond['use_robot_command']
        self.cond_factors = tuple(cond['factors'])
        rospy.loginfo('shared_control_node: condition %s -- %s',
                      self.condition_id, cond['description'])

        # ------------------------------------------------------------
        # Per-volunteer config (intrinsic human parameters ONLY: id,
        # arm segment lengths, pre-registered joint ranges). Every
        # experiment parameter is homogeneous across volunteers and
        # comes from config/shared_control.yaml, not from here.
        # ~subject_file empty -> run from plain params (back-compat).
        # ------------------------------------------------------------
        self.subject = None
        subject_file = rospy.get_param('~subject_file', '')
        if subject_file:
            try:
                self.subject = SubjectConfig(subject_file)
            except SubjectConfigError as e:
                rospy.logfatal('shared_control_node: %s', e)
                raise rospy.ROSInitException(str(e))
            rospy.loginfo('shared_control_node: %s', self.subject.summary())
            if self.subject.dominant_arm and self.subject.dominant_arm != 'right':
                rospy.logwarn('shared_control_node: subject %s dominant_arm=%r; '
                              'the visuo-tactile pipeline tracks the RIGHT arm '
                              'only (paper inclusion criterion).',
                              self.subject.subject_id, self.subject.dominant_arm)

        # ------------------------------------------------------------
        # Topics -- CHECK THESE before every session, same discipline
        # as the rest of this codebase: a wrong topic name here does
        # NOT fail loudly, it just means q_h/l1/l2 never arrive and the
        # node silently runs with joint_safety disabled (see the
        # staleness fail-safe in _human_state_fresh()).
        # ------------------------------------------------------------
        self.franka_states_topic = rospy.get_param(
            '~franka_states_topic', '/franka_state_controller/franka_states')
        self.cmd_topic = rospy.get_param(
            '~cmd_topic', '/robot_vel_ctrl/vel_cmd')
        # Visuo-tactile pipeline, layout confirmed 2026-09 with the
        # pipeline owner: sensor_msgs/JointState with a NAMED 7-entry
        # chain -- right_arm_q1, right_arm_q2, upperarm_length,
        # right_arm_q3, right_arm_q4, forearm_length, right_arm_q5.
        # The two *_length entries are prismatic "joints" carrying the
        # LIVE bone-length estimates l1/l2 (m); right_arm_q5 (wrist
        # pronation) is outside the 4-DoF safety model. We index q1..q4
        # and l1,l2 BY NAME (see _human_joint_state_cb) -- a positional
        # slice would feed upperarm_length in as q3 and drop the elbow.
        self.human_joint_state_topic = rospy.get_param(
            '~human_joint_state_topic', '/right_arm/joint_states')
        # Optional SEPARATE l1,l2 stream (std_msgs/Float64MultiArray,
        # data = [l1, l2] m). Not needed with the pipeline above (l1,l2
        # ride in ~human_joint_state_topic); kept for other rigs / as an
        # override. Silent if the topic never publishes.
        self.human_link_lengths_topic = rospy.get_param(
            '~human_link_lengths_topic', '/right_arm/link_lengths')
        # PLACEHOLDER -- confirm with the visuo-tactile pipeline owner.
        # The shoulder/elbow/wrist Cartesian points in the pipeline's
        # FIXED frame (camera or robot base) -- the signal that captures
        # how the shoulder itself moves during the trial, which the
        # shoulder-frame FK cannot. Expected here:
        # std_msgs/Float64MultiArray, data = [sx,sy,sz, ex,ey,ez, wx,wy,wz].
        # If the pipeline uses another type, the raw topic is still
        # recorded by the launch's rosbag regardless; only the
        # ~diag/arm_points re-publish below needs the subscriber updated.
        self.human_arm_points_topic = rospy.get_param(
            '~human_arm_points_topic', '/right_arm/arm_points')
        self.human_points_frame = rospy.get_param(
            '~human_points_frame', 'fr3_link0')  # set to the pipeline's frame
        # ~diag/arm_points_fk (FK of q_h,l1,l2; shoulder at the origin)
        # is published in this frame -- set it to the pipeline's shoulder
        # TF frame so RViz anchors the FK arm at the real shoulder.
        self.arm_fk_frame = rospy.get_param(
            '~human_shoulder_frame', 'base_shoulder')
        self.eta_topic = rospy.get_param('~eta_topic', '~eta')

        self.base_frame = rospy.get_param('~base_frame', 'fr3_link0')

        # ------------------------------------------------------------
        # Path (nominal/stressed placement -- see paper Sec. 5.1): a
        # circle in the robot base frame. Override via ROS params /
        # launch args per condition; do NOT hardcode per-session values
        # here.
        # ------------------------------------------------------------
        center = rospy.get_param('~path_center', [0.45, 0.0, 0.45])
        radius = rospy.get_param('~path_radius', 0.05)
        normal = rospy.get_param('~path_normal', [0.0, 0.0, 1.0])
        Ka = rospy.get_param('~Ka', 1.0)
        # rho_min: pure-pursuit lookahead when on the path -> the
        # traversal speed around the circle is ~Ka*rho_min + cruise_speed.
        # cruise_speed: tangential feed-forward so the robot keeps
        # advancing along the path even at near-zero cross-track error
        # (a purely proportional v_r can stall there).
        rho_min = rospy.get_param('~rho_min', 0.02)
        lam = rospy.get_param('~lam', 1.02)
        cruise_speed = rospy.get_param('~cruise_speed', 0.03)
        # 'crosstrack' = pull to the nearest reference point (tracks the
        # reference radius); 'lookahead' = the [1] virtual-sphere law
        # (cuts corners under loop lag -> traced circle smaller than the
        # reference).
        follower_mode = rospy.get_param('~follower_mode', 'crosstrack')
        # 'forward' = trace the circle in the path's native sense;
        # 'reverse' = the other way round.
        pd = str(rospy.get_param('~path_direction', 'forward')).lower()
        if pd not in ('forward', 'reverse'):
            rospy.logwarn("shared_control_node: ~path_direction=%r, expected "
                          "'forward' or 'reverse'; using 'forward'.", pd)
        self.path_dir = -1 if pd == 'reverse' else 1
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        self.follower = ReactivePathFollower(
            self.path, Ka=Ka, rho_min=rho_min, lam=lam,
            cruise_speed=cruise_speed, mode=follower_mode,
            direction=self.path_dir)
        self.lap_counter = LapCounter(direction=self.path_dir)
        # Trial ends after this many laps (0 = run until Ctrl-C). The
        # paper's trial is 4 loops, the first discarded as training.
        # ~trial_end = 'shutdown' (default): stop the node (with
        # required="true" in the launch this tears the whole launch
        # down and closes the bag/CSV); 'hold': keep the node alive at
        # zero velocity, operator ends the session.
        self.trial_laps = int(rospy.get_param('~trial_laps', 0))
        self.trial_end = str(rospy.get_param('~trial_end', 'shutdown')).lower()
        self._trial_done = False
        self._trial_done_t = None

        # If the condition needs joint_safety (C, E) and no fresh q_h +
        # l1,l2 ever arrive, the factor is silently dropped every cycle
        # -> a whole session recorded with joint_safety_h / m* all NaN,
        # which is an INVALID trial for those conditions. With
        # ~require_fresh_human:=true the node holds ZERO velocity (and
        # logs an error) until the human state is fresh, so you cannot
        # record such a session unnoticed.
        self.require_fresh_human = bool(
            rospy.get_param('~require_fresh_human', False))
        self._human_ever_fresh = False

        # ------------------------------------------------------------
        # Admittance model for v_h (force -> human-intent velocity).
        # This is the "admittance law already in place" referred to in
        # the paper; it lives HERE because this package is the first
        # place in this repo that reads O_F_ext_hat_K.
        # ------------------------------------------------------------
        self.M_h = np.array(rospy.get_param('~admittance_mass', [1.0, 1.0, 1.0]))
        self.B_h = np.array(rospy.get_param('~admittance_damping', [10.0, 10.0, 10.0]))
        self.deadzone_N = rospy.get_param('~force_deadzone_N', 1.0)
        self.alpha_f = rospy.get_param('~force_lpf_alpha', 0.1)
        self.v_h = np.zeros(3)
        self.f_filtered = np.zeros(3)

        # O_F_ext_hat_K is rarely exactly zero at rest (an EE payload /
        # handle not in the FR3 load model leaves a roughly constant
        # offset, mostly -z in the base frame). Un-taured, the admittance
        # turns it into a steady spurious v_h and the robot drifts /
        # limit-cycles against the path follower. Average the wrench for
        # ~force_tare_s at startup (HANDS OFF) and subtract it; 0 or a
        # negative value disables the tare. ~tare (std_srvs/Empty)
        # re-runs it mid-session.
        self.force_tare_s = float(rospy.get_param('~force_tare_s', 1.0))
        self.f_bias = np.zeros(3)
        self._tare_samples = []
        self._tare_t0 = None
        self._tared = self.force_tare_s <= 0.0

        # ------------------------------------------------------------
        # Shared-control law. Weight/gain defaults below are the ones
        # tuned in offline simulation (see the paper, Sec. 4);
        # re-tune on hardware before trusting them, they are a start
        # point, not a validated-on-robot result. Load config/shared_control.yaml
        # (via the launch file) to override them per condition.
        # ------------------------------------------------------------
        weights = {
            'smoothness': rospy.get_param('~w_smoothness', 1.0),
            'directness': rospy.get_param('~w_directness', 1.0),
            'joint_safety': rospy.get_param('~w_joint_safety', 16.0),
            'manipulability': rospy.get_param('~w_manipulability', 24.0),
        }
        self.v_max = rospy.get_param('~v_max', 0.15)
        self.lpf_alpha = rospy.get_param('~lpf_alpha', 0.2)

        # Pre-registered per-VOLUNTEER physiological joint ranges for the
        # joint-safety factor. Primary source: config/subjects/SXX.yaml
        # (joint_limits_rad). Override: ~human_joint_limits, a flat list
        # [q1min,q1max, q2min,q2max, q3min,q3max, q4min,q4max] (rad).
        # Neither -> performance.DEFAULT_JOINT_LIMITS.
        jl_flat = rospy.get_param('~human_joint_limits', None)
        self.human_joint_limits = None
        if jl_flat is not None:
            arr = np.asarray(jl_flat, dtype=float)
            if arr.size != 8:
                rospy.logfatal('shared_control_node: ~human_joint_limits must '
                               'have 8 values ([qi_min,qi_max]*4), got %d',
                               arr.size)
                raise rospy.ROSInitException('bad ~human_joint_limits')
            self.human_joint_limits = arr.reshape(4, 2)
            rospy.loginfo('shared_control_node: joint ranges from '
                          '~human_joint_limits (overriding subject file)')
        elif self.subject is not None and self.subject.has_joint_limits:
            self.human_joint_limits = self.subject.joint_limits
            rospy.loginfo('shared_control_node: joint ranges from subject %s',
                          self.subject.subject_id)
        else:
            rospy.loginfo('shared_control_node: joint ranges = study defaults '
                          '(performance.DEFAULT_JOINT_LIMITS)')

        # Affine per-joint calibration of the incoming q1..q4:
        #     q_model = gain * (q_pipeline - offset)
        # so the visuo-tactile pipeline's angle convention maps onto the
        # DH model / joint_limits (q=0 at the goniometric neutral, and
        # the model's own sign/scale). Properties of the PIPELINE, not
        # the volunteer -> config/shared_control.yaml. Affects the limits
        # AND the arm Jacobian (both use q).
        #   offset : the pipeline reading at the model's zero for that
        #            joint (e.g. q4 with the arm fully extended).
        #   gain   : sign/scale. From two calibration postures a,b with
        #            known model angles qa,qb:
        #            gain = (qb-qa)/(q_pipe_b - q_pipe_a),
        #            offset = q_pipe_a - qa/gain.
        # Defaults: offsets 0, gains 1 -> q unchanged.
        jo = np.asarray(rospy.get_param('~human_joint_offsets',
                                        [0.0, 0.0, 0.0, 0.0]), dtype=float)
        jg = np.asarray(rospy.get_param('~human_joint_gains',
                                        [1.0, 1.0, 1.0, 1.0]), dtype=float)
        if jo.size != 4 or jg.size != 4:
            rospy.logfatal('shared_control_node: ~human_joint_offsets and '
                           '~human_joint_gains must each have 4 values '
                           '(q1..q4); got %d, %d', jo.size, jg.size)
            raise rospy.ROSInitException('bad ~human_joint_offsets/gains')
        self.human_joint_offsets = jo
        self.human_joint_gains = jg
        if np.any(jo != 0.0) or np.any(jg != 1.0):
            rospy.loginfo('shared_control_node: q_model = gain*(q_pipeline - '
                          'offset); gain=%s offset(rad)=%s',
                          jg.tolist(), jo.tolist())

        # tau: the joint-margin threshold IS homogeneous across
        # volunteers -- it stays in config/shared_control.yaml.
        self.proximity_threshold = rospy.get_param('~proximity_threshold', 0.3)

        # Robot Jacobian for the manipulability factor.
        #   analytic (default) -- self-contained pure-numpy FR3 FK +
        #     Jacobian from q (fr3_model.py); per-candidate lookahead,
        #     no PyKDL, no dependency on the velocity controller.
        #   kdl   -- same, via PyKDL + /robot_description.
        #   topic -- J supplied on ~robot_jacobian_topic
        #     (std_msgs/Float64MultiArray, 6xN row-major); lookahead is
        #     only time-propagated, NOT per-candidate.
        #   none  -- disable the manipulability factor.
        base_link = rospy.get_param('~base_link', 'fr3_link0')
        ee_link = rospy.get_param('~ee_link', 'fr3_link8')
        self.jacobian_source = rospy.get_param('~jacobian_source', 'analytic')
        self.robot_jacobian_topic = rospy.get_param(
            '~robot_jacobian_topic', '/taislab_controller/jacobian')
        self.jac_rows = int(rospy.get_param('~robot_jacobian_rows', 6))
        self.jac_cols = int(rospy.get_param('~robot_jacobian_cols', 7))
        self.max_jacobian_age = rospy.get_param('~max_jacobian_age', 0.2)
        self.robot_jac_stamp = None

        robot_model = None
        if self.jacobian_source == 'none':
            rospy.loginfo('shared_control_node: ~jacobian_source=none -- '
                          'manipulability factor disabled.')
        elif self.jacobian_source == 'kdl':
            try:
                robot_model = RobotModel(backend='kdl', base_link=base_link,
                                         ee_link=ee_link)
            except RuntimeError as e:
                rospy.logerr('%s', e)
                rospy.logerr('shared_control_node: disabling the '
                              'manipulability factor for this run.')
        elif self.jacobian_source == 'topic':
            robot_model = RobotModel(backend='topic')
            rospy.loginfo('shared_control_node: robot Jacobian from %s '
                          '(%dx%d Float64MultiArray, row-major)',
                          self.robot_jacobian_topic, self.jac_rows,
                          self.jac_cols)
        else:  # 'analytic'
            robot_model = RobotModel(backend='analytic')
            rospy.loginfo('shared_control_node: robot Jacobian computed '
                          'analytically from q (fr3_model, no PyKDL).')
        self.manipulability_available = robot_model is not None

        if ('manipulability' in self.cond_factors
                and not self.manipulability_available):
            rospy.logerr('shared_control_node: condition %s REQUIRES the '
                          'manipulability factor but no robot Jacobian '
                          'source is available (~jacobian_source=%s) -- this '
                          'run will not be a valid %s trial.',
                          self.condition_id, self.jacobian_source,
                          self.condition_id)

        self.core = SharedControlCore(
            dt_lookahead=rospy.get_param('~dt_lookahead', 0.2),
            v_max=self.v_max,
            lpf_alpha=self.lpf_alpha,
            weights=weights,
            C1=rospy.get_param('~C1', 1.0),
            C2=rospy.get_param('~C2', 1.0),
            Cs=rospy.get_param('~Cs', 12.0),
            Cm=rospy.get_param('~Cm', 24.0),
            robot_model=robot_model,
            joint_limits=self.human_joint_limits,
            proximity_threshold=self.proximity_threshold,
        )

        # Standalone (condition A) command shaping: same LPF + speed
        # saturation the core applies to v_s, so A and the assisted
        # conditions differ only in whether the robot assists, not in
        # output smoothing / speed cap.
        self._standalone_filt = np.zeros(3)

        # ------------------------------------------------------------
        # Human-arm state (fail-safe: stale/missing -> drop
        # joint_safety from active_factors rather than run with a
        # frozen/garbage q_h, mirroring the Gamma-timeout fail-safe
        # pattern already used elsewhere in this lab's controllers).
        #
        # q_h is inherently dynamic -> topic only. l1/l2: the paper's
        # visuo-tactile pipeline publishes time-varying estimates on
        # ~human_link_lengths_topic; the sibling package
        # (sc_effort_experiment) instead uses hand-measured per-subject
        # l1/l2. Static fallback here, in priority order:
        #   1. config/subjects/SXX.yaml (anthropometry.l1_m / l2_m)
        #   2. ~human_link_lengths = [l1, l2]  (explicit override)
        # The live topic is used while fresh; the static value fills in
        # otherwise; joint_safety only drops if neither is available.
        # ------------------------------------------------------------
        self.q_h = None
        self.l1 = None
        self.l2 = None
        self.q_h_stamp = rospy.Time(0)
        self.link_lengths_stamp = rospy.Time(0)
        self.max_human_state_age = rospy.get_param('~max_human_state_age', 0.3)

        # Shoulder/elbow/wrist Cartesian points from the pipeline
        # (fixed frame). Recorded as-is; re-published normalised below.
        self.arm_points_raw = None            # (9,) [sx..sz, ex..ez, wx..wz]
        self.arm_points_stamp = rospy.Time(0)

        self.l1_static = None
        self.l2_static = None
        ll_static = rospy.get_param('~human_link_lengths', None)
        if ll_static is not None:
            arr = np.asarray(ll_static, dtype=float)
            if arr.size != 2 or np.any(arr <= 0.0):
                rospy.logfatal('shared_control_node: ~human_link_lengths must '
                               'be [l1, l2] in metres, both > 0; got %s',
                               ll_static)
                raise rospy.ROSInitException('bad ~human_link_lengths')
            self.l1_static, self.l2_static = float(arr[0]), float(arr[1])
            rospy.loginfo('shared_control_node: static link lengths from '
                          '~human_link_lengths l1=%.3f l2=%.3f m',
                          self.l1_static, self.l2_static)
        elif self.subject is not None:
            self.l1_static, self.l2_static = self.subject.link_lengths()
            rospy.loginfo('shared_control_node: static link lengths from '
                          'subject %s l1=%.3f l2=%.3f m (fallback when %s is '
                          'silent/stale)', self.subject.subject_id,
                          self.l1_static, self.l2_static,
                          self.human_link_lengths_topic)

        # ------------------------------------------------------------
        # Robot state (from franka_states)
        # ------------------------------------------------------------
        self.x = np.zeros(3)
        self.q_robot = None
        self.J_robot = None
        self.have_robot_state = False

        # ------------------------------------------------------------
        # I/O
        # ------------------------------------------------------------
        self.cmd_pub = rospy.Publisher(self.cmd_topic, TwistStamped, queue_size=1)
        self.eta_pub = rospy.Publisher(self.eta_topic, Float64MultiArray, queue_size=1)

        # Diagnostics for offline metric analysis (Sec. V-D). Latched
        # where the value is constant for the whole run.
        self.diag = {
            'condition': rospy.Publisher('~diag/condition', String,
                                          queue_size=1, latch=True),
            'factors_layout': rospy.Publisher('~diag/factors_layout', String,
                                               queue_size=1, latch=True),
            'factors_h': rospy.Publisher('~diag/factors_h', Float64MultiArray,
                                          queue_size=1),
            'factors_r': rospy.Publisher('~diag/factors_r', Float64MultiArray,
                                          queue_size=1),
            # blend candidate v_hat_s, same [smoothness,directness,
            # joint_safety,manipulability] layout as factors_h/r
            'factors_s': rospy.Publisher('~diag/factors_s', Float64MultiArray,
                                          queue_size=1),
            'v_h': rospy.Publisher('~diag/v_h', Vector3Stamped, queue_size=1),
            'v_r': rospy.Publisher('~diag/v_r', Vector3Stamped, queue_size=1),
            'v_s': rospy.Publisher('~diag/v_s', Vector3Stamped, queue_size=1),
            'force': rospy.Publisher('~diag/force', Vector3Stamped, queue_size=1),
            'joint_margins': rospy.Publisher('~diag/joint_margins',
                                              Float64MultiArray, queue_size=1),
            # signed per-joint position rho in [-1,1] (0 mid, +-1 limit)
            'joint_rho': rospy.Publisher('~diag/joint_rho', Float64MultiArray,
                                          queue_size=1),
            # q1..q4 in DEGREES -- one rqt_plot window shows all four
            # joint angles at once (rho is unitless, radians are
            # awkward): rqt_plot /shared_control_node/diag/joint_deg/data[0]:data[1]:data[2]:data[3]
            'joint_deg': rospy.Publisher('~diag/joint_deg', Float64MultiArray,
                                          queue_size=1),
            # q1..q4 (deg) one controller lookahead (~dt_lookahead) ahead
            # along qdot_k -- the joint velocity each shared-control
            # candidate command induces (k = h: v_h, r: v_r, s: the
            # eta-weighted blend v_hat_s, BEFORE eta_s scales the
            # output), i.e. what joint_safety's dynamic term scores for
            # that candidate.
            'joint_deg_future_h': rospy.Publisher('~diag/joint_deg_future_h',
                                                   Float64MultiArray,
                                                   queue_size=1),
            'joint_deg_future_r': rospy.Publisher('~diag/joint_deg_future_r',
                                                   Float64MultiArray,
                                                   queue_size=1),
            'joint_deg_future_s': rospy.Publisher('~diag/joint_deg_future_s',
                                                   Float64MultiArray,
                                                   queue_size=1),
            # [q1min,q1max, ...q4min,q4max] (rad), latched
            'joint_limits': rospy.Publisher('~diag/joint_limits',
                                             Float64MultiArray, queue_size=1,
                                             latch=True),
            # same, in DEGREES -- reference lines for the joint_deg plot
            'joint_deg_limits': rospy.Publisher('~diag/joint_deg_limits',
                                                 Float64MultiArray, queue_size=1,
                                                 latch=True),
            'manipulability': rospy.Publisher('~diag/manipulability', Float64,
                                               queue_size=1),
            'path_progress': rospy.Publisher('~diag/path_progress',
                                              Float64MultiArray, queue_size=1),
            # shoulder/elbow/wrist points, poses[0..2] in that order:
            #  - arm_points     : from the pipeline, pipeline fixed frame
            #  - arm_points_fk  : FK from (q_h, l1, l2), 'human_shoulder'
            #                     frame (shoulder at origin) -- consistency
            #                     check, does NOT show shoulder drift
            'arm_points': rospy.Publisher('~diag/arm_points', PoseArray,
                                           queue_size=1),
            'arm_points_fk': rospy.Publisher('~diag/arm_points_fk', PoseArray,
                                              queue_size=1),
            'arm_points_layout': rospy.Publisher('~diag/arm_points_layout',
                                                  String, queue_size=1,
                                                  latch=True),
            'trial_done': rospy.Publisher('~diag/trial_done', Bool,
                                           queue_size=1, latch=True),
        }
        self.diag['trial_done'].publish(Bool(data=False))
        self.diag['condition'].publish(String(data=self.condition_id))
        self.diag['factors_layout'].publish(
            String(data=','.join(FACTOR_SLOTS)))
        self.diag['arm_points_layout'].publish(String(data='shoulder,elbow,wrist'))
        _jl = (self.human_joint_limits if self.human_joint_limits is not None
               else DEFAULT_JOINT_LIMITS)
        self.diag['joint_limits'].publish(Float64MultiArray(
            data=[float(v) for row in np.asarray(_jl) for v in row]))
        self.diag['joint_deg_limits'].publish(Float64MultiArray(
            data=[float(np.degrees(v)) for row in np.asarray(_jl) for v in row]))

        rospy.Service('~tare', Empty, self._tare_srv)

        # Optional plain-CSV log (one row per control cycle) -- an
        # analysis-ready file that does not need the rosbag. ~csv_path
        # gives an explicit file; else ~csv_dir auto-names
        # <label>_<YYYY-MM-DD_HH-MM-SS>.csv, where <label> = ~trial_label
        # if set, else <subject>_<condition> (subject id from the loaded
        # YAML) or just <condition>. Empty ~csv_path and ~csv_dir -> no
        # CSV. (The rosbag shares the <label> prefix; its timestamp uses
        # rosbag's own all-dashes format.)
        self.csv_fh = None
        self.csv_w = None
        csv_path = rospy.get_param('~csv_path', '')
        csv_dir = rospy.get_param('~csv_dir', '')
        if not csv_path and csv_dir:
            label = rospy.get_param('~trial_label', '')
            if not label:
                sid = (self.subject.subject_id + '_'
                       if self.subject is not None else '')
                label = sid + self.condition_id
            csv_path = os.path.join(
                os.path.expanduser(csv_dir),
                '%s_%s.csv' % (label, time.strftime('%Y-%m-%d_%H-%M-%S')))
        if csv_path:
            csv_path = os.path.expanduser(csv_path)
            d = os.path.dirname(csv_path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            self.csv_fh = open(csv_path, 'w')
            self.csv_w = csv.writer(self.csv_fh)
            self._csv_t0 = None
            self._csv_rows = 0
            self.csv_w.writerow(
                ['t', 't_rel', 'wall_time', 'condition', 'lap', 's_near',
                 'cross_track', 'px', 'py', 'pz',
                 'vh_x', 'vh_y', 'vh_z', 'vr_x', 'vr_y', 'vr_z',
                 'vs_x', 'vs_y', 'vs_z', 'fx', 'fy', 'fz',
                 'human_fresh', 'jac_fresh',
                 'eta_h', 'eta_r', 'eta_s',
                 'smoothness_h', 'directness_h', 'joint_safety_h', 'manip_h',
                 'rho1', 'rho2', 'rho3', 'rho4',
                 'm1', 'm2', 'm3', 'm4', 'm_min', 'w_qr'])
            rospy.loginfo('shared_control_node: CSV log -> %s', csv_path)
            rospy.on_shutdown(self._close_csv)

        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)
        rospy.Subscriber(self.human_joint_state_topic, JointState,
                          self._human_joint_state_cb, queue_size=1)
        rospy.Subscriber(self.human_link_lengths_topic, Float64MultiArray,
                          self._human_link_lengths_cb, queue_size=1)
        rospy.Subscriber(self.human_arm_points_topic, Float64MultiArray,
                          self._human_arm_points_cb, queue_size=1)
        if self.jacobian_source == 'topic':
            rospy.Subscriber(self.robot_jacobian_topic, Float64MultiArray,
                              self._robot_jacobian_cb, queue_size=1)

        self.rate_hz = rospy.get_param('~rate_hz', 200.0)
        self.dt = 1.0 / self.rate_hz

        if 'joint_safety' in self.cond_factors:
            has_ll = (self.l1_static is not None
                      or rospy.get_param('~human_link_lengths', None) is not None
                      or self.subject is not None)
            rospy.logwarn('shared_control_node: condition %s uses joint_safety '
                          '-- needs fresh q_h on %s AND l1,l2 (topic %s%s). '
                          'Run `rostopic hz` on both; with '
                          '~require_fresh_human:=true the node holds zero '
                          'velocity until they arrive.',
                          self.condition_id, self.human_joint_state_topic,
                          self.human_link_lengths_topic,
                          '' if has_ll else ' -- NO l1,l2 fallback configured, '
                          'pass subject:= or ~human_link_lengths')

        rospy.loginfo('shared_control_node: waiting for first franka_states '
                       'on %s ...', self.franka_states_topic)

    # ------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------
    def _franka_state_cb(self, msg):
        # O_F_ext_hat_K: estimated external wrench at the EE in the
        # base frame, from the FR3's joint-torque sensors + dynamic
        # model (NOT from the compliant gripper -- see paper Sec. 4.1).
        f_ext = -np.array(msg.O_F_ext_hat_K[0:3])

        if not self._tared:
            now = rospy.Time.now()
            if self._tare_t0 is None:
                self._tare_t0 = now
                rospy.logwarn('shared_control_node: taring the external-force '
                              'estimate for %.1f s -- HANDS OFF the robot.',
                              self.force_tare_s)
            self._tare_samples.append(f_ext)
            if (now - self._tare_t0).to_sec() >= self.force_tare_s:
                self.f_bias = np.mean(self._tare_samples, axis=0)
                self._tared = True
                self._tare_samples = []
                rospy.loginfo('shared_control_node: force bias = [%.2f %.2f '
                              '%.2f] N (|b|=%.2f). If large, set the FR3 EE '
                              'load so O_F_ext_hat_K reads ~0 at rest.',
                              self.f_bias[0], self.f_bias[1], self.f_bias[2],
                              float(np.linalg.norm(self.f_bias)))
        else:
            self._update_force(f_ext - self.f_bias)

        # O_T_EE is a 4x4 pose stored column-major; translation is
        # indices 12, 13, 14.
        self.x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
        self.q_robot = np.array(msg.q)
        self.have_robot_state = True

    def _tare_srv(self, _req):
        """std_srvs/Empty: re-run the startup force tare."""
        self.f_bias = np.zeros(3)
        self._tare_samples = []
        self._tare_t0 = None
        self._tared = False
        rospy.logwarn('shared_control_node: re-taring on request.')
        return EmptyResponse()

    # Human-arm chain, in priority order:
    #   1. by name -- right_arm_q1..q4 + upperarm_length / forearm_length
    #      (the visuo-tactile pipeline; robust to entry order).
    #   2. positional 7-entry layout q1,q2,l1,q3,q4,l2,q5.
    #   3. positional 4-entry layout q1..q4 (no l1,l2).
    _Q_NAMES = ('right_arm_q1', 'right_arm_q2', 'right_arm_q3', 'right_arm_q4')
    _L1_NAME = 'upperarm_length'
    _L2_NAME = 'forearm_length'

    def _human_joint_state_cb(self, msg):
        pos = msg.position
        idx = {n: i for i, n in enumerate(msg.name)} if msg.name else {}
        l1 = l2 = None
        if all(n in idx and idx[n] < len(pos) for n in self._Q_NAMES):
            q = np.array([pos[idx[n]] for n in self._Q_NAMES])
            if (self._L1_NAME in idx and self._L2_NAME in idx
                    and idx[self._L1_NAME] < len(pos)
                    and idx[self._L2_NAME] < len(pos)):
                l1, l2 = float(pos[idx[self._L1_NAME]]), float(pos[idx[self._L2_NAME]])
        elif len(pos) >= 7:
            q = np.array([pos[0], pos[1], pos[3], pos[4]])   # q1,q2,q3,q4
            l1, l2 = float(pos[2]), float(pos[5])            # upperarm, forearm
        elif len(pos) >= 4:
            q = np.array(pos[0:4])
        else:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published %d positions / names '
                '%s -- cannot extract q1..q4; ignoring.',
                self.human_joint_state_topic, len(pos), list(msg.name))
            return

        # pipeline convention -> DH model: q_model = gain*(q_pipeline - offset)
        q = self.human_joint_gains * (q - self.human_joint_offsets)
        now = rospy.Time.now()
        if self.q_h is None:
            rospy.loginfo('shared_control_node: first q_h on %s = %s  l1,l2=%s',
                          self.human_joint_state_topic, np.round(q, 3).tolist(),
                          None if l1 is None else [round(l1, 3), round(l2, 3)])
        self.q_h = q
        self.q_h_stamp = now
        # l1,l2 ride in this message for the pipeline chain -- treat them
        # exactly like the separate ~human_link_lengths_topic stream.
        if l1 is not None and l1 > 0.0 and l2 > 0.0:
            self.l1, self.l2 = l1, l2
            self.link_lengths_stamp = now

    def _human_link_lengths_cb(self, msg):
        if len(msg.data) < 2:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published %d values '
                '(need >= 2: l1, l2); ignoring.', self.human_link_lengths_topic,
                len(msg.data))
            return
        if self.l1 is None:
            rospy.loginfo('shared_control_node: first l1,l2 on %s = %.3f, %.3f',
                          self.human_link_lengths_topic,
                          float(msg.data[0]), float(msg.data[1]))
        self.l1, self.l2 = float(msg.data[0]), float(msg.data[1])
        self.link_lengths_stamp = rospy.Time.now()

    def _human_arm_points_cb(self, msg):
        if len(msg.data) < 9:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published <9 values '
                '(got %d); expected [sx,sy,sz, ex,ey,ez, wx,wy,wz]. '
                'Ignoring.', self.human_arm_points_topic, len(msg.data))
            return
        self.arm_points_raw = np.array(msg.data[0:9], dtype=float)
        self.arm_points_stamp = rospy.Time.now()

    def _robot_jacobian_cb(self, msg):
        n = self.jac_rows * self.jac_cols
        if len(msg.data) != n:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published %d values; expected '
                '%d (%dx%d row-major). Ignoring.', self.robot_jacobian_topic,
                len(msg.data), n, self.jac_rows, self.jac_cols)
            return
        J = np.asarray(msg.data, dtype=float).reshape(self.jac_rows,
                                                      self.jac_cols)
        now = rospy.Time.now()
        self.core.robot_model.update_current_state(self.q_robot, J,
                                                   stamp=now.to_sec())
        self.robot_jac_stamp = now

    def _update_force(self, f_ext_raw):
        f_norm = np.linalg.norm(f_ext_raw)
        f_deadzone = (np.zeros(3) if f_norm < self.deadzone_N else
                      f_ext_raw * (1.0 - self.deadzone_N / f_norm))
        self.f_filtered = (self.alpha_f * f_deadzone
                            + (1.0 - self.alpha_f) * self.f_filtered)

    def _link_lengths_topic_fresh(self):
        return (self.l1 is not None and self.l2 is not None
                and (rospy.Time.now() - self.link_lengths_stamp).to_sec()
                <= self.max_human_state_age)

    def _current_link_lengths(self):
        """(l1, l2) to use this cycle: the topic estimate while fresh,
        else the static ~human_link_lengths fallback, else (None, None)."""
        if self._link_lengths_topic_fresh():
            return self.l1, self.l2
        if self.l1_static is not None:
            return self.l1_static, self.l2_static
        return None, None

    def _human_state_fresh(self):
        now = rospy.Time.now()
        q_ok = (self.q_h is not None
                and (now - self.q_h_stamp).to_sec() <= self.max_human_state_age)
        l1, l2 = self._current_link_lengths()
        return q_ok and l1 is not None

    def _jacobian_fresh(self):
        """Whether the manipulability factor can run this cycle.
        analytic / kdl compute from the current q on demand -> fresh as
        long as we have a robot state; the topic backend needs a recent
        message (fail soft, like the human-state staleness)."""
        if not self.manipulability_available:
            return False
        if self.jacobian_source != 'topic':
            return self.q_robot is not None
        return (self.robot_jac_stamp is not None
                and (rospy.Time.now() - self.robot_jac_stamp).to_sec()
                <= self.max_jacobian_age)

    def _shape_standalone(self, v_h):
        """Condition A output: LPF + speed cap on the human command,
        identical to the shaping the core applies to v_s."""
        self._standalone_filt = (self.lpf_alpha * v_h
                                  + (1.0 - self.lpf_alpha) * self._standalone_filt)
        speed = np.linalg.norm(self._standalone_filt)
        if speed <= self.v_max:
            return self._standalone_filt
        return self._standalone_filt * self.v_max / max(speed, 1e-9)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    @staticmethod
    def _pose_array(points, frame_id, stamp):
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = frame_id
        for p in points:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                float(p[0]), float(p[1]), float(p[2]))
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        return pa

    def _publish_diag(self, stamp, v_h, v_r, v_s, factors_h, factors_r,
                       s_near, lap, cross_track, human_fresh, l1_cur, l2_cur,
                       v_hat_s=None, factors_s=None):
        def vec3(pub_key, v):
            m = Vector3Stamped()
            m.header.stamp = stamp
            m.header.frame_id = self.base_frame
            m.vector.x, m.vector.y, m.vector.z = (float(v[0]), float(v[1]),
                                                   float(v[2]))
            self.diag[pub_key].publish(m)

        vec3('v_h', v_h)
        vec3('v_r', v_r)
        vec3('v_s', v_s)
        vec3('force', self.f_filtered)

        self.diag['factors_h'].publish(
            Float64MultiArray(data=_factors_to_array(factors_h)))
        self.diag['factors_r'].publish(
            Float64MultiArray(data=_factors_to_array(factors_r)))
        self.diag['factors_s'].publish(
            Float64MultiArray(data=_factors_to_array(factors_s or {})))

        # [s_near, completed laps, continuous progress (laps, monotone
        #  in the travel direction), cross-track error m]
        self.diag['path_progress'].publish(Float64MultiArray(
            data=[float(s_near), float(lap),
                  float(self.lap_counter.total_progress),
                  float(cross_track)]))

        if human_fresh and self.q_h is not None:
            margins, min_margin = joint_margins(self.q_h,
                                                self.human_joint_limits)
            self.diag['joint_margins'].publish(Float64MultiArray(
                data=list(map(float, margins)) + [float(min_margin)]))
            self.diag['joint_rho'].publish(Float64MultiArray(
                data=[float(r) for r in joint_rho(self.q_h,
                                                  self.human_joint_limits)]))
            self.diag['joint_deg'].publish(Float64MultiArray(
                data=[float(np.degrees(q)) for q in self.q_h[:4]]))
            if l1_cur is not None:
                # Extrapolate q_h along the joint velocity each candidate
                # command (v_h, v_r, v_hat_s) induces -- what
                # joint_safety's dynamic term scores -- over the
                # controller's own lookahead horizon dt_lookahead. One
                # arm Jacobian, reused across the three (as core.step does).
                J_arm = human_arm_jacobian(self.q_h, l1_cur, l2_cur)
                dtl = self.core.dt_lookahead
                for key, v_k in (('joint_deg_future_h', v_h),
                                 ('joint_deg_future_r', v_r),
                                 ('joint_deg_future_s', v_hat_s)):
                    if v_k is None:
                        continue
                    qdot_k = cartesian_to_human_joint_velocity(
                        self.q_h, l1_cur, l2_cur, v_k, J=J_arm)
                    q_future = self.q_h[:4] + qdot_k[:4] * dtl
                    self.diag[key].publish(Float64MultiArray(
                        data=[float(np.degrees(q)) for q in q_future]))

        if self.manipulability_available and self.J_robot is not None:
            self.diag['manipulability'].publish(
                Float64(data=float(manipulability_index(self.J_robot))))

        # Shoulder/elbow/wrist Cartesian points.
        if (self.arm_points_raw is not None
                and (stamp - self.arm_points_stamp).to_sec()
                <= self.max_human_state_age):
            pts = self.arm_points_raw.reshape(3, 3)
            self.diag['arm_points'].publish(
                self._pose_array(pts, self.human_points_frame, stamp))
        if (self.q_h is not None and l1_cur is not None
                and (stamp - self.q_h_stamp).to_sec() <= self.max_human_state_age):
            sh, el, wr = human_arm_points(self.q_h, l1_cur, l2_cur)
            self.diag['arm_points_fk'].publish(
                self._pose_array((sh, el, wr), self.arm_fk_frame, stamp))

    # ------------------------------------------------------------
    # CSV log (one row per cycle; analysis without the rosbag)
    # ------------------------------------------------------------
    def _write_csv_row(self, stamp, s_near, lap, cross_track, v_r, v_s,
                        factors_h, eta_h, eta_r, eta_s, human_fresh, jac_fresh,
                        l1_cur, l2_cur):
        m = [float('nan')] * 9   # rho1..4, m1..4, m_min
        if human_fresh and self.q_h is not None:
            margins, mmin = joint_margins(self.q_h, self.human_joint_limits)
            rho = joint_rho(self.q_h, self.human_joint_limits)
            m = ([float(r) for r in rho] + list(map(float, margins))
                 + [float(mmin)])
        w = float('nan')
        if self.J_robot is not None:
            w = float(manipulability_index(self.J_robot))
        fh = factors_h or {}
        ts = stamp.to_sec()
        if self._csv_t0 is None:
            self._csv_t0 = ts
        wall = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
                + ('.%03d' % int((ts % 1.0) * 1000)))
        row = [ts, ts - self._csv_t0, wall,
               self.condition_id, lap, s_near, cross_track,
               self.x[0], self.x[1], self.x[2],
               self.v_h[0], self.v_h[1], self.v_h[2],
               v_r[0], v_r[1], v_r[2], v_s[0], v_s[1], v_s[2],
               self.f_filtered[0], self.f_filtered[1], self.f_filtered[2],
               int(bool(human_fresh)), int(bool(jac_fresh)),
               eta_h, eta_r, eta_s,
               fh.get('smoothness', float('nan')),
               fh.get('directness', float('nan')),
               fh.get('joint_safety', float('nan')),
               fh.get('manipulability', float('nan'))] + m + [w]
        # row[0]=t (epoch), row[1]=t_rel: need full precision, not %g.
        out = [row[0], row[1]] + [
            ('%.6g' % v if isinstance(v, float) else v) for v in row[2:]]
        out[0] = '%.3f' % out[0]
        out[1] = '%.3f' % out[1]
        self.csv_w.writerow(out)
        self._csv_rows += 1
        if self._csv_rows % 200 == 0:        # ~1 s at 200 Hz
            self.csv_fh.flush()

    def _close_csv(self):
        if self.csv_fh is not None:
            try:
                self.csv_fh.flush()
                self.csv_fh.close()
            except Exception:
                pass
            self.csv_fh = None

    # ------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        warned_stale_human_state = False
        warned_stale_jacobian = False

        while not rospy.is_shutdown():
            if not self.have_robot_state or not self._tared:
                # No command published until the force tare has finished
                # (robot must not drift on an un-taured wrench).
                rate.sleep()
                continue

            # 1. Human-intent velocity from the admittance model.
            accel = (self.f_filtered - self.B_h * self.v_h) / self.M_h
            self.v_h = self.v_h + accel * self.dt

            # 2. Path progress (observable) + robot command.
            s_near, cross_track = self.follower.progress(self.x)
            lap = self.lap_counter.update(s_near)

            if (self.trial_laps > 0 and lap >= self.trial_laps
                    and not self._trial_done):
                self._trial_done = True
                self._trial_done_t = rospy.Time.now()
                self.diag['trial_done'].publish(Bool(data=True))
                rospy.loginfo('shared_control_node: trial complete (%d laps). '
                              '%s', self.trial_laps,
                              'shutting down.' if self.trial_end == 'shutdown'
                              else 'holding zero velocity -- Ctrl-C to end.')

            if (self._trial_done and self.trial_end == 'shutdown'
                    and (rospy.Time.now() - self._trial_done_t).to_sec() > 0.5):
                # published ~0.5 s of zero commands -> the robot is
                # stopped; now exit (required="true" brings down rviz +
                # rosbag too, closing the bag/CSV cleanly).
                rospy.signal_shutdown('trial complete')
                break

            if self.cond_use_robot:
                v_r, tangent = self.follower.robot_command(self.x)
            else:
                v_r = np.zeros(3)
                tangent = self.path_dir * self.path.tangent(s_near)

            # 3. Active factors: start from the CONDITION's factor set,
            #    then drop joint_safety if the human-arm state is
            #    missing/stale, drop manipulability if KDL never
            #    initialized. Fail SOFT (fewer factors), never fail
            #    SILENT-WRONG (use garbage q_h / a stale Jacobian).
            human_fresh = self._human_state_fresh()
            jac_fresh = self._jacobian_fresh()
            if human_fresh:
                self._human_ever_fresh = True
            active_factors = []
            for f in self.cond_factors:
                if f == 'joint_safety' and not human_fresh:
                    continue
                if f == 'manipulability' and not jac_fresh:
                    continue
                active_factors.append(f)

            need_jac = 'manipulability' in self.cond_factors
            if (need_jac and self.manipulability_available and not jac_fresh
                    and not warned_stale_jacobian):
                rospy.logwarn('shared_control_node: robot Jacobian stale/'
                               'missing on %s -- condition %s running WITHOUT '
                               'manipulability until it recovers.',
                               self.robot_jacobian_topic, self.condition_id)
                warned_stale_jacobian = True
            if need_jac and jac_fresh and warned_stale_jacobian:
                rospy.loginfo('shared_control_node: robot Jacobian recovered, '
                               'manipulability re-enabled.')
                warned_stale_jacobian = False

            need_human = 'joint_safety' in self.cond_factors
            hold_for_human = (need_human and not human_fresh
                              and self.require_fresh_human)
            if need_human and not human_fresh:
                if not warned_stale_human_state:
                    rospy.logwarn('shared_control_node: human arm state '
                                   'stale/missing (topics %s, %s) -- condition '
                                   '%s WITHOUT joint_safety.',
                                   self.human_joint_state_topic,
                                   self.human_link_lengths_topic,
                                   self.condition_id)
                    warned_stale_human_state = True
                # Keep shouting for the whole session, not just once --
                # this is an invalid trial for a joint_safety condition.
                rospy.logerr_throttle(
                    10.0, 'shared_control_node: condition %s REQUIRES '
                    'joint_safety but no fresh q_h+(l1,l2) -- %s. Check '
                    '`rostopic hz %s %s`; set subject:= (filled l1_m/l2_m) '
                    'or ~human_link_lengths for the l1,l2 fallback.',
                    self.condition_id,
                    'HOLDING ZERO VELOCITY (require_fresh_human)'
                    if self.require_fresh_human else 'recording an INVALID '
                    'trial (joint_safety_h/m* will be all NaN)',
                    self.human_joint_state_topic, self.human_link_lengths_topic)
            if need_human and human_fresh and warned_stale_human_state:
                rospy.loginfo('shared_control_node: human arm state '
                               'recovered, joint_safety re-enabled.')
                warned_stale_human_state = False

            # 4. Emergent command.
            self.J_robot = None
            if self.manipulability_available and jac_fresh:
                self.J_robot = self.core.robot_model.jacobian(self.q_robot)

            l1_cur, l2_cur = self._current_link_lengths()
            v_hat_s = None
            if self.cond_use_robot:
                v_s, info = self.core.step(
                    self.v_h, v_r, tangent,
                    q_human=self.q_h if human_fresh else None,
                    l1=l1_cur if human_fresh else None,
                    l2=l2_cur if human_fresh else None,
                    q_robot=self.q_robot,
                    J_robot=self.J_robot,
                    active_factors=tuple(active_factors))
                eta_h, eta_r, eta_s = (info['eta_h'], info['eta_r'],
                                       info['eta_s'])
                factors_h, factors_r = info['factors_h'], info['factors_r']
                factors_s = info['factors_s']
                v_hat_s = info['v_hat_s']   # blend before eta_s scales it
            else:
                # Condition A: no assistance. Shape v_h only; the core
                # blend / eta_s pass are bypassed by design.
                v_s = self._shape_standalone(self.v_h)
                eta_h, eta_r, eta_s = 1.0, 0.0, 1.0
                factors_h, factors_r, factors_s = {}, {}, {}
                self.core.v_prev = v_s  # keep smoothness reference coherent

            if self._trial_done or hold_for_human:
                v_s = np.zeros(3)       # trial over / waiting for q_h -> hold still

            # 5. Publish command + eta + diagnostics.
            stamp = rospy.Time.now()

            twist_msg = TwistStamped()
            twist_msg.header.stamp = stamp
            twist_msg.header.frame_id = self.base_frame
            twist_msg.twist.linear.x = float(v_s[0])
            twist_msg.twist.linear.y = float(v_s[1])
            twist_msg.twist.linear.z = float(v_s[2])
            self.cmd_pub.publish(twist_msg)

            eta_msg = Float64MultiArray()
            eta_msg.data = [eta_h, eta_r, eta_s]
            self.eta_pub.publish(eta_msg)

            self._publish_diag(stamp, self.v_h, v_r, v_s, factors_h,
                                factors_r, s_near, lap, cross_track,
                                human_fresh, l1_cur, l2_cur, v_hat_s,
                                factors_s)

            if self.csv_w is not None:
                self._write_csv_row(stamp, s_near, lap, cross_track, v_r, v_s,
                                     factors_h, eta_h, eta_r, eta_s,
                                     human_fresh, jac_fresh, l1_cur, l2_cur)

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SharedControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
