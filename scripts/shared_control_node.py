#!/usr/bin/env python3
"""
shared_control_node.py
-----------------------
ROS1 node implementing the reactive, performance-weighted shared
control law (human joint-limit safety + robot manipulability factors)
on top of the FR3's external Cartesian velocity controller.

Reads:
  * /franka_state_controller/franka_states (franka_msgs/FrankaState)
        -> O_F_ext_hat_K (interaction force/torque at the EE, base
           frame) drives the admittance model that produces v_h;
           O_T_EE (EE pose) and q (joint angles) give the current
           Cartesian position x and robot configuration q_robot.
  * human arm joint state topic (q_h = q1..q4) and human arm segment
    lengths (l1, l2), published by the visuo-tactile pipeline.
        !!! TOPIC NAMES/MESSAGE TYPES BELOW ARE PLACEHOLDERS !!!
        Confirm the real ones with the visuo-tactile pipeline owner
        and override via the ROS params below -- no code change
        needed once confirmed, just launch-file arguments.

Publishes:
  * <cmd_topic> (geometry_msgs/TwistStamped): the emergent Cartesian
    velocity v_s, consumed by the FR3's external Cartesian velocity
    controller (taislab_franka_controllers /
    cartesian_velocity_external_controller).
  * ~eta (std_msgs/Float64MultiArray): [eta_h, eta_r, eta_s], logged
    for offline analysis (e.g. the "compromise-direction"/fork-in-
    the-path diagnostic described in the paper).

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

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from sc_ros_empathic.shared_control_core import SharedControlCore
from sc_ros_empathic.robot_model import RobotModel
from sc_ros_empathic.path_follower import CirclePath, ReactivePathFollower


class SharedControlNode(object):
    def __init__(self):
        rospy.init_node('shared_control_node', anonymous=False)

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
        # PLACEHOLDER -- confirm with the visuo-tactile pipeline owner.
        # Expected: sensor_msgs/JointState with position[0:4] = q1..q4
        # (shoulder flex/ext, abd/add, int/ext rotation, elbow flex/ext).
        self.human_joint_state_topic = rospy.get_param(
            '~human_joint_state_topic', '/right_arm/joint_states')
        # PLACEHOLDER -- confirm message type/layout. Expected:
        # std_msgs/Float64MultiArray with data = [l1, l2] (m).
        self.human_link_lengths_topic = rospy.get_param(
            '~human_link_lengths_topic', '/right_arm/link_lengths')
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
        Ka = rospy.get_param('~Ka', 2.0)
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        self.follower = ReactivePathFollower(self.path, Ka=Ka)

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

        # ------------------------------------------------------------
        # Shared-control law. Weight/gain defaults below are the ones
        # tuned in offline simulation (see the paper, Sec. 4);
        # re-tune on hardware before trusting them, they are a start
        # point, not a validated-on-robot result.
        # ------------------------------------------------------------
        weights = {
            'smoothness': rospy.get_param('~w_smoothness', 1.0),
            'directness': rospy.get_param('~w_directness', 1.0),
            'joint_safety': rospy.get_param('~w_joint_safety', 16.0),
            'manipulability': rospy.get_param('~w_manipulability', 24.0),
        }
        base_link = rospy.get_param('~base_link', 'fr3_link0')
        ee_link = rospy.get_param('~ee_link', 'fr3_link8')
        try:
            robot_model = RobotModel(backend='kdl', base_link=base_link, ee_link=ee_link)
        except RuntimeError as e:
            rospy.logerr('%s', e)
            rospy.logerr('shared_control_node: disabling the manipulability '
                          'factor for this run (active_factors will drop it).')
            robot_model = None
        self.manipulability_available = robot_model is not None

        self.core = SharedControlCore(
            dt_lookahead=rospy.get_param('~dt_lookahead', 0.2),
            v_max=rospy.get_param('~v_max', 0.15),
            lpf_alpha=rospy.get_param('~lpf_alpha', 0.2),
            weights=weights,
            C1=rospy.get_param('~C1', 1.0),
            C2=rospy.get_param('~C2', 1.0),
            Cs=rospy.get_param('~Cs', 12.0),
            Cm=rospy.get_param('~Cm', 24.0),
            robot_model=robot_model,
        )

        # ------------------------------------------------------------
        # Human-arm state (fail-safe: stale/missing -> drop
        # joint_safety from active_factors rather than run with a
        # frozen/garbage q_h, mirroring the Gamma-timeout fail-safe
        # pattern already used elsewhere in this lab's controllers).
        # ------------------------------------------------------------
        self.q_h = None
        self.l1 = None
        self.l2 = None
        self.q_h_stamp = rospy.Time(0)
        self.link_lengths_stamp = rospy.Time(0)
        self.max_human_state_age = rospy.get_param('~max_human_state_age', 0.3)

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

        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)
        rospy.Subscriber(self.human_joint_state_topic, JointState,
                          self._human_joint_state_cb, queue_size=1)
        rospy.Subscriber(self.human_link_lengths_topic, Float64MultiArray,
                          self._human_link_lengths_cb, queue_size=1)

        self.rate_hz = rospy.get_param('~rate_hz', 200.0)
        self.dt = 1.0 / self.rate_hz

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
        self._update_force(f_ext)

        # O_T_EE is a 4x4 pose stored column-major; translation is
        # indices 12, 13, 14.
        self.x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
        self.q_robot = np.array(msg.q)
        self.have_robot_state = True

    def _human_joint_state_cb(self, msg):
        if len(msg.position) < 4:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published <4 positions '
                '(got %d); ignoring.', self.human_joint_state_topic,
                len(msg.position))
            return
        self.q_h = np.array(msg.position[0:4])
        self.q_h_stamp = rospy.Time.now()

    def _human_link_lengths_cb(self, msg):
        if len(msg.data) < 2:
            rospy.logwarn_throttle(
                5.0, 'shared_control_node: %s published <2 values '
                '(got %d); ignoring.', self.human_link_lengths_topic,
                len(msg.data))
            return
        self.l1, self.l2 = float(msg.data[0]), float(msg.data[1])
        self.link_lengths_stamp = rospy.Time.now()

    def _update_force(self, f_ext_raw):
        f_norm = np.linalg.norm(f_ext_raw)
        f_deadzone = (np.zeros(3) if f_norm < self.deadzone_N else
                      f_ext_raw * (1.0 - self.deadzone_N / f_norm))
        self.f_filtered = (self.alpha_f * f_deadzone
                            + (1.0 - self.alpha_f) * self.f_filtered)

    def _human_state_fresh(self):
        now = rospy.Time.now()
        q_ok = (self.q_h is not None
                and (now - self.q_h_stamp).to_sec() <= self.max_human_state_age)
        len_ok = (self.l1 is not None and self.l2 is not None
                  and (now - self.link_lengths_stamp).to_sec() <= self.max_human_state_age)
        return q_ok and len_ok

    # ------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(self.rate_hz)
        warned_stale_human_state = False

        while not rospy.is_shutdown():
            if not self.have_robot_state:
                rate.sleep()
                continue

            # 1. Human-intent velocity from the admittance model.
            accel = (self.f_filtered - self.B_h * self.v_h) / self.M_h
            self.v_h = self.v_h + accel * self.dt

            # 2. Robot path-following velocity + path tangent.
            v_r, tangent = self.follower.robot_command(self.x)

            # 3. Active factors: drop joint_safety if the human-arm
            #    state is missing/stale, drop manipulability if KDL
            #    never initialized. Fail SOFT (degrade to fewer
            #    factors), never fail SILENT-WRONG (use garbage q_h).
            active_factors = ['smoothness', 'directness']
            human_fresh = self._human_state_fresh()
            if human_fresh:
                active_factors.append('joint_safety')
            elif not warned_stale_human_state:
                rospy.logwarn('shared_control_node: human arm state '
                               'stale/missing (topics %s, %s) -- running '
                               'WITHOUT joint_safety until it recovers.',
                               self.human_joint_state_topic,
                               self.human_link_lengths_topic)
                warned_stale_human_state = True
            if human_fresh and warned_stale_human_state:
                rospy.loginfo('shared_control_node: human arm state '
                               'recovered, joint_safety re-enabled.')
                warned_stale_human_state = False
            if self.manipulability_available:
                active_factors.append('manipulability')

            J_robot = None
            if self.manipulability_available:
                J_robot = self.core.robot_model.jacobian(self.q_robot)

            v_s, info = self.core.step(
                self.v_h, v_r, tangent,
                q_human=self.q_h if human_fresh else None,
                l1=self.l1 if human_fresh else None,
                l2=self.l2 if human_fresh else None,
                q_robot=self.q_robot,
                J_robot=J_robot,
                active_factors=tuple(active_factors))

            # 4. Publish.
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()
            twist_msg.header.frame_id = self.base_frame
            twist_msg.twist.linear.x = float(v_s[0])
            twist_msg.twist.linear.y = float(v_s[1])
            twist_msg.twist.linear.z = float(v_s[2])
            self.cmd_pub.publish(twist_msg)

            eta_msg = Float64MultiArray()
            eta_msg.data = [info['eta_h'], info['eta_r'], info['eta_s']]
            self.eta_pub.publish(eta_msg)

            rate.sleep()


if __name__ == '__main__':
    try:
        node = SharedControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
