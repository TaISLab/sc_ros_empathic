#!/usr/bin/env python3
"""
baseline_aan_node.py
--------------------
Experimental condition F (paper Sec. V-B): the impedance-control
assist-as-needed (AAN) baseline of Zhang et al. [9], nominal placement
only. A different controller from shared_control_node.py -- it shares
none of the four-factor machinery -- but it uses the SAME node-side
infrastructure (ros_helpers): the startup force tare + `~tare` service,
the direction-aware jitter-robust lap counter, `~trial_laps` /
`~trial_end`, the per-cycle CSV log, and the same conservative
admittance / loop defaults, so an F session runs and records like any
other condition.

    !!! IMPLEMENTATION STATUS -- READ BEFORE USING FOR A PUBLISHED RESULT
    Reference [9] (L. Zhang, S. Guo, F. Xi, Mechatronics 89:102919,
    2023) is NOT reproduced verbatim. This implements the standard
    structure of impedance-based AAN control:
      * a virtual impedance (stiffness K, damping D) pulling the grasped
        wrist toward the moving reference x_d(t) on the circular path;
      * an assist-as-needed dead-band: no assistive action inside a tube
        of radius `deadband_m`, ramping in beyond it;
      * realised as a Cartesian velocity command (the FR3 interface here
        is velocity), blended with the same admittance-derived v_h.
    Set K, D, the dead-band / performance-scaling law and the update
    rule to match [9] and PRE-REGISTER them before recording a real
    condition-F result.

Publishes <cmd_topic> and a reduced ~diag/* set (condition, v_h,
v_r == v_assist, v_s, force, path_progress, trial_done).
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from std_msgs.msg import Bool, Float64MultiArray, String

from sc_ros_empathic.path_follower import CirclePath, ReactivePathFollower
from sc_ros_empathic.ros_helpers import ForceProcessor, TrialManager, CsvLogger


class BaselineAANNode(object):
    def __init__(self):
        rospy.init_node('baseline_aan_node', anonymous=False)

        self.franka_states_topic = rospy.get_param(
            '~franka_states_topic', '/franka_state_controller/franka_states')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/robot_vel_ctrl/vel_cmd')
        self.base_frame = rospy.get_param('~base_frame', 'fr3_link0')

        # --- path (nominal placement) --------------------------------
        center = rospy.get_param('~path_center', [0.45, 0.0, 0.45])
        radius = rospy.get_param('~path_radius', 0.05)
        normal = rospy.get_param('~path_normal', [0.0, 0.0, 1.0])
        pd = str(rospy.get_param('~path_direction', 'forward')).lower()
        self.path_dir = -1 if pd == 'reverse' else 1
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        # follower is only used to get the moving reference x_d and the
        # travel tangent; F's assist comes from its own impedance, not
        # from the follower command.
        self.follower = ReactivePathFollower(
            self.path, Ka=1.0, rho_min=rospy.get_param('~rho_min', 0.02),
            cruise_speed=0.0, mode='lookahead', direction=self.path_dir)

        # --- human admittance (same conservative defaults as the
        #     main node) -----------------------------------------------
        self.M_h = np.array(rospy.get_param('~admittance_mass', [2.0, 2.0, 2.0]))
        self.B_h = np.array(rospy.get_param('~admittance_damping', [40.0, 40.0, 40.0]))
        self.v_h = np.zeros(3)

        # --- impedance-AAN parameters (PLACEHOLDERS -- set from [9]) --
        self.K = float(rospy.get_param('~impedance_stiffness', 300.0))     # N/m
        self.D = float(rospy.get_param('~impedance_damping', 40.0))        # N.s/m
        self.adm_gain = float(rospy.get_param('~assist_admittance_gain', 0.002))
        self.deadband_m = float(rospy.get_param('~deadband_m', 0.01))
        self.assist_ramp = float(rospy.get_param('~assist_ramp', 1.0))     # 1/m

        # --- output shaping ---------------------------------------------
        self.v_max = float(rospy.get_param('~v_max', 0.08))
        self.lpf_alpha = float(rospy.get_param('~lpf_alpha', 0.15))
        self.v_s_filt = np.zeros(3)

        self.x = np.zeros(3)
        self.x_prev = None
        self.have_robot_state = False

        # --- I/O ------------------------------------------------------
        self.cmd_pub = rospy.Publisher(self.cmd_topic, TwistStamped, queue_size=1)
        self.diag = {
            'condition': rospy.Publisher('~diag/condition', String,
                                          queue_size=1, latch=True),
            'trial_done': rospy.Publisher('~diag/trial_done', Bool,
                                           queue_size=1, latch=True),
            'v_h': rospy.Publisher('~diag/v_h', Vector3Stamped, queue_size=1),
            'v_r': rospy.Publisher('~diag/v_r', Vector3Stamped, queue_size=1),
            'v_s': rospy.Publisher('~diag/v_s', Vector3Stamped, queue_size=1),
            'force': rospy.Publisher('~diag/force', Vector3Stamped, queue_size=1),
            'path_progress': rospy.Publisher('~diag/path_progress',
                                              Float64MultiArray, queue_size=1),
        }
        self.diag['condition'].publish(String(data='F_impedance_aan'))

        # --- shared node-side infrastructure ------------------------
        self.force = ForceProcessor(
            deadzone_N=rospy.get_param('~force_deadzone_N', 2.0),
            lpf_alpha=rospy.get_param('~force_lpf_alpha', 0.1),
            tare_s=rospy.get_param('~force_tare_s', 1.0))
        self.trial = TrialManager(
            direction=self.path_dir,
            trial_laps=rospy.get_param('~trial_laps', 0),
            trial_end=rospy.get_param('~trial_end', 'shutdown'),
            done_pub=self.diag['trial_done'])
        self.csv = CsvLogger(
            ['condition', 'lap', 's_near', 'cross_track', 'px', 'py', 'pz',
             'vh_x', 'vh_y', 'vh_z', 'vr_x', 'vr_y', 'vr_z',
             'vs_x', 'vs_y', 'vs_z', 'fx', 'fy', 'fz'],
            path=rospy.get_param('~csv_path', ''),
            out_dir=rospy.get_param('~csv_dir', ''),
            label=rospy.get_param('~trial_label', '') or 'F_impedance_aan')

        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)

        self.rate_hz = rospy.get_param('~rate_hz', 200.0)
        self.dt = 1.0 / self.rate_hz
        rospy.loginfo('baseline_aan_node: condition F (impedance-AAN, Zhang '
                       'et al. [9]) -- verify K/D/dead-band against [9] and '
                       'pre-register before recording.')

    def _franka_state_cb(self, msg):
        self.force.update(-np.array(msg.O_F_ext_hat_K[0:3]))
        self.x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
        self.have_robot_state = True

    def _vec3(self, key, v, stamp):
        m = Vector3Stamped()
        m.header.stamp = stamp
        m.header.frame_id = self.base_frame
        m.vector.x, m.vector.y, m.vector.z = float(v[0]), float(v[1]), float(v[2])
        self.diag[key].publish(m)

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if not self.have_robot_state or not self.force.ready:
                rate.sleep()
                continue

            # Human command (admittance).
            accel = (self.force.f_filtered - self.B_h * self.v_h) / self.M_h
            self.v_h = self.v_h + accel * self.dt

            x_d, tangent, _ = self.follower.next_goal(self.x)
            x_dot = (np.zeros(3) if self.x_prev is None
                     else (self.x - self.x_prev) / self.dt)
            self.x_prev = self.x.copy()

            s_near, cross_track = self.follower.progress(self.x)
            lap = self.trial.update(s_near)

            # Assist-as-needed gate + virtual impedance -> velocity.
            err = x_d - self.x
            gate = min(1.0, self.assist_ramp
                       * max(0.0, np.linalg.norm(err) - self.deadband_m))
            v_assist = self.adm_gain * gate * (self.K * err - self.D * x_dot)

            v_s = self.v_h + v_assist
            self.v_s_filt = (self.lpf_alpha * v_s
                              + (1.0 - self.lpf_alpha) * self.v_s_filt)
            speed = np.linalg.norm(self.v_s_filt)
            v_out = (self.v_s_filt if speed <= self.v_max
                     else self.v_s_filt * self.v_max / max(speed, 1e-9))
            if self.trial.done:
                v_out = np.zeros(3)

            stamp = rospy.Time.now()
            tw = TwistStamped()
            tw.header.stamp = stamp
            tw.header.frame_id = self.base_frame
            tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = (
                float(v_out[0]), float(v_out[1]), float(v_out[2]))
            self.cmd_pub.publish(tw)

            self._vec3('v_h', self.v_h, stamp)
            self._vec3('v_r', v_assist, stamp)
            self._vec3('v_s', v_out, stamp)
            self._vec3('force', self.force.f_filtered, stamp)
            self.diag['path_progress'].publish(Float64MultiArray(
                data=[float(s_near), float(lap),
                      float(self.trial.total_progress), float(cross_track)]))
            if self.csv.active:
                self.csv.row(stamp.to_sec(), [
                    'F_impedance_aan', lap, s_near, cross_track,
                    self.x[0], self.x[1], self.x[2],
                    self.v_h[0], self.v_h[1], self.v_h[2],
                    v_assist[0], v_assist[1], v_assist[2],
                    v_out[0], v_out[1], v_out[2],
                    self.force.f_filtered[0], self.force.f_filtered[1],
                    self.force.f_filtered[2]])

            if self.trial.should_stop():
                rospy.signal_shutdown('trial complete')
                break

            rate.sleep()


if __name__ == '__main__':
    try:
        BaselineAANNode().run()
    except rospy.ROSInterruptException:
        pass
