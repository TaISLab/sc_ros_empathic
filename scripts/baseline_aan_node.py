#!/usr/bin/env python3
"""
baseline_aan_node.py
--------------------
Experimental condition F (paper Sec. V-B): the impedance-control
assist-as-needed (AAN) baseline of Zhang et al. [9], used ONLY in the
nominal placement to anchor comparability with prior model-based AAN
work. It is a different controller from the performance-weighted law in
shared_control_node.py and shares none of its four-factor machinery.

    !!! IMPLEMENTATION STATUS -- READ BEFORE USING FOR A PUBLISHED RESULT
    Reference [9] (L. Zhang, S. Guo, F. Xi, "Performance-based assistance
    control for robot-mediated upper-limbs rehabilitation", Mechatronics
    89:102919, 2023) is not reproduced verbatim here. This node
    implements the *standard structure* of impedance-based AAN control
    so the condition can be run and recorded on the same hardware/
    interface as the other conditions:

      * a virtual impedance (stiffness K, damping D) pulling the grasped
        wrist toward the moving reference x_d(t) on the same circular
        path used by the other conditions;
      * an assist-as-needed dead-band: no assistive action while the
        wrist stays within a tube of radius `deadband_m` of the path,
        assistance ramping in beyond it (optionally scaled by a
        performance estimate);
      * realised as a Cartesian *velocity* command (the FR3 interface
        available here is velocity, not torque), blended with the same
        admittance-derived human command v_h as the other conditions.

    Set K, D, the dead-band / performance-scaling law and the update
    rule to match [9] exactly, and PRE-REGISTER them, before this is
    used as the paper's condition F. The parameter names below are the
    knobs to do that; the numeric defaults are placeholders.

Publishes the same command topic and a reduced ~diag/* set
(condition, v_h, v_r == v_assist, v_s, force, path_progress) so the
Sec. V-D offline analysis can treat an F bag like any other.
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import TwistStamped, Vector3Stamped
from std_msgs.msg import Float64MultiArray, String

from sc_ros_empathic.path_follower import CirclePath, ReactivePathFollower
from sc_ros_empathic.experiment import LapCounter


class BaselineAANNode(object):
    def __init__(self):
        rospy.init_node('baseline_aan_node', anonymous=False)

        self.franka_states_topic = rospy.get_param(
            '~franka_states_topic', '/franka_state_controller/franka_states')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/robot_vel_ctrl/vel_cmd')
        self.base_frame = rospy.get_param('~base_frame', 'fr3_link0')

        center = rospy.get_param('~path_center', [0.45, 0.0, 0.45])
        radius = rospy.get_param('~path_radius', 0.05)
        normal = rospy.get_param('~path_normal', [0.0, 0.0, 1.0])
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        self.follower = ReactivePathFollower(self.path, Ka=1.0)
        self.lap_counter = LapCounter()

        # Human admittance (same law/params as shared_control_node.py so
        # the human side is identical across conditions).
        self.M_h = np.array(rospy.get_param('~admittance_mass', [1.0, 1.0, 1.0]))
        self.B_h = np.array(rospy.get_param('~admittance_damping', [10.0, 10.0, 10.0]))
        self.deadzone_N = rospy.get_param('~force_deadzone_N', 1.0)
        self.alpha_f = rospy.get_param('~force_lpf_alpha', 0.1)
        self.v_h = np.zeros(3)
        self.f_filtered = np.zeros(3)

        # --- Impedance-AAN parameters (PLACEHOLDERS -- set from [9]) ---
        self.K = float(rospy.get_param('~impedance_stiffness', 300.0))   # N/m
        self.D = float(rospy.get_param('~impedance_damping', 40.0))      # N.s/m
        self.adm_gain = float(rospy.get_param('~assist_admittance_gain', 0.002))
        self.deadband_m = float(rospy.get_param('~deadband_m', 0.01))    # AAN tube radius
        self.assist_ramp = float(rospy.get_param('~assist_ramp', 1.0))   # 1/m beyond tube
        self.v_max = float(rospy.get_param('~v_max', 0.15))
        self.lpf_alpha = float(rospy.get_param('~lpf_alpha', 0.2))

        self.v_s_filt = np.zeros(3)
        self.x = np.zeros(3)
        self.x_prev = None
        self.have_robot_state = False

        self.cmd_pub = rospy.Publisher(self.cmd_topic, TwistStamped, queue_size=1)
        self.diag = {
            'condition': rospy.Publisher('~diag/condition', String,
                                          queue_size=1, latch=True),
            'v_h': rospy.Publisher('~diag/v_h', Vector3Stamped, queue_size=1),
            'v_r': rospy.Publisher('~diag/v_r', Vector3Stamped, queue_size=1),
            'v_s': rospy.Publisher('~diag/v_s', Vector3Stamped, queue_size=1),
            'force': rospy.Publisher('~diag/force', Vector3Stamped, queue_size=1),
            'path_progress': rospy.Publisher('~diag/path_progress',
                                              Float64MultiArray, queue_size=1),
        }
        self.diag['condition'].publish(String(data='F_impedance_aan'))

        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)

        self.rate_hz = rospy.get_param('~rate_hz', 200.0)
        self.dt = 1.0 / self.rate_hz
        rospy.loginfo('baseline_aan_node: condition F (impedance-AAN, '
                       'Zhang et al. [9]) -- verify K/D/dead-band against '
                       '[9] and pre-register before recording.')

    def _franka_state_cb(self, msg):
        f_ext = -np.array(msg.O_F_ext_hat_K[0:3])
        f_norm = np.linalg.norm(f_ext)
        f_dz = (np.zeros(3) if f_norm < self.deadzone_N
                else f_ext * (1.0 - self.deadzone_N / f_norm))
        self.f_filtered = (self.alpha_f * f_dz
                            + (1.0 - self.alpha_f) * self.f_filtered)
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
            if not self.have_robot_state:
                rate.sleep()
                continue

            # Human command (admittance).
            accel = (self.f_filtered - self.B_h * self.v_h) / self.M_h
            self.v_h = self.v_h + accel * self.dt

            # Reference + measured EE velocity.
            x_d, tangent, best_s = self.follower.next_goal(self.x)
            x_dot = (np.zeros(3) if self.x_prev is None
                     else (self.x - self.x_prev) / self.dt)
            self.x_prev = self.x.copy()

            s_near, cross_track = self.follower.progress(self.x)
            lap = self.lap_counter.update(s_near)

            # Assist-as-needed gate: zero inside the tube, linear ramp
            # outside it (replace with [9]'s performance-based law).
            err = x_d - self.x
            err_norm = np.linalg.norm(err)
            excess = max(0.0, err_norm - self.deadband_m)
            gate = min(1.0, self.assist_ramp * excess)

            # Virtual impedance -> assistive force -> velocity via a
            # simple admittance (velocity interface only).
            f_assist = gate * (self.K * err - self.D * x_dot)
            v_assist = self.adm_gain * f_assist

            v_s = self.v_h + v_assist
            self.v_s_filt = (self.lpf_alpha * v_s
                              + (1.0 - self.lpf_alpha) * self.v_s_filt)
            speed = np.linalg.norm(self.v_s_filt)
            v_out = (self.v_s_filt if speed <= self.v_max
                     else self.v_s_filt * self.v_max / max(speed, 1e-9))

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
            self._vec3('force', self.f_filtered, stamp)
            self.diag['path_progress'].publish(Float64MultiArray(
                data=[float(s_near), float(lap),
                      float(lap) + float(s_near), float(cross_track)]))

            rate.sleep()


if __name__ == '__main__':
    try:
        BaselineAANNode().run()
    except rospy.ROSInterruptException:
        pass
