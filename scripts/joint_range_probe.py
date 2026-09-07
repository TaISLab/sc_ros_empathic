#!/usr/bin/env python3
"""
joint_range_probe.py
--------------------
Record the per-joint min / max the participant actually reaches while
doing the task (or a free range-of-motion sweep), so
config/subjects/SXX.yaml `joint_limits_rad` can be set from real data
instead of the study defaults.

Subscribes to ~topic (default /right_arm/joint_states, sensor_msgs/
JointState), indexes right_arm_q1..q4 by name, and applies the SAME
transforms shared_control_node does so the numbers are directly
paste-ready into config/subjects/SXX.yaml joint_limits_rad:
  * q4  <- pi - right_arm_q4         (pipeline sends the elbow interior angle)
  * q_i <- gain_i * (q_i - offset_i) (~human_joint_offsets / ~human_joint_gains,
                                      read from the running node or overridable)
Then tracks the running min / max. Live readout ~2 Hz; a final table
plus the ready-to-paste block on Ctrl-C.

Only the visuo-tactile pipeline needs to be running (the offsets are
read from /shared_control_node's params if that node is up, else from
this node's ~human_joint_offsets / ~human_joint_gains, else identity).

Do a DELIBERATE full-range sweep of EACH joint to its comfortable
extremes (it records all four continuously and cannot tell a real
sweep from incidental motion). Restart it for a clean session.

  rosrun sc_ros_empathic joint_range_probe.py
  rosrun sc_ros_empathic joint_range_probe.py _margin_deg:=5
"""
import numpy as np
import rospy
from sensor_msgs.msg import JointState

QN = ('right_arm_q1', 'right_arm_q2', 'right_arm_q3', 'right_arm_q4')
LBL = ('q1 shoulder flex/ext', 'q2 shoulder abd/add',
       'q3 shoulder int/ext rot', 'q4 elbow flex (0=ext)')


class Probe(object):
    def __init__(self):
        rospy.init_node('joint_range_probe', anonymous=True)
        self.topic = rospy.get_param('~topic', '/right_arm/joint_states')
        self.margin = float(rospy.get_param('~margin_deg', 0.0))
        node = rospy.get_param('~sc_node', '/shared_control_node')
        self.offset = np.asarray(rospy.get_param(
            '~human_joint_offsets',
            rospy.get_param(node + '/human_joint_offsets', [0.0] * 4)), float)
        self.gain = np.asarray(rospy.get_param(
            '~human_joint_gains',
            rospy.get_param(node + '/human_joint_gains', [1.0] * 4)), float)
        self.lo = np.full(4, np.inf)
        self.hi = np.full(4, -np.inf)
        self.n = 0
        rospy.Subscriber(self.topic, JointState, self._cb, queue_size=20)
        rospy.Timer(rospy.Duration(0.5), self._show)
        rospy.on_shutdown(self._final)
        rospy.loginfo('joint_range_probe: recording from %s  (gain=%s '
                      'offset=%s) -- sweep each joint to its extremes, '
                      'Ctrl-C when done', self.topic,
                      self.gain.tolist(), self.offset.round(3).tolist())

    def _cb(self, m):
        idx = {nm: i for i, nm in enumerate(m.name)}
        if not all(k in idx and idx[k] < len(m.position) for k in QN):
            return
        q = np.array([m.position[idx[k]] for k in QN], dtype=float)
        q[3] = np.pi - q[3]                       # elbow interior angle -> flexion
        q = self.gain * (q - self.offset)        # match _human_joint_state_cb
        self.lo = np.minimum(self.lo, q)
        self.hi = np.maximum(self.hi, q)
        self.n += 1

    def _show(self, _evt):
        if self.n == 0:
            return
        parts = ['%s [%+6.1f, %+6.1f]' % (k[9:], np.degrees(a), np.degrees(b))
                 for k, a, b in zip(QN, self.lo, self.hi)]
        rospy.loginfo_throttle(0.5, 'range deg  n=%d  %s',
                               self.n, '  '.join(parts))

    def _final(self):
        if self.n == 0:
            rospy.logwarn('joint_range_probe: no valid samples on %s',
                          self.topic)
            return
        m = np.radians(self.margin)
        print('\n=== observed joint range over %d samples ===' % self.n)
        for lbl, a, b in zip(LBL, self.lo, self.hi):
            print('  %-24s [%+7.1f, %+7.1f] deg   [%+.3f, %+.3f] rad'
                  % (lbl, np.degrees(a), np.degrees(b), a, b))
        print('\n# config/subjects/SXX.yaml  (observed range +/- %.0f deg margin):'
              % self.margin)
        print('joint_limits_rad:')
        for name, a, b in zip(('q1', 'q2', 'q3', 'q4'),
                              self.lo - m, self.hi + m):
            print('  %s: [%.3f, %.3f]' % (name, a, b))


if __name__ == '__main__':
    Probe()
    rospy.spin()
