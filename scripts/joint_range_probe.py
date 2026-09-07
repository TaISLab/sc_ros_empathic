#!/usr/bin/env python3
"""
joint_range_probe.py
--------------------
Record the per-joint min / max the participant actually reaches while
doing the task (or a free range-of-motion sweep), so
config/subjects/SXX.yaml `joint_limits_rad` can be set from real data
instead of the study defaults.

Subscribes to ~topic (default /right_arm/joint_states, sensor_msgs/
JointState), indexes right_arm_q1..q4 by name, applies the same
q4 <- pi - right_arm_q4 conversion shared_control_node does, and tracks
the running min / max. Live readout ~2 Hz; a final table plus a
ready-to-paste `joint_limits_rad:` block on Ctrl-C.

Only the visuo-tactile pipeline needs to be running -- not the full
shared-control launch.

  rosrun sc_ros_empathic joint_range_probe.py
  rosrun sc_ros_empathic joint_range_probe.py _margin_deg:=5
  rosrun sc_ros_empathic joint_range_probe.py _topic:=/right_arm/joint_states
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
        self.lo = np.full(4, np.inf)
        self.hi = np.full(4, -np.inf)
        self.n = 0
        rospy.Subscriber(self.topic, JointState, self._cb, queue_size=20)
        rospy.Timer(rospy.Duration(0.5), self._show)
        rospy.on_shutdown(self._final)
        rospy.loginfo('joint_range_probe: recording from %s -- move through '
                      'the whole task / range of motion, Ctrl-C when done',
                      self.topic)

    def _cb(self, m):
        idx = {nm: i for i, nm in enumerate(m.name)}
        if not all(k in idx and idx[k] < len(m.position) for k in QN):
            return
        q = np.array([m.position[idx[k]] for k in QN], dtype=float)
        q[3] = np.pi - q[3]                      # match shared_control_node
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
