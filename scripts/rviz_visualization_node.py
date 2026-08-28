#!/usr/bin/env python3
"""
rviz_visualization_node.py
----------------------------
Minimal RViz visualization for the shared-control experiment:

  * The path to trace (LINE_STRIP, static, closed circle) -- uses the
    SAME ~path_center/~path_radius/~path_normal params as
    shared_control_node.py, so it always shows the path actually being
    followed, not a hardcoded copy of it.
  * The disc that circle bounds (TRIANGLE_LIST, translucent), so the
    plane it lies in is visible and not just an ambiguous outline.
    Disable with ~show_plane:=false.
  * The current end-effector position (SPHERE) and the trail it has
    swept (LINE_STRIP, last ~trail_len points), updated from
    franka_states' O_T_EE -- this is the "trajectory followed" overlay.
    Disable the trail with ~trail_len:=0.

Publishes a single visualization_msgs/MarkerArray to ~viz_topic
(default /sc_ros_empathic/viz -- an ABSOLUTE name so it does not depend
on this node's name). Fixed Frame in RViz = ~base_frame (fr3_link0).
rviz/shared_control.rviz already has a MarkerArray display on that
topic.
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from sc_ros_empathic.path_follower import CirclePath


class RvizVisualizationNode(object):
    def __init__(self):
        rospy.init_node('sc_ros_empathic_viz', anonymous=False)

        self.base_frame = rospy.get_param('~base_frame', 'fr3_link0')
        self.franka_states_topic = rospy.get_param(
            '~franka_states_topic', '/franka_state_controller/franka_states')
        self.viz_topic = rospy.get_param('~viz_topic', '/sc_ros_empathic/viz')

        # Same path definition as shared_control_node.py -- keep these
        # params in sync (pass the same launch args to both nodes)
        # rather than hardcoding the circle twice.
        center = rospy.get_param('~path_center', [0.45, 0.0, 0.45])
        radius = rospy.get_param('~path_radius', 0.05)
        normal = rospy.get_param('~path_normal', [0.0, 0.0, 1.0])
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        self.path_points = [self.path.point(s)
                             for s in np.linspace(0.0, 1.0, 96, endpoint=True)]

        self.show_plane = bool(rospy.get_param('~show_plane', True))
        # EE trail (the trajectory actually followed).
        self.trail_len = int(rospy.get_param('~trail_len', 3000))
        self.trail_step = float(rospy.get_param('~trail_step', 0.002))  # m
        self.trail = []

        self.x_actual = None

        self.viz_pub = rospy.Publisher(self.viz_topic, MarkerArray, queue_size=1)
        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)

        # The path itself is static: republish it at a slow rate so a
        # late-joining RViz instance still picks it up without needing
        # a latched publisher.
        self.path_republish_period = rospy.get_param('~path_republish_period', 1.0)
        rospy.Timer(rospy.Duration(self.path_republish_period),
                    self._publish_path_marker)

    def _franka_state_cb(self, msg):
        x = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
        self.x_actual = x
        if self.trail_len > 0 and (
                not self.trail
                or np.linalg.norm(x - self.trail[-1]) >= self.trail_step):
            self.trail.append(x)
            if len(self.trail) > self.trail_len:
                self.trail.pop(0)
        self._publish_ee_marker()

    def _header(self):
        h = Header()
        h.frame_id = self.base_frame
        h.stamp = rospy.Time.now()
        return h

    def _publish_path_marker(self, _event=None):
        m = Marker()
        m.header = self._header()
        m.ns = 'sc_ros_empathic'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.004  # line width, m
        m.color = ColorRGBA(0.0, 0.6, 1.0, 1.0)
        for p in self.path_points:
            m.points.append(Point(x=p[0], y=p[1], z=p[2]))

        array = MarkerArray()
        array.markers.append(m)
        if self.show_plane:
            array.markers.append(self._disc_marker(m.header))
        self.viz_pub.publish(array)

    def _disc_marker(self, header):
        """Translucent triangle-fan filling the circle, so the plane the
        circle lies in (horizontal by default, ~path_normal) is obvious."""
        d = Marker()
        d.header = header
        d.ns = 'sc_ros_empathic'
        d.id = 2
        d.type = Marker.TRIANGLE_LIST
        d.action = Marker.ADD
        d.pose.orientation.w = 1.0
        d.scale.x = d.scale.y = d.scale.z = 1.0
        d.color = ColorRGBA(0.0, 0.6, 1.0, 0.12)
        c = self.path.center
        c_pt = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
        for i in range(len(self.path_points) - 1):
            p0, p1 = self.path_points[i], self.path_points[i + 1]
            d.points.append(c_pt)
            d.points.append(Point(x=p0[0], y=p0[1], z=p0[2]))
            d.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
        return d

    def _publish_ee_marker(self):
        if self.x_actual is None:
            return
        array = MarkerArray()

        s = Marker()
        s.header = self._header()
        s.ns = 'sc_ros_empathic'
        s.id = 1
        s.type = Marker.SPHERE
        s.action = Marker.ADD
        s.pose.position.x = float(self.x_actual[0])
        s.pose.position.y = float(self.x_actual[1])
        s.pose.position.z = float(self.x_actual[2])
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 0.03
        s.color = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        array.markers.append(s)

        if self.trail_len > 0 and len(self.trail) >= 2:
            t = Marker()
            t.header = self._header()
            t.ns = 'sc_ros_empathic'
            t.id = 3
            t.type = Marker.LINE_STRIP
            t.action = Marker.ADD
            t.pose.orientation.w = 1.0
            t.scale.x = 0.003
            t.color = ColorRGBA(1.0, 0.8, 0.0, 1.0)
            for p in self.trail:
                t.points.append(Point(x=float(p[0]), y=float(p[1]),
                                      z=float(p[2])))
            array.markers.append(t)

        self.viz_pub.publish(array)


if __name__ == '__main__':
    try:
        RvizVisualizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
