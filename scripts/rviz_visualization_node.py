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
  * The current end-effector position (SPHERE), updated from
    franka_states' O_T_EE every time a new robot state arrives.

Publishes a single visualization_msgs/MarkerArray to ~viz_topic
(default /sc_ros_empathic/viz). Add a MarkerArray display in RViz
pointed at that topic, Fixed Frame = ~base_frame (default fr3_link0).
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from sc_ros_empathic.path_follower import CirclePath


class RvizVisualizationNode(object):
    def __init__(self):
        rospy.init_node('sc_ros_empathic_viz', anonymous=False)

        self.base_frame = rospy.get_param('~base_frame', 'fr3_link0')
        self.franka_states_topic = rospy.get_param(
            '~franka_states_topic', '/franka_state_controller/franka_states')
        self.viz_topic = rospy.get_param('~viz_topic', '~viz')

        # Same path definition as shared_control_node.py -- keep these
        # params in sync (pass the same launch args to both nodes)
        # rather than hardcoding the circle twice.
        center = rospy.get_param('~path_center', [0.45, 0.0, 0.45])
        radius = rospy.get_param('~path_radius', 0.05)
        normal = rospy.get_param('~path_normal', [0.0, 0.0, 1.0])
        self.path = CirclePath(center=center, radius=radius, normal=normal)
        self.path_points = [self.path.point(s)
                             for s in np.linspace(0.0, 1.0, 96, endpoint=True)]

        # Also draw the disc spanned by the circle, so its PLANE and
        # orientation are visible in RViz (a bare outline gives no depth
        # cue -- a horizontal circle and a frontal one can project to
        # the same ellipse on screen). Translucent; outline stays on top.
        self.show_plane = bool(rospy.get_param('~show_plane', True))

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
        self.x_actual = np.array([msg.O_T_EE[12], msg.O_T_EE[13], msg.O_T_EE[14]])
        self._publish_ee_marker()

    def _publish_path_marker(self, _event=None):
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = 'sc_ros_empathic'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.004  # line width, m
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.6
        m.color.b = 1.0
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
        d.color.a = 0.12
        d.color.r = 0.0
        d.color.g = 0.6
        d.color.b = 1.0
        c = self.path.center
        c_pt = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
        n = len(self.path_points) - 1
        for i in range(n):
            p0, p1 = self.path_points[i], self.path_points[i + 1]
            d.points.append(c_pt)
            d.points.append(Point(x=p0[0], y=p0[1], z=p0[2]))
            d.points.append(Point(x=p1[0], y=p1[1], z=p1[2]))
        return d

    def _publish_ee_marker(self):
        if self.x_actual is None:
            return
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = rospy.Time.now()
        m.ns = 'sc_ros_empathic'
        m.id = 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.x_actual[0])
        m.pose.position.y = float(self.x_actual[1])
        m.pose.position.z = float(self.x_actual[2])
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.03
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0

        array = MarkerArray()
        array.markers.append(m)
        self.viz_pub.publish(array)


if __name__ == '__main__':
    try:
        RvizVisualizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
