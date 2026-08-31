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
  * v_h / v_r / v_s as ARROW markers rooted at the EE (from
    shared_control_node's ~diag/v_* topics), scaled by ~vel_arrow_gain
    metres per m/s. Colours: v_h green, v_r blue, v_s red.
    Disable with ~show_vel_arrows:=false.
  * eta_h / eta_r / eta_s as a floating TEXT marker above the EE (from
    ~eta). Disable with ~show_eta_text:=false. For a time plot use
    rqt_plot, not RViz.
  * Per-joint gauges (from ~diag/joint_rho / joint_margins /
    joint_limits): each human joint q_i on a bar from q_min to q_max
    with a marker at the current value, coloured by margin -- a live
    check of what the joint-safety efficiency is computed from.
    ~show_joint_gauges:=false hides them; ~joint_gauge_origin places
    them.

The human arm is NOT drawn here: the visuo-tactile pipeline already
publishes it (/skeleton_3D/connectors marker, /skeleton_3D/keypoints
point cloud, /right_arm_description URDF). rviz/shared_control.rviz has
displays for those.

Publishes a single visualization_msgs/MarkerArray to ~viz_topic
(default /sc_ros_empathic/viz -- an ABSOLUTE name so it does not depend
on this node's name). Fixed Frame in RViz = ~base_frame.
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Point, Vector3Stamped
from std_msgs.msg import ColorRGBA, Float64MultiArray, Header
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

        # v_h / v_r / v_s arrows + eta text, from shared_control_node.
        self.show_vel_arrows = bool(rospy.get_param('~show_vel_arrows', True))
        self.show_eta_text = bool(rospy.get_param('~show_eta_text', True))
        self.vel_arrow_gain = float(rospy.get_param('~vel_arrow_gain', 2.0))
        # Per-joint gauges: each human joint on a bar from q_min to
        # q_max, a marker at q_i, coloured by margin -- a live check of
        # what the joint-safety efficiency is computed from.
        self.show_joint_gauges = bool(rospy.get_param('~show_joint_gauges', True))
        self.gauge_origin = rospy.get_param('~joint_gauge_origin',
                                            [0.18, 0.18, 0.45])
        self.gauge_len = float(rospy.get_param('~joint_gauge_len', 0.12))
        sc = rospy.get_param('~sc_node', '/shared_control_node')
        self._v = {'v_h': None, 'v_r': None, 'v_s': None}
        self._eta = None
        self._jrho = None
        self._jmargin = None
        self._jlim = None

        self.x_actual = None

        self.viz_pub = rospy.Publisher(self.viz_topic, MarkerArray, queue_size=1)
        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)
        if self.show_vel_arrows:
            rospy.Subscriber(sc + '/diag/v_h', Vector3Stamped, self._vh_cb, queue_size=1)
            rospy.Subscriber(sc + '/diag/v_r', Vector3Stamped, self._vr_cb, queue_size=1)
            rospy.Subscriber(sc + '/diag/v_s', Vector3Stamped, self._vs_cb, queue_size=1)
        if self.show_eta_text:
            rospy.Subscriber(sc + '/eta', Float64MultiArray, self._eta_cb, queue_size=1)
        if self.show_joint_gauges:
            rospy.Subscriber(sc + '/diag/joint_rho', Float64MultiArray,
                              lambda m: setattr(self, '_jrho', list(m.data)),
                              queue_size=1)
            rospy.Subscriber(sc + '/diag/joint_margins', Float64MultiArray,
                              lambda m: setattr(self, '_jmargin', list(m.data)),
                              queue_size=1)
            rospy.Subscriber(sc + '/diag/joint_limits', Float64MultiArray,
                              lambda m: setattr(self, '_jlim', list(m.data)),
                              queue_size=1)

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

    def _vh_cb(self, m):
        self._v['v_h'] = m.vector

    def _vr_cb(self, m):
        self._v['v_r'] = m.vector

    def _vs_cb(self, m):
        self._v['v_s'] = m.vector

    def _eta_cb(self, m):
        self._eta = list(m.data)

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

    def _joint_gauge_markers(self):
        """Per-joint bar (q_min..q_max) with a marker at q_i and a
        text label. Uses ~diag/joint_rho (position in [-1,1]),
        joint_margins (colour) and joint_limits (label numbers)."""
        out = []
        rho = self._jrho or []
        if not rho:
            for i in range(4):                      # nothing yet -> clear
                for off in (0, 1, 2):
                    m = Marker()
                    m.header = self._header()
                    m.ns = 'sc_ros_empathic'
                    m.id = 40 + 3 * i + off
                    m.action = Marker.DELETE
                    out.append(m)
            return out
        margin = self._jmargin or [1.0] * 4
        lim = self._jlim or []
        ox, oy, oz = (float(self.gauge_origin[0]), float(self.gauge_origin[1]),
                      float(self.gauge_origin[2]))
        L = self.gauge_len
        dy = 0.035
        for i in range(min(4, len(rho))):
            y = oy - i * dy
            r = max(-1.2, min(1.2, float(rho[i])))
            mrg = float(margin[i]) if i < len(margin) else 1.0
            col = ((0.1, 0.9, 0.1, 1.0) if mrg > 0.3 else
                   (1.0, 0.85, 0.0, 1.0) if mrg > 0.1 else (1.0, 0.1, 0.1, 1.0))

            bar = Marker()
            bar.header = self._header()
            bar.ns = 'sc_ros_empathic'
            bar.id = 40 + 3 * i
            bar.type = Marker.LINE_STRIP
            bar.action = Marker.ADD
            bar.pose.orientation.w = 1.0
            bar.scale.x = 0.004
            bar.color = ColorRGBA(0.6, 0.6, 0.6, 1.0)
            for xx in (ox, ox + 0.15 * L, ox + 0.5 * L, ox + 0.85 * L, ox + L):
                bar.points.append(Point(x=xx, y=y, z=oz))
            out.append(bar)

            cur = Marker()
            cur.header = self._header()
            cur.ns = 'sc_ros_empathic'
            cur.id = 40 + 3 * i + 1
            cur.type = Marker.CUBE
            cur.action = Marker.ADD
            cur.pose.position.x = ox + 0.5 * (r + 1.0) * L
            cur.pose.position.y = y
            cur.pose.position.z = oz
            cur.pose.orientation.w = 1.0
            cur.scale.x = 0.006
            cur.scale.y = 0.018
            cur.scale.z = 0.018
            cur.color = ColorRGBA(*col)
            out.append(cur)

            txt = Marker()
            txt.header = self._header()
            txt.ns = 'sc_ros_empathic'
            txt.id = 40 + 3 * i + 2
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = ox + L + 0.02
            txt.pose.position.y = y
            txt.pose.position.z = oz
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.02
            txt.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            if len(lim) >= 2 * (i + 1):
                qmid = 0.5 * (lim[2 * i] + lim[2 * i + 1])
                qhalf = 0.5 * (lim[2 * i + 1] - lim[2 * i])
                q = qmid + r * qhalf
                txt.text = ('q%d %+.2f rad  [%.2f, %.2f]  m=%.2f'
                            % (i + 1, q, lim[2 * i], lim[2 * i + 1], mrg))
            else:
                txt.text = 'q%d  rho %+.2f  m=%.2f' % (i + 1, r, mrg)
            out.append(txt)
        return out

    def _publish_ee_marker(self):
        array = MarkerArray()

        if self.show_joint_gauges:
            array.markers.extend(self._joint_gauge_markers())

        if self.x_actual is None:
            self.viz_pub.publish(array)
            return

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

        if self.show_vel_arrows:
            x = self.x_actual
            for mid, key, rgba in ((10, 'v_h', (0.0, 0.85, 0.0, 1.0)),
                                   (11, 'v_r', (0.2, 0.45, 1.0, 1.0)),
                                   (12, 'v_s', (1.0, 0.0, 0.0, 1.0))):
                a = self._arrow_marker(mid, x, self._v[key], rgba)
                if a is not None:
                    array.markers.append(a)

        if self.show_eta_text:
            array.markers.append(self._eta_text_marker(self.x_actual))

        self.viz_pub.publish(array)

    def _arrow_marker(self, mid, origin, vec, rgba):
        a = Marker()
        a.header = self._header()
        a.ns = 'sc_ros_empathic'
        a.id = mid
        a.type = Marker.ARROW
        if vec is None:
            a.action = Marker.DELETE
            return a
        g = self.vel_arrow_gain
        tip = (origin[0] + vec.x * g, origin[1] + vec.y * g,
               origin[2] + vec.z * g)
        if (tip[0] - origin[0]) ** 2 + (tip[1] - origin[1]) ** 2 \
                + (tip[2] - origin[2]) ** 2 < 1e-8:
            a.action = Marker.DELETE
            return a
        a.action = Marker.ADD
        a.pose.orientation.w = 1.0
        a.points = [Point(x=float(origin[0]), y=float(origin[1]),
                          z=float(origin[2])),
                    Point(x=float(tip[0]), y=float(tip[1]), z=float(tip[2]))]
        a.scale.x = 0.006   # shaft diameter
        a.scale.y = 0.013   # head diameter
        a.scale.z = 0.02    # head length
        a.color = ColorRGBA(*rgba)
        return a

    def _eta_text_marker(self, origin):
        m = Marker()
        m.header = self._header()
        m.ns = 'sc_ros_empathic'
        m.id = 20
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(origin[0])
        m.pose.position.y = float(origin[1])
        m.pose.position.z = float(origin[2]) + 0.09
        m.pose.orientation.w = 1.0
        m.scale.z = 0.03    # text height, m
        m.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
        e = self._eta
        if e is not None and len(e) >= 3:
            m.text = 'eta_h %.2f   eta_r %.2f   eta_s %.2f' % (e[0], e[1], e[2])
        else:
            m.text = 'eta: (waiting for /eta)'
        return m


if __name__ == '__main__':
    try:
        RvizVisualizationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
