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
  * The human arm as a SPHERE_LIST + LINE_STRIP polyline (cyan) through
    the pipeline's shoulder/elbow/wrist keypoints, in order. Source:
    ~human_arm_source = 'topic' (default: named keypoints in
    ~human_arm_topic, assumed visualization_msgs/MarkerArray, picked by
    ~human_arm_keys) or 'tf' (origins of ~human_arm_frames, needs
    tf2_ros). A live polyline = the human is being detected.
    ~show_arm:=false hides it. Optional magenta ~diag/arm_points_fk
    overlay with ~show_arm_fk:=true.

Publishes a single visualization_msgs/MarkerArray to ~viz_topic
(default /sc_ros_empathic/viz -- an ABSOLUTE name so it does not depend
on this node's name). Fixed Frame in RViz = ~base_frame (fr3_link0).
rviz/shared_control.rviz already has a MarkerArray display on that
topic.
"""

import numpy as np
import rospy
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Point, PoseArray, Vector3Stamped
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
        # Human arm polyline through the pipeline's shoulder/elbow/wrist
        # keypoints. ~human_arm_source:
        #   'topic' (default): named keypoints in ~human_arm_topic
        #       (assumed visualization_msgs/MarkerArray; the joint name
        #        is taken from marker.ns or marker.text or 'joint<id>'),
        #        picked in the order ~human_arm_keys.
        #   'tf': the origins of the frames listed in ~human_arm_frames.
        self.show_arm = bool(rospy.get_param('~show_arm', True))
        self.arm_source = str(rospy.get_param('~human_arm_source', 'topic'))
        self.arm_topic = rospy.get_param('~human_arm_topic',
                                         '/right_arm/kp_URDF')
        self.arm_keys = rospy.get_param(
            '~human_arm_keys', ['right_shoulder', 'right_elbow', 'right_wrist'])
        self.arm_frames = rospy.get_param('~human_arm_frames', [])
        for _a in ('arm_keys', 'arm_frames'):
            v = getattr(self, _a)
            if isinstance(v, str):
                setattr(self, _a, [s for s in v.replace(',', ' ').split() if s])
        self.show_arm_fk = bool(rospy.get_param('~show_arm_fk', False))
        sc = rospy.get_param('~sc_node', '/shared_control_node')
        self._v = {'v_h': None, 'v_r': None, 'v_s': None}
        self._eta = None
        self._arm_fk = None
        self._arm_kp = None            # latest keypoints MarkerArray

        self.x_actual = None
        self.tf_buffer = None
        if self.show_arm and self.arm_source == 'tf':
            import tf2_ros
            self._tf2 = tf2_ros
            self.tf_buffer = tf2_ros.Buffer()
            self._tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        if self.show_arm and self.arm_source == 'topic':
            rospy.Subscriber(self.arm_topic, MarkerArray, self._arm_kp_cb,
                              queue_size=1)

        self.viz_pub = rospy.Publisher(self.viz_topic, MarkerArray, queue_size=1)
        rospy.Subscriber(self.franka_states_topic, FrankaState,
                          self._franka_state_cb, queue_size=1)
        if self.show_vel_arrows:
            rospy.Subscriber(sc + '/diag/v_h', Vector3Stamped, self._vh_cb, queue_size=1)
            rospy.Subscriber(sc + '/diag/v_r', Vector3Stamped, self._vr_cb, queue_size=1)
            rospy.Subscriber(sc + '/diag/v_s', Vector3Stamped, self._vs_cb, queue_size=1)
        if self.show_eta_text:
            rospy.Subscriber(sc + '/eta', Float64MultiArray, self._eta_cb, queue_size=1)
        if self.show_arm and self.show_arm_fk:
            rospy.Subscriber(sc + '/diag/arm_points_fk', PoseArray,
                              self._arm_fk_cb, queue_size=1)

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

    def _arm_fk_cb(self, msg):
        self._arm_fk = msg

    def _stick(self, base_id, frame_id, pts_xyz, rgba):
        """LINE_STRIP + SPHERE_LIST through the given points in
        `frame_id`. < 2 points -> DELETE both markers."""
        line = Marker()
        line.header = self._header()
        line.ns = 'sc_ros_empathic'
        line.id = base_id
        line.type = Marker.LINE_STRIP
        sph = Marker()
        sph.header = self._header()
        sph.ns = 'sc_ros_empathic'
        sph.id = base_id + 1
        sph.type = Marker.SPHERE_LIST
        if not pts_xyz or len(pts_xyz) < 2:
            line.action = sph.action = Marker.DELETE
            return [line, sph]
        line.header.frame_id = sph.header.frame_id = frame_id
        line.action = sph.action = Marker.ADD
        line.pose.orientation.w = sph.pose.orientation.w = 1.0
        line.scale.x = 0.008
        sph.scale.x = sph.scale.y = sph.scale.z = 0.025
        line.color = sph.color = ColorRGBA(*rgba)
        for p in pts_xyz:
            q = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            line.points.append(q)
            sph.points.append(q)
        return [line, sph]

    def _arm_kp_cb(self, msg):
        self._arm_kp = msg

    @staticmethod
    def _marker_name(m):
        return (m.ns or getattr(m, 'text', '') or 'joint%d' % m.id).strip()

    def _arm_kp_points(self):
        """(points, frame_id): the ~human_arm_keys keypoints from the
        latest ~human_arm_topic MarkerArray, in order. ([], base) if
        missing keys."""
        msg = self._arm_kp
        if msg is None:
            rospy.logwarn_throttle(5.0, 'sc_ros_empathic_viz: no message on '
                                   '%s yet -- human not detected?',
                                   self.arm_topic)
            return [], self.base_frame
        by_name = {self._marker_name(m): m for m in msg.markers}
        frame = (msg.markers[0].header.frame_id if msg.markers
                 else self.base_frame)
        out = []
        for k in self.arm_keys:
            m = by_name.get(k)
            if m is None:
                rospy.logwarn_throttle(
                    5.0, 'sc_ros_empathic_viz: keypoint %r not in %s '
                    '(have: %s)', k, self.arm_topic, sorted(by_name))
                return [], frame
            out.append((m.pose.position.x, m.pose.position.y,
                        m.pose.position.z))
        return out, frame

    def _arm_tf_points(self):
        """The ~human_arm_frames origins in ~base_frame, in order, or []
        if any lookup fails."""
        out = []
        for fr in self.arm_frames:
            try:
                tr = self.tf_buffer.lookup_transform(
                    self.base_frame, fr, rospy.Time(0), rospy.Duration(0.0))
            except self._tf2.TransformException:
                rospy.logwarn_throttle(
                    5.0, 'sc_ros_empathic_viz: TF %s -> %s missing -- human '
                    'not detected, or wrong name in ~human_arm_frames %s',
                    self.base_frame, fr, self.arm_frames)
                return []
            t = tr.transform.translation
            out.append((t.x, t.y, t.z))
        return out

    def _publish_ee_marker(self):
        array = MarkerArray()

        if self.show_arm:
            if self.arm_source == 'tf':
                pts, frame = self._arm_tf_points(), self.base_frame
            else:
                pts, frame = self._arm_kp_points()
            array.markers.extend(self._stick(30, frame, pts,
                                             (0.0, 0.9, 0.9, 1.0)))
            if self.show_arm_fk:
                fk = self._arm_fk
                fk_pts = ([(p.position.x, p.position.y, p.position.z)
                           for p in fk.poses] if fk and len(fk.poses) >= 2
                          else [])
                array.markers.extend(self._stick(
                    32, (fk.header.frame_id if fk else self.base_frame),
                    fk_pts, (1.0, 0.3, 0.9, 1.0)))

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
