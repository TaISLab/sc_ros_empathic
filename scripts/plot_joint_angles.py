#!/usr/bin/env python3
"""
plot_joint_angles.py
--------------------
Live plot of the human arm's four joint angles q1..q4 (degrees) with
each joint's [min, max] range drawn as a shaded band IN THE SAME COLOUR
as the joint's trace -- the colour match rqt_plot cannot do.

  q1  shoulder flex/ext      q3  shoulder int/ext rotation
  q2  shoulder abd/add       q4  elbow flex/ext

A second y-axis (right) carries the human joint-safety efficiency
factor -- factors_h[2], "joint_safety", in (0, 1]: 1 = no joint-limit
penalty, -> 0 as any joint nears a limit. Watch it drop as a trace
enters its coloured band. NaN (condition without joint_safety, or no
fresh human state) simply leaves a gap.

Subscribes:
  ~joint_deg_topic         (std_msgs/Float64MultiArray, data = q1..q4 deg)
      default /shared_control_node/diag/joint_deg
  ~joint_deg_limits_topic  (std_msgs/Float64MultiArray, latched,
                            data = [q1min,q1max, ... q4min,q4max] deg)
      default /shared_control_node/diag/joint_deg_limits
  ~factors_h_topic         (std_msgs/Float64MultiArray, data =
                            [smoothness, directness, joint_safety, manip])
      default /shared_control_node/diag/factors_h

Params:
  ~window_s          rolling time window shown (s), default 30
  ~redraw_hz         figure redraw rate (Hz),      default 15
  ~show_joint_safety draw the eta3 trace,          default true

For the raw time series without the bands, rqt_plot is still fine:
  rqt_plot /shared_control_node/diag/joint_deg/data[0]:data[1]:data[2]:data[3]
"""

import collections
import math
import threading

import rospy
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt

# q1..q4 colours -- reused for the trace, the band and the limit lines.
JOINT_COLOURS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd')
JOINT_LABELS = ('q1 shoulder flex/ext', 'q2 shoulder abd/add',
                'q3 shoulder rot', 'q4 elbow')
ETA3_COLOUR = '#111111'
FACTORS_H_JOINT_SAFETY_IDX = 2   # [smoothness, directness, joint_safety, manip]


class JointAnglePlot(object):
    def __init__(self):
        rospy.init_node('sc_joint_angle_plot', anonymous=False)

        self.window_s = float(rospy.get_param('~window_s', 30.0))
        self.redraw_hz = float(rospy.get_param('~redraw_hz', 15.0))
        self.show_eta3 = bool(rospy.get_param('~show_joint_safety', True))
        deg_topic = rospy.get_param(
            '~joint_deg_topic', '/shared_control_node/diag/joint_deg')
        lim_topic = rospy.get_param(
            '~joint_deg_limits_topic',
            '/shared_control_node/diag/joint_deg_limits')
        fh_topic = rospy.get_param(
            '~factors_h_topic', '/shared_control_node/diag/factors_h')

        # ROS delivers each callback on its own thread; the draw loop
        # runs on the main thread. All deque access is under this lock
        # (otherwise a mid-snapshot append leaves the x/y arrays one
        # sample apart and matplotlib raises a broadcast error).
        self._lock = threading.Lock()
        self.t = collections.deque()
        self.q = [collections.deque() for _ in range(4)]
        self.limits = None            # [(min,max)] x4, degrees
        self._t0 = None
        # eta3 keeps its own (t, value) history: it arrives on a
        # separate topic and may be absent for whole runs.
        self.eta3_t = collections.deque()
        self.eta3 = collections.deque()

        rospy.Subscriber(deg_topic, Float64MultiArray, self._deg_cb,
                         queue_size=5)
        rospy.Subscriber(lim_topic, Float64MultiArray, self._lim_cb,
                         queue_size=1)
        if self.show_eta3:
            rospy.Subscriber(fh_topic, Float64MultiArray, self._fh_cb,
                             queue_size=5)

        self.fig, self.ax = plt.subplots(figsize=(9, 5))
        self.ax.set_xlabel('t (s)')
        self.ax.set_ylabel('joint angle (deg)')
        self.ax.set_title('Human joint angles vs limits')
        self.lines = [
            self.ax.plot([], [], color=JOINT_COLOURS[i], lw=1.6,
                         label=JOINT_LABELS[i])[0]
            for i in range(4)
        ]
        self._band_artists = []
        self.ax.grid(True, alpha=0.3)

        self.ax2 = None
        self.eta3_line = None
        if self.show_eta3:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel('eta3  joint_safety')
            self.ax2.set_ylim(-0.02, 1.05)
            (self.eta3_line,) = self.ax2.plot(
                [], [], color=ETA3_COLOUR, lw=2.2, ls='-',
                label='eta3 joint_safety')

        handles = list(self.lines) + (
            [self.eta3_line] if self.eta3_line is not None else [])
        self.ax.legend(handles=handles, loc='upper left', fontsize=8, ncol=2)

    def _deg_cb(self, msg):
        if len(msg.data) < 4:
            return
        vals = [float(msg.data[i]) for i in range(4)]
        with self._lock:
            self.t.append(self._elapsed())
            for i in range(4):
                self.q[i].append(vals[i])
            self._trim(self.t, self.q)

    def _fh_cb(self, msg):
        if len(msg.data) <= FACTORS_H_JOINT_SAFETY_IDX:
            return
        v = float(msg.data[FACTORS_H_JOINT_SAFETY_IDX])
        if not math.isfinite(v):
            return
        with self._lock:
            self.eta3_t.append(self._elapsed())
            self.eta3.append(v)
            self._trim(self.eta3_t, [self.eta3])

    def _lim_cb(self, msg):
        if len(msg.data) >= 8:
            lims = [(float(msg.data[2 * i]), float(msg.data[2 * i + 1]))
                    for i in range(4)]
            with self._lock:
                self.limits = lims

    def _elapsed(self):
        now = rospy.Time.now().to_sec()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    def _trim(self, tq, series):
        if not tq:
            return
        cutoff = tq[-1] - self.window_s
        while tq and tq[0] < cutoff:
            tq.popleft()
            for s in series:
                s.popleft()

    def _draw_bands(self, limits):
        for a in self._band_artists:
            a.remove()
        self._band_artists = []
        if not limits:
            return
        for i, (lo, hi) in enumerate(limits):
            c = JOINT_COLOURS[i]
            self._band_artists.append(
                self.ax.axhspan(lo, hi, color=c, alpha=0.08, zorder=0))
            for y in (lo, hi):
                self._band_artists.append(
                    self.ax.axhline(y, color=c, ls='--', lw=1.0, alpha=0.7,
                                    zorder=1))

    def _snapshot(self):
        """Consistent copy of the plot data, taken under the lock."""
        with self._lock:
            tt = list(self.t)
            qq = [list(d) for d in self.q]
            e_t = list(self.eta3_t)
            e_v = list(self.eta3)
            lims = list(self.limits) if self.limits else None
        return tt, qq, e_t, e_v, lims

    def spin(self):
        plt.ion()
        plt.show()
        rate = rospy.Rate(self.redraw_hz)
        while not rospy.is_shutdown():
            tt, qq, e_t, e_v, lims = self._snapshot()
            last = max((s[-1] for s in (tt, e_t) if s), default=None)
            if last is not None:
                for i in range(4):
                    n = min(len(tt), len(qq[i]))
                    self.lines[i].set_data(tt[:n], qq[i][:n])
                if self.eta3_line is not None and e_t:
                    n = min(len(e_t), len(e_v))
                    self.eta3_line.set_data(e_t[:n], e_v[:n])
                self._draw_bands(lims)
                self.ax.set_xlim(max(0.0, last - self.window_s),
                                 max(self.window_s, last))
                self.ax.relim()
                self.ax.autoscale_view(scalex=False, scaley=True)
            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                break
            rate.sleep()


if __name__ == '__main__':
    try:
        JointAnglePlot().spin()
    except rospy.ROSInterruptException:
        pass
