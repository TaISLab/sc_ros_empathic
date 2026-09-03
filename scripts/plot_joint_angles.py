#!/usr/bin/env python3
"""
plot_joint_angles.py
--------------------
Live plot of the human arm's four joint angles q1..q4 (degrees) with
each joint's [min, max] range drawn as a shaded band IN THE SAME COLOUR
as the joint's trace -- the colour match rqt_plot cannot do.

  q1  shoulder flex/ext      q3  shoulder int/ext rotation
  q2  shoulder abd/add       q4  elbow flex/ext

Per joint, four traces in that joint's colour:
  solid    measured q_i (~diag/joint_deg)
  dashed   q_i extrapolated along the velocity the HUMAN command v_h
           induces          (~diag/joint_deg_future_h)
  dotted   ... the ROBOT command v_r induces
                            (~diag/joint_deg_future_r)
  dash-dot ... the eta-weighted BLEND v_hat_s induces, before eta_s
           scales the output (~diag/joint_deg_future_s)
Each future trace is q_h + qdot_k * dt_lookahead: qdot_k is what
joint_safety's dynamic term scores for candidate k, over the
controller's own lookahead horizon.

A second y-axis (right) carries the three shared-control efficiencies
eta_h / eta_r / eta_s (~eta), same dashed / dotted / dash-dot key,
in black.

Params:
  ~window_s     rolling time window shown (s), default 30
  ~redraw_hz    figure redraw rate (Hz),      default 15
  ~show_future  draw the 3 extrapolation sets, default true
  ~show_eta     draw eta_h/eta_r/eta_s,        default true
"""

import collections
import threading

import rospy
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt

# q1..q4 colours -- reused for every trace and the limit band of a joint.
JOINT_COLOURS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd')
JOINT_LABELS = ('q1 shoulder flex/ext', 'q2 shoulder abd/add',
                'q3 shoulder rot', 'q4 elbow')
ETA_COLOUR = '#111111'
# candidate command -> line style (measured q_h is 'solid')
STYLE = {'meas': '-', 'h': '--', 'r': ':', 's': '-.'}
STYLE_LABEL = {'meas': 'measured', 'h': 'v_h projection',
               'r': 'v_r projection', 's': 'v_sum projection'}


class _MT(object):
    """A time series of n parallel channels held in rolling deques."""
    def __init__(self, n):
        self.t = collections.deque()
        self.v = [collections.deque() for _ in range(n)]


class JointAnglePlot(object):
    def __init__(self):
        rospy.init_node('sc_joint_angle_plot', anonymous=False)

        self.window_s = float(rospy.get_param('~window_s', 30.0))
        self.redraw_hz = float(rospy.get_param('~redraw_hz', 15.0))
        self.show_future = bool(rospy.get_param('~show_future', True))
        self.show_eta = bool(rospy.get_param('~show_eta', True))

        base = '/shared_control_node'
        deg_topic = rospy.get_param('~joint_deg_topic', base + '/diag/joint_deg')
        fut_topics = {
            'h': rospy.get_param('~joint_deg_future_h_topic',
                                 base + '/diag/joint_deg_future_h'),
            'r': rospy.get_param('~joint_deg_future_r_topic',
                                 base + '/diag/joint_deg_future_r'),
            's': rospy.get_param('~joint_deg_future_s_topic',
                                 base + '/diag/joint_deg_future_s'),
        }
        lim_topic = rospy.get_param('~joint_deg_limits_topic',
                                    base + '/diag/joint_deg_limits')
        eta_topic = rospy.get_param('~eta_topic', base + '/eta')

        # ROS delivers each callback on its own thread; the draw loop
        # runs on the main thread. All deque access is under this lock
        # (a mid-snapshot append otherwise leaves x/y one sample apart
        # and matplotlib raises a broadcast error).
        self._lock = threading.Lock()
        self._t0 = None
        self.limits = None                 # [(min,max)] x4, degrees
        self.meas = _MT(4)
        self.fut = {k: _MT(4) for k in ('h', 'r', 's')}
        self.eta = _MT(3)

        rospy.Subscriber(deg_topic, Float64MultiArray, self._deg_cb,
                         queue_size=5)
        rospy.Subscriber(lim_topic, Float64MultiArray, self._lim_cb,
                         queue_size=1)
        if self.show_future:
            for k, topic in fut_topics.items():
                rospy.Subscriber(topic, Float64MultiArray,
                                 lambda m, kk=k: self._fut_cb(kk, m),
                                 queue_size=5)
        if self.show_eta:
            rospy.Subscriber(eta_topic, Float64MultiArray, self._eta_cb,
                             queue_size=5)

        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.ax.set_xlabel('t (s)')
        self.ax.set_ylabel('joint angle (deg)')
        self.ax.set_title('Human joint angles vs limits')
        self.ax.grid(True, alpha=0.3)

        self.meas_lines = [
            self.ax.plot([], [], color=JOINT_COLOURS[i], lw=1.7,
                         ls=STYLE['meas'])[0]
            for i in range(4)
        ]
        self.fut_lines = {}
        if self.show_future:
            for k in ('h', 'r', 's'):
                self.fut_lines[k] = [
                    self.ax.plot([], [], color=JOINT_COLOURS[i], lw=1.2,
                                 ls=STYLE[k], alpha=0.9)[0]
                    for i in range(4)
                ]

        self.ax2 = None
        self.eta_lines = []
        if self.show_eta:
            self.ax2 = self.ax.twinx()
            self.ax2.set_ylabel('eta_h / eta_r / eta_s')
            self.ax2.set_ylim(-0.02, 1.05)
            for j, k in enumerate(('h', 'r', 's')):
                (ln,) = self.ax2.plot([], [], color=ETA_COLOUR, lw=2.0,
                                      ls=STYLE[k])
                self.eta_lines.append(ln)

        self._band_artists = []
        self._build_legend()

    # ---- legend -----------------------------------------------------
    def _build_legend(self):
        handles = [plt.Line2D([], [], color=JOINT_COLOURS[i], lw=2,
                              label=JOINT_LABELS[i]) for i in range(4)]
        keys = ['meas'] + (['h', 'r', 's'] if self.show_future else [])
        for k in keys:
            handles.append(plt.Line2D([], [], color='#666666', lw=1.5,
                                      ls=STYLE[k], label=STYLE_LABEL[k]))
        if self.show_eta:
            for k in ('h', 'r', 's'):
                handles.append(plt.Line2D([], [], color=ETA_COLOUR, lw=2,
                                          ls=STYLE[k], label='eta_' + k))
        self.ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=3)

    # ---- callbacks ------------------------------------------------
    def _elapsed(self):
        now = rospy.Time.now().to_sec()
        if self._t0 is None:
            self._t0 = now
        return now - self._t0

    def _push(self, mt, vals):
        mt.t.append(self._elapsed())
        for i, x in enumerate(vals):
            mt.v[i].append(x)
        cutoff = mt.t[-1] - self.window_s
        while mt.t and mt.t[0] < cutoff:
            mt.t.popleft()
            for d in mt.v:
                d.popleft()

    def _deg_cb(self, msg):
        if len(msg.data) < 4:
            return
        vals = [float(msg.data[i]) for i in range(4)]
        with self._lock:
            self._push(self.meas, vals)

    def _fut_cb(self, key, msg):
        if len(msg.data) < 4:
            return
        vals = [float(msg.data[i]) for i in range(4)]
        with self._lock:
            self._push(self.fut[key], vals)

    def _eta_cb(self, msg):
        if len(msg.data) < 3:
            return
        vals = [float(msg.data[i]) for i in range(3)]
        with self._lock:
            self._push(self.eta, vals)

    def _lim_cb(self, msg):
        if len(msg.data) >= 8:
            lims = [(float(msg.data[2 * i]), float(msg.data[2 * i + 1]))
                    for i in range(4)]
            with self._lock:
                self.limits = lims

    # ---- draw ---------------------------------------------------
    def _draw_bands(self, limits):
        for a in self._band_artists:
            a.remove()
        self._band_artists = []
        if not limits:
            return
        for i, (lo, hi) in enumerate(limits):
            c = JOINT_COLOURS[i]
            self._band_artists.append(
                self.ax.axhspan(lo, hi, color=c, alpha=0.07, zorder=0))
            for y in (lo, hi):
                self._band_artists.append(
                    self.ax.axhline(y, color=c, ls='--', lw=1.0, alpha=0.6,
                                    zorder=1))

    @staticmethod
    def _set(line, t, v):
        n = min(len(t), len(v))
        line.set_data(t[:n], v[:n])

    def _snapshot(self):
        with self._lock:
            def cp(mt):
                return list(mt.t), [list(d) for d in mt.v]
            meas = cp(self.meas)
            fut = {k: cp(self.fut[k]) for k in self.fut} if self.show_future \
                else {}
            eta = cp(self.eta) if self.show_eta else ([], [])
            lims = list(self.limits) if self.limits else None
        return meas, fut, eta, lims

    def spin(self):
        plt.ion()
        plt.show()
        rate = rospy.Rate(self.redraw_hz)
        while not rospy.is_shutdown():
            meas, fut, eta, lims = self._snapshot()
            all_t = [meas[0]] + [fut[k][0] for k in fut] + [eta[0]]
            last = max((s[-1] for s in all_t if s), default=None)
            if last is not None:
                for i in range(4):
                    self._set(self.meas_lines[i], meas[0], meas[1][i])
                for k, lines in self.fut_lines.items():
                    ft, fv = fut[k]
                    for i in range(4):
                        self._set(lines[i], ft, fv[i])
                for j, ln in enumerate(self.eta_lines):
                    self._set(ln, eta[0], eta[1][j])
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
