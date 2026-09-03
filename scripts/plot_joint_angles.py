#!/usr/bin/env python3
"""
plot_joint_angles.py
--------------------
Live plot of the human arm's joint angles, one SUBPLOT per joint plus a
final subplot for the joint-limit-safety efficiency (shared time axis).

  q1  shoulder flex/ext      q3  shoulder int/ext rotation
  q2  shoulder abd/add       q4  elbow flex/ext (0 = extended)

Per joint subplot, in that joint's colour:
  solid    measured q_i (~diag/joint_deg)
  dashed   q_i extrapolated along the velocity the HUMAN command v_h
           induces          (~diag/joint_deg_future_h)
  dotted   ... the ROBOT command v_r induces
                            (~diag/joint_deg_future_r)
  dash-dot ... the eta-weighted BLEND v_hat_s induces, before eta_s
           scales the output (~diag/joint_deg_future_s)
Each future trace is q_h + qdot_k * dt_lookahead: qdot_k is what
joint_safety's dynamic term scores for candidate k, over the
controller's own lookahead horizon. The joint's [min, max] range is a
shaded band in the same subplot.

Efficiency subplot: ONLY the joint-limit-safety factor (factors_*[2],
"joint_safety", in (0, 1]) for each candidate -- js_h / js_r / js_s
from ~diag/factors_{h,r,s} -- not the full weighted eta_h/eta_r/eta_s.

Params:
  ~window_s           rolling time window shown (s), default 30
  ~redraw_hz          figure redraw rate (Hz),      default 15
  ~show_future        draw the 3 extrapolation traces, default true
  ~show_joint_safety  draw the joint_safety subplot,   default true
"""

import collections
import threading

import rospy
from std_msgs.msg import Float64MultiArray

import matplotlib.pyplot as plt

# q1..q4 colours -- one per joint subplot.
JOINT_COLOURS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd')
JOINT_LABELS = ('q1 shoulder flex/ext', 'q2 shoulder abd/add',
                'q3 shoulder int/ext rot', 'q4 elbow (0=extended)')
# joint_safety factor per candidate -> colour
JS_COLOURS = {'h': '#1b9e77', 'r': '#7570b3', 's': '#d95f02'}
# candidate command -> line style (measured q_h is 'solid')
STYLE = {'meas': '-', 'h': '--', 'r': ':', 's': '-.'}
STYLE_LABEL = {'meas': 'measured', 'h': 'v_h projection',
               'r': 'v_r projection', 's': 'v_sum projection'}
FACTOR_JOINT_SAFETY_IDX = 2   # [smoothness, directness, joint_safety, manip]


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
        self.show_js = bool(rospy.get_param('~show_joint_safety', True))

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
        fac_topics = {
            'h': rospy.get_param('~factors_h_topic', base + '/diag/factors_h'),
            'r': rospy.get_param('~factors_r_topic', base + '/diag/factors_r'),
            's': rospy.get_param('~factors_s_topic', base + '/diag/factors_s'),
        }
        lim_topic = rospy.get_param('~joint_deg_limits_topic',
                                    base + '/diag/joint_deg_limits')

        # ROS delivers each callback on its own thread; the draw loop
        # runs on the main thread. All deque access is under this lock
        # (a mid-snapshot append otherwise leaves x/y one sample apart
        # and matplotlib raises a broadcast error).
        self._lock = threading.Lock()
        self._t0 = None
        self.limits = None                 # [(min,max)] x4, degrees
        self.meas = _MT(4)
        self.fut = {k: _MT(4) for k in ('h', 'r', 's')}
        self.js = {k: _MT(1) for k in ('h', 'r', 's')}

        rospy.Subscriber(deg_topic, Float64MultiArray, self._deg_cb,
                         queue_size=5)
        rospy.Subscriber(lim_topic, Float64MultiArray, self._lim_cb,
                         queue_size=1)
        if self.show_future:
            for k, topic in fut_topics.items():
                rospy.Subscriber(topic, Float64MultiArray,
                                 lambda m, kk=k: self._fut_cb(kk, m),
                                 queue_size=5)
        if self.show_js:
            for k, topic in fac_topics.items():
                rospy.Subscriber(topic, Float64MultiArray,
                                 lambda m, kk=k: self._js_cb(kk, m),
                                 queue_size=5)

        nrows = 5 if self.show_js else 4
        self.fig, axs = plt.subplots(
            nrows, 1, sharex=True, figsize=(9, 2.0 * nrows + 1),
            constrained_layout=True)
        self.jaxs = list(axs[:4])
        self.jsax = axs[4] if self.show_js else None
        self.fig.suptitle('Human joint angles vs limits')

        self.meas_lines = []
        self.fut_lines = {k: [] for k in ('h', 'r', 's')}
        for i, ax in enumerate(self.jaxs):
            c = JOINT_COLOURS[i]
            ax.set_ylabel('%s\n(deg)' % JOINT_LABELS[i], color=c, fontsize=8)
            ax.tick_params(axis='y', labelcolor=c)
            ax.grid(True, alpha=0.3)
            self.meas_lines.append(
                ax.plot([], [], color=c, lw=1.7, ls=STYLE['meas'])[0])
            if self.show_future:
                for k in ('h', 'r', 's'):
                    self.fut_lines[k].append(
                        ax.plot([], [], color=c, lw=1.2, ls=STYLE[k],
                                alpha=0.9)[0])
        self.jaxs[-1].set_xlabel('t (s)')

        self.js_lines = {}
        if self.show_js:
            self.jsax.set_ylabel('joint_safety\n(0..1)', fontsize=8)
            self.jsax.set_ylim(-0.02, 1.05)
            self.jsax.grid(True, alpha=0.3)
            self.jsax.set_xlabel('t (s)')
            self.jaxs[-1].set_xlabel('')
            for k in ('h', 'r', 's'):
                (ln,) = self.jsax.plot([], [], color=JS_COLOURS[k], lw=1.8,
                                       ls=STYLE[k], label='js_' + k)
                self.js_lines[k] = ln
            self.jsax.legend(loc='lower left', fontsize=8, ncol=3)

        # style key -- once, on the top joint subplot
        keys = ['meas'] + (['h', 'r', 's'] if self.show_future else [])
        self.jaxs[0].legend(
            handles=[plt.Line2D([], [], color='#666', lw=1.5, ls=STYLE[k],
                                label=STYLE_LABEL[k]) for k in keys],
            loc='upper left', fontsize=7, ncol=len(keys))

        self._band_artists = []

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

    def _js_cb(self, key, msg):
        if len(msg.data) <= FACTOR_JOINT_SAFETY_IDX:
            return
        with self._lock:
            self._push(self.js[key], [float(msg.data[FACTOR_JOINT_SAFETY_IDX])])

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
            ax, c = self.jaxs[i], JOINT_COLOURS[i]
            self._band_artists.append(
                ax.axhspan(lo, hi, color=c, alpha=0.10, zorder=0))
            for y in (lo, hi):
                self._band_artists.append(
                    ax.axhline(y, color=c, ls='--', lw=1.0, alpha=0.6,
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
            js = {k: cp(self.js[k]) for k in self.js} if self.show_js else {}
            lims = list(self.limits) if self.limits else None
        return meas, fut, js, lims

    def spin(self):
        plt.ion()
        plt.show()
        rate = rospy.Rate(self.redraw_hz)
        while not rospy.is_shutdown():
            meas, fut, js, lims = self._snapshot()
            all_t = ([meas[0]] + [fut[k][0] for k in fut]
                     + [js[k][0] for k in js])
            last = max((s[-1] for s in all_t if s), default=None)
            if last is not None:
                for i in range(4):
                    self._set(self.meas_lines[i], meas[0], meas[1][i])
                    for k in self.fut_lines:
                        ft, fv = fut[k]
                        self._set(self.fut_lines[k][i], ft, fv[i])
                for k, ln in self.js_lines.items():
                    jt, jv = js[k]
                    self._set(ln, jt, jv[0])
                self._draw_bands(lims)
                self.jaxs[0].set_xlim(max(0.0, last - self.window_s),
                                      max(self.window_s, last))
                for ax in self.jaxs:
                    ax.relim()
                    ax.autoscale_view(scalex=False, scaley=True)
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
