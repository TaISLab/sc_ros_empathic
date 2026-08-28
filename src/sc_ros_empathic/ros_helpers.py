"""
ros_helpers.py
--------------
Node-side glue shared by the experiment nodes (shared_control_node.py,
baseline_aan_node.py): the external-force deadzone + LPF + startup tare,
the trial-end / lap bookkeeping, and the per-cycle CSV log.

Unlike the rest of this package, this module DOES import rospy -- it is
ROS glue, not part of the offline-portable control core. Keep the
control law itself in the rospy-free modules.
"""

import csv
import os
import time

import numpy as np
import rospy
from std_msgs.msg import Bool
from std_srvs.srv import Empty, EmptyResponse

from .experiment import LapCounter


class ForceProcessor(object):
    """O_F_ext_hat_K[0:3] -> f_filtered: startup tare (average the
    resting wrench for `tare_s` and subtract it), a dead-zone, and a
    first-order LPF. `ready` is False during the tare window (do not
    command the robot then). Registers a std_srvs/Empty `service_name`
    to re-run the tare mid-session.
    """

    def __init__(self, deadzone_N=2.0, lpf_alpha=0.1, tare_s=1.0,
                 service_name='~tare'):
        self.deadzone_N = float(deadzone_N)
        self.alpha = float(lpf_alpha)
        self.tare_s = float(tare_s)
        self.f_filtered = np.zeros(3)
        self.f_bias = np.zeros(3)
        self._samples = []
        self._t0 = None
        self.ready = self.tare_s <= 0.0
        if service_name:
            rospy.Service(service_name, Empty, self._srv)

    def update(self, f_ext_raw):
        f = np.asarray(f_ext_raw, dtype=float)
        if not self.ready:
            now = rospy.Time.now()
            if self._t0 is None:
                self._t0 = now
                rospy.logwarn('force tare: averaging the wrench for %.1f s '
                              '-- HANDS OFF the robot.', self.tare_s)
            self._samples.append(f)
            if (now - self._t0).to_sec() >= self.tare_s:
                self.f_bias = np.mean(self._samples, axis=0)
                self._samples = []
                self.ready = True
                rospy.loginfo('force bias = [%.2f %.2f %.2f] N (|b|=%.2f). '
                              'If large, set the FR3 EE load so '
                              'O_F_ext_hat_K reads ~0 at rest.',
                              float(self.f_bias[0]), float(self.f_bias[1]),
                              float(self.f_bias[2]),
                              float(np.linalg.norm(self.f_bias)))
            return
        f = f - self.f_bias
        n = np.linalg.norm(f)
        f_dz = (np.zeros(3) if n < self.deadzone_N
                else f * (1.0 - self.deadzone_N / n))
        self.f_filtered = self.alpha * f_dz + (1.0 - self.alpha) * self.f_filtered

    def retare(self):
        self.f_bias = np.zeros(3)
        self._samples = []
        self._t0 = None
        self.ready = False

    def _srv(self, _req):
        rospy.logwarn('force: re-taring on request.')
        self.retare()
        return EmptyResponse()


class TrialManager(object):
    """Lap counting (direction-aware, jitter-robust) + trial end.
    Feed s_near every cycle. When `trial_laps` completed laps are
    reached: latch `done`, publish trial_done (if a publisher is
    given), and -- for trial_end == 'shutdown' -- ask should_stop()
    after ~0.5 s so the caller can rospy.signal_shutdown().
    """

    def __init__(self, direction=1, trial_laps=0, trial_end='shutdown',
                 done_pub=None):
        self.lc = LapCounter(direction=direction)
        self.trial_laps = int(trial_laps)
        self.trial_end = str(trial_end).lower()
        self.done_pub = done_pub
        self.done = False
        self._done_t = None
        if self.done_pub is not None:
            self.done_pub.publish(Bool(data=False))

    def update(self, s_near):
        lap = self.lc.update(s_near)
        if (self.trial_laps > 0 and lap >= self.trial_laps and not self.done):
            self.done = True
            self._done_t = rospy.Time.now()
            if self.done_pub is not None:
                self.done_pub.publish(Bool(data=True))
            rospy.loginfo('trial complete (%d laps). %s', self.trial_laps,
                          'shutting down.' if self.trial_end == 'shutdown'
                          else 'holding zero velocity -- Ctrl-C to end.')
        return lap

    def should_stop(self):
        return (self.done and self.trial_end == 'shutdown'
                and self._done_t is not None
                and (rospy.Time.now() - self._done_t).to_sec() > 0.5)

    @property
    def lap(self):
        return self.lc.lap

    @property
    def total_progress(self):
        return self.lc.total_progress


class CsvLogger(object):
    """One row per control cycle. The first three columns are always
    t (epoch s), t_rel (s from the first row) and wall_time; `header`
    is just the remaining column names and row(t, values) the matching
    values. Flushed ~1x/s, closed on rospy shutdown. Inactive (a no-op)
    when neither `path` nor `out_dir` is given.
    """

    def __init__(self, header, path='', out_dir='', label='trial'):
        self.fh = None
        self.w = None
        self._t0 = None
        self._n = 0
        if not path and out_dir:
            path = os.path.join(
                os.path.expanduser(out_dir),
                '%s_%s.csv' % (label, time.strftime('%Y-%m-%d_%H-%M-%S')))
        if not path:
            return
        path = os.path.expanduser(path)
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        self.fh = open(path, 'w', newline='')
        self.w = csv.writer(self.fh)
        self.w.writerow(['t', 't_rel', 'wall_time'] + list(header))
        rospy.loginfo('CSV log -> %s', path)
        rospy.on_shutdown(self.close)

    @property
    def active(self):
        return self.w is not None

    def row(self, t, values):
        if self.w is None:
            return
        if self._t0 is None:
            self._t0 = t
        wall = (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))
                + ('.%03d' % int((t % 1.0) * 1000)))
        out = ['%.3f' % t, '%.3f' % (t - self._t0), wall]
        out += ['%.6g' % v if isinstance(v, float) else v for v in values]
        self.w.writerow(out)
        self._n += 1
        if self._n % 200 == 0:
            self.fh.flush()

    def close(self):
        if self.fh is not None:
            try:
                self.fh.flush()
                self.fh.close()
            except Exception:
                pass
            self.fh = None
