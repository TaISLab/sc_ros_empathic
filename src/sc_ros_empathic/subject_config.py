"""
subject_config.py
-----------------
Per-volunteer configuration for the human-subject protocol.

ONE file per volunteer (config/subjects/SXX.yaml), holding ONLY
intrinsic human parameters: the volunteer id, the upper-arm / forearm
lengths, and the pre-registered physiological joint ranges (q1..q4).

Everything about the experiment itself -- controller gains, factor
weights, the joint-margin threshold tau, path geometry, loop rate --
is the SAME for every volunteer and lives in
config/shared_control.yaml, NOT here.

Same discipline as the sibling package (sc_effort_experiment): a
missing required value raises immediately with a clear message rather
than letting a node run with a silent default (a run with l1/l2 = 0 or
the wrong ROM would look fine and quietly corrupt the joint-safety
factor).
"""

import os

import numpy as np

try:
    import yaml
    _HAS_YAML = True
except ImportError:                       # pragma: no cover - ROS ships PyYAML
    _HAS_YAML = False

_JOINT_KEYS = ("q1", "q2", "q3", "q4")
_JOINT_DESC = {
    "q1": "shoulder flexion/extension",
    "q2": "shoulder abduction/adduction",
    "q3": "shoulder internal/external rotation",
    "q4": "elbow flexion/extension",
}


class SubjectConfigError(Exception):
    pass


class SubjectConfig(object):
    """Load and validate one config/subjects/SXX.yaml.

    Attributes after construction:
      subject_id        str, non-empty
      l1, l2            float, metres, > 0
      joint_limits      (4, 2) ndarray [q_min, q_max] for q1..q4, or None
      has_joint_limits  bool  (False -> caller uses the study defaults)
      dominant_arm      str, lower-case, may be ''
      date, notes       str, may be ''
      sex               str, lower-case, may be ''      (bookkeeping)
      age_years         float or None                   (bookkeeping)
      height_m          float or None                   (bookkeeping)
      mass_kg           float or None                   (bookkeeping)

    The demographic fields (sex/age/height/mass) are recorded for the
    paper's participant table; nothing in the control law reads them,
    so a blank or 0 value is accepted (treated as "not filled yet").
    """

    def __init__(self, path):
        if not path:
            raise SubjectConfigError(
                "no subject file given (launch with subject:=SXX)")
        path = os.path.expanduser(str(path))
        if not os.path.isfile(path):
            raise SubjectConfigError("subject file not found: %s" % path)
        if not _HAS_YAML:
            raise SubjectConfigError(
                "PyYAML is required to read %s (install python3-yaml)" % path)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        self.path = path
        self._build(data, path)

    @classmethod
    def from_dict(cls, data, path="<dict>"):
        """Build from an already-parsed mapping (used by the tests, and
        by any caller that has the dict already)."""
        obj = cls.__new__(cls)
        obj.path = path
        obj._build(data, path)
        return obj

    # ------------------------------------------------------------------
    def _build(self, data, path):
        name = os.path.basename(path)
        if not isinstance(data, dict):
            raise SubjectConfigError("%s is not a YAML mapping" % name)

        subj = data.get("subject") or {}
        self.subject_id = str(subj.get("id") or "").strip()
        if not self.subject_id:
            raise SubjectConfigError(
                "%s: subject.id is missing" % name)
        self.dominant_arm = str(subj.get("dominant_arm") or "").strip().lower()
        self.date = str(subj.get("date") or "").strip()
        self.notes = str(subj.get("notes") or "").strip()

        # Demographics -- bookkeeping for the participant table, not
        # used by the control law. Blank / 0 -> "not filled yet".
        self.sex = str(subj.get("sex") or "").strip().lower()
        self.age_years = self._opt_float(subj.get("age_years"),
                                         "subject.age_years", name)
        self.height_m = self._opt_float(subj.get("height_m"),
                                        "subject.height_m", name)
        self.mass_kg = self._opt_float(subj.get("mass_kg"),
                                       "subject.mass_kg", name)

        anth = data.get("anthropometry") or {}
        self.l1 = self._pos_float(anth.get("l1_m"), "anthropometry.l1_m", name)
        self.l2 = self._pos_float(anth.get("l2_m"), "anthropometry.l2_m", name)

        jl = data.get("joint_limits_rad")
        if jl in (None, {}):
            self.joint_limits = None
            self.has_joint_limits = False
        else:
            self.joint_limits = self._joint_limits(jl, name)
            self.has_joint_limits = True

    @staticmethod
    def _opt_float(value, key, name):
        """Optional non-negative float: blank / None / 0 -> None; a
        present non-numeric or negative value is an error."""
        if value in (None, "", 0, 0.0):
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise SubjectConfigError(
                "%s: %s must be a number or left blank" % (name, key))
        if f < 0.0:
            raise SubjectConfigError(
                "%s: %s must be >= 0 (got %r)" % (name, key, value))
        return f

    @staticmethod
    def _pos_float(value, key, name):
        try:
            f = float(value)
        except (TypeError, ValueError):
            raise SubjectConfigError(
                "%s: %s is missing or not a number" % (name, key))
        if not f > 0.0:
            raise SubjectConfigError(
                "%s: %s must be > 0 (got %r)" % (name, key, value))
        return f

    @staticmethod
    def _joint_limits(jl, name):
        if not isinstance(jl, dict):
            raise SubjectConfigError(
                "%s: joint_limits_rad must be a mapping with q1..q4" % name)
        rows = []
        for k in _JOINT_KEYS:
            if k not in jl:
                raise SubjectConfigError(
                    "%s: joint_limits_rad.%s (%s) is missing -- give all four "
                    "joints or omit the whole block to use the study defaults"
                    % (name, k, _JOINT_DESC[k]))
            try:
                lo, hi = float(jl[k][0]), float(jl[k][1])
            except (TypeError, ValueError, IndexError, KeyError):
                raise SubjectConfigError(
                    "%s: joint_limits_rad.%s must be [min, max] in radians"
                    % (name, k))
            if not hi > lo:
                raise SubjectConfigError(
                    "%s: joint_limits_rad.%s has max <= min (%.3f, %.3f)"
                    % (name, k, lo, hi))
            rows.append([lo, hi])
        return np.asarray(rows, dtype=float)

    # ------------------------------------------------------------------
    def link_lengths(self):
        return self.l1, self.l2

    def summary(self):
        rom = "custom ROM" if self.has_joint_limits else "study-default ROM"
        bits = ["subject %s" % self.subject_id]
        demo = []
        if self.sex:
            demo.append(self.sex)
        if self.age_years is not None:
            demo.append("%gy" % self.age_years)
        if self.height_m is not None:
            demo.append("%.2fm" % self.height_m)
        if self.mass_kg is not None:
            demo.append("%gkg" % self.mass_kg)
        if demo:
            bits.append("/".join(demo))
        bits.append("l1 %.3f l2 %.3f m" % (self.l1, self.l2))
        bits.append(rom)
        if self.dominant_arm:
            bits.append("dominant %s" % self.dominant_arm)
        return " | ".join(bits)
