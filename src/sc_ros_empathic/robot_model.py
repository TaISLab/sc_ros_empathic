"""
robot_model.py
--------------
Source of the FR3 geometric Jacobian for the manipulability /
singularity-avoidance factor.

Deployment backend: "topic" (default). The taislab_controller C++
Cartesian-velocity controller already computes the FR3 Jacobian every
cycle in its update(); it publishes it (std_msgs/Float64MultiArray,
the 6xN matrix flattened row-major) and shared_control_node.py feeds
each message straight into update_current_state(). No PyKDL, no URDF
parsing, and -- crucially -- the exact Jacobian the 1 kHz low-level
loop is using, so there is no risk of a KDL/URDF vs. libfranka model
mismatch.

    Lookahead limitation: the manipulability factor is defined on
    delta_w between the current configuration and q + qdot_candidate*dt.
    With only the stream of CURRENT Jacobians there is no analytic
    model to evaluate J at a hypothetical future q, so predict_jacobian()
    for this backend propagates the published Jacobian forward in TIME
    (finite difference of the last two messages) by dt_lookahead. That
    estimate does not depend on which candidate command is being scored,
    so eta_k4 comes out equal for v_h, v_r and the blend: the factor
    then acts as a "reduce authority while the trajectory is heading
    into a low-manipulability region" term rather than a per-candidate
    ranking. Per-candidate scoring would need J(q) at an arbitrary q
    (an analytic/KDL model) -- use backend="kdl" for that.

Other backends:
  * "kdl"  -- analytic Jacobian from `q` via PyKDL + /robot_description.
             Gives a true per-candidate predicted Jacobian. Requires
             python-orocos-kdl and kdl-parser-py.
  * "cached" -- alias of "topic" (same mechanism), kept for callers /
             offline code that already used the name.
"""

import numpy as np

try:
    import PyKDL  # noqa: F401
    from kdl_parser_py.urdf import treeFromParam
    _HAS_KDL = True
except ImportError:
    _HAS_KDL = False

_TOPIC_BACKENDS = ("topic", "cached")


class RobotModel(object):
    def __init__(self, backend="topic", base_link="fr3_link0",
                 ee_link="fr3_link8"):
        if backend == "kdl" and not _HAS_KDL:
            raise RuntimeError(
                "RobotModel: backend='kdl' requires PyKDL and "
                "kdl_parser_py (ros-<distro>-python-orocos-kdl and "
                "ros-<distro>-kdl-parser-py), and a valid "
                "/robot_description. Refusing to silently fall back to "
                "an approximate Jacobian for the manipulability factor.")
        if backend not in _TOPIC_BACKENDS and backend != "kdl":
            raise RuntimeError("RobotModel: unknown backend %r "
                               "(topic|kdl|cached)" % backend)
        self.backend = backend
        self._last_J = None
        self._last_q = None
        # (t, J) history for the time-propagated predicted Jacobian
        # (topic backend only); at most the two most recent samples.
        self._J_hist = []

        if self.backend == "kdl":
            ok, tree = treeFromParam("/robot_description")
            if not ok:
                raise RuntimeError(
                    "RobotModel: failed to parse /robot_description for "
                    "KDL -- is the FR3 description loaded on the "
                    "parameter server before this node starts?")
            self.chain = tree.getChain(base_link, ee_link)
            self.n_joints = self.chain.getNrOfJoints()
            self._jac_solver = PyKDL.ChainJntToJacSolver(self.chain)

    # -- current-state intake (topic / cached backend) ----------------
    def update_current_state(self, q, J_current, stamp=None):
        """Store the latest Jacobian from the external source (the C++
        controller). `stamp` is a float time in seconds; when given, it
        feeds the finite-difference predicted Jacobian."""
        self._last_q = None if q is None else np.asarray(q, dtype=float)
        self._last_J = np.asarray(J_current, dtype=float)
        if stamp is not None:
            self._J_hist.append((float(stamp), self._last_J))
            if len(self._J_hist) > 2:
                self._J_hist.pop(0)

    def has_jacobian(self):
        return self._last_J is not None

    def jacobian(self, q):
        """Return the 6xN geometric Jacobian at configuration q (KDL),
        or the last one received (topic/cached backend, q ignored)."""
        if self.backend == "kdl":
            q = np.asarray(q, dtype=float)
            jnt = PyKDL.JntArray(self.n_joints)
            for i in range(self.n_joints):
                jnt[i] = q[i]
            jac = PyKDL.Jacobian(self.n_joints)
            self._jac_solver.JntToJac(jnt, jac)
            return np.array([[jac[r, c] for c in range(self.n_joints)]
                              for r in range(6)])

        if self._last_J is None:
            raise RuntimeError(
                "RobotModel: backend=%r but no Jacobian has been "
                "received yet (update_current_state() never called)."
                % self.backend)
        return self._last_J

    def predict_jacobian(self, q_current, qdot_candidate, dt):
        """Jacobian at the lookahead horizon.

        KDL: analytic J at q_current + qdot_candidate*dt (per-candidate).
        topic/cached: the published J propagated forward in TIME by dt
        via finite difference of the last two messages -- NOT
        per-candidate (see module docstring). Falls back to the current
        J until two timestamped samples are available.
        """
        if self.backend == "kdl":
            return self.jacobian(np.asarray(q_current, dtype=float)
                                 + np.asarray(qdot_candidate, dtype=float) * dt)

        if self._last_J is None:
            raise RuntimeError(
                "RobotModel: predict_jacobian before any Jacobian was "
                "received.")
        if len(self._J_hist) < 2:
            return self._last_J
        (t0, J0), (t1, J1) = self._J_hist[-2], self._J_hist[-1]
        span = t1 - t0
        if span <= 1e-6:
            return self._last_J
        return J1 + (J1 - J0) * (float(dt) / span)
