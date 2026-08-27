"""
robot_model.py
--------------
Source of the FR3 geometric Jacobian for the manipulability /
singularity-avoidance factor. It must return J(q) at an ARBITRARY q:
the current configuration and the lookahead q + qdot_candidate*dt for
every candidate command.

Backends
  * "analytic" (default) -- pure-numpy FR3 forward kinematics +
    geometric Jacobian from the nominal Denavit-Hartenberg parameters
    (see fr3_model.py). Computed here, independently of the velocity
    controller and without PyKDL; the only input is q (from
    franka_states). Gives a true per-candidate predicted Jacobian.
  * "kdl" -- analytic Jacobian from q via PyKDL + /robot_description.
    Equivalent to "analytic" but pulls in python-orocos-kdl /
    kdl-parser-py and depends on a loaded URDF.
  * "topic" / "cached" -- the Jacobian is supplied from outside via
    update_current_state() (e.g. a controller that publishes its own
    6xN Jacobian). Only the CURRENT J is available, so predict_jacobian
    propagates it forward in TIME (finite difference of the last two
    messages), which is NOT per-candidate -- see the note there.

`franka_ros` does not publish a Jacobian field (checked against
franka_msgs/FrankaState.msg), so one of the above is always needed.
"""

import numpy as np

from . import fr3_model

try:
    import PyKDL  # noqa: F401
    from kdl_parser_py.urdf import treeFromParam
    _HAS_KDL = True
except ImportError:
    _HAS_KDL = False

_EXTERNAL_BACKENDS = ("topic", "cached")
_VALID_BACKENDS = ("analytic", "kdl") + _EXTERNAL_BACKENDS


class RobotModel(object):
    def __init__(self, backend="analytic", base_link="fr3_link0",
                 ee_link="fr3_link8"):
        if backend not in _VALID_BACKENDS:
            raise RuntimeError("RobotModel: unknown backend %r "
                               "(analytic|kdl|topic|cached)" % backend)
        if backend == "kdl" and not _HAS_KDL:
            raise RuntimeError(
                "RobotModel: backend='kdl' requires PyKDL and "
                "kdl_parser_py (ros-<distro>-python-orocos-kdl and "
                "ros-<distro>-kdl-parser-py), and a valid "
                "/robot_description. Use backend='analytic' for a "
                "self-contained Jacobian.")
        self.backend = backend
        self._last_J = None
        self._last_q = None
        self._J_hist = []          # up to two (t, J) samples (external backend)

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

    # -- external Jacobian intake ("topic" / "cached") ----------------
    def update_current_state(self, q, J_current, stamp=None):
        """Store a Jacobian supplied from outside. `stamp` (float
        seconds), when given, feeds the finite-difference predicted
        Jacobian."""
        self._last_q = None if q is None else np.asarray(q, dtype=float)
        self._last_J = np.asarray(J_current, dtype=float)
        if stamp is not None:
            self._J_hist.append((float(stamp), self._last_J))
            if len(self._J_hist) > 2:
                self._J_hist.pop(0)

    def has_jacobian(self):
        if self.backend in _EXTERNAL_BACKENDS:
            return self._last_J is not None
        return True

    def jacobian(self, q):
        """6xN geometric Jacobian at configuration q."""
        if self.backend == "analytic":
            return fr3_model.jacobian(np.asarray(q, dtype=float))

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

        analytic / kdl: J at q_current + qdot_candidate*dt (per-candidate,
        exact for the nominal kinematics).
        topic / cached: the supplied J propagated forward in TIME by dt
        (finite difference of the last two messages) -- NOT
        per-candidate; falls back to the current J until two timestamped
        samples exist.
        """
        if self.backend in ("analytic", "kdl"):
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
