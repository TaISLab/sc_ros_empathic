"""
robot_model.py
--------------
Thin wrapper around the FR3 kinematic model needed ONLY for the
manipulability / singularity-avoidance factor: it must be able to
return the geometric Jacobian at an arbitrary joint configuration q
(not just the current one, since we need to predict the Jacobian at
q + qdot_candidate * dt for each candidate command).

CORRECTION vs. the offline simulation package this was ported from:
that version's default "franka_state" backend assumed franka_ros
publishes an `O_Jac_EE` field in franka_msgs/FrankaState. It does not
-- checked against the authoritative message definition
(frankarobotics/franka_ros, franka_msgs/msg/FrankaState.msg): there is
no Jacobian field there at all, only q, dq, tau_J, O_T_EE, O_F_ext_hat_K,
etc. So the real deployment backend here is "kdl" by default: the
Jacobian is computed analytically (closed form, not finite differences)
from `q` via PyKDL + the URDF already published on `/robot_description`
by the franka_description bring-up. This is fast (no numeric
differentiation) and does not depend on any nonexistent topic.

If PyKDL is not installed, this class raises at construction time
rather than silently falling back to a wrong/stale Jacobian -- a
wrong manipulability factor is worse than a node that refuses to
start, since it would silently corrupt candidate scoring instead of
failing loudly.
"""

import numpy as np

try:
    import PyKDL  # noqa: F401
    from kdl_parser_py.urdf import treeFromParam
    _HAS_KDL = True
except ImportError:
    _HAS_KDL = False


class RobotModel(object):
    def __init__(self, backend="kdl", base_link="fr3_link0",
                 ee_link="fr3_link8"):
        if backend == "kdl" and not _HAS_KDL:
            raise RuntimeError(
                "RobotModel: backend='kdl' requires PyKDL and "
                "kdl_parser_py (ros-<distro>-python-orocos-kdl and "
                "ros-<distro>-kdl-parser-py), and a valid "
                "/robot_description. Refusing to silently fall back to "
                "an approximate Jacobian for the manipulability factor.")
        self.backend = backend
        self._last_J = None
        self._last_q = None

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

    # -- current-state bookkeeping (only used by the "cached" backend) --
    def update_current_state(self, q, J_current):
        """Store the latest (q, J) if using a pre-computed/cached
        Jacobian source instead of computing it here on demand."""
        self._last_q = np.asarray(q, dtype=float)
        self._last_J = np.asarray(J_current, dtype=float)

    def jacobian(self, q):
        """Return the 6xN geometric Jacobian at configuration q."""
        q = np.asarray(q, dtype=float)
        if self.backend == "kdl":
            jnt = PyKDL.JntArray(self.n_joints)
            for i in range(self.n_joints):
                jnt[i] = q[i]
            jac = PyKDL.Jacobian(self.n_joints)
            self._jac_solver.JntToJac(jnt, jac)
            return np.array([[jac[r, c] for c in range(self.n_joints)]
                              for r in range(6)])

        # "cached" fallback: whatever was last passed to
        # update_current_state(), for callers that already have their
        # own Jacobian source and just want this class's interface.
        if self._last_J is None:
            raise RuntimeError("RobotModel: backend='cached' but "
                                "update_current_state() was never called.")
        return self._last_J

    def predict_jacobian(self, q_current, qdot_candidate, dt):
        q_pred = q_current + qdot_candidate * dt
        return self.jacobian(q_pred)
