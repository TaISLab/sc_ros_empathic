"""
path_follower.py
-----------------
Reactive path follower (virtual-sphere lookahead), exactly as described
in the reactive shared-control framework: given the current EE
position x and a predefined 3D path P(s), s in [0, 1], it dynamically
picks the next local goal x_d so that the robot always advances in the
positive direction of the path.

    rho = lambda * d   if d >= rho_min
          rho_min       otherwise

    d       = dist(x, P(s_near)),  s_near = argmin_s dist(x, P(s))
    s_c     = argmin_{s > s_near} | dist(x, P(s)) - rho |
    x_d     = P(s_c)

Also exposes a proportional robot command v_r = Ka * (x_d - x) and the
path tangent vector needed by the directness performance factor.

Ported verbatim from the offline simulation/verification package.
"""

import numpy as np


class CirclePath:
    """Convenience path generator for the circle-tracing rehabilitation
    task used in the pilot experiments (5-15 cm radius, arbitrary
    center/normal). Any other parametric or waypoint-based path can be
    used as long as it exposes point(s) and tangent(s)."""

    def __init__(self, center, radius, normal=(0, 0, 1)):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        n = np.asarray(normal, dtype=float)
        self.normal = n / np.linalg.norm(n)
        # Build an orthonormal basis (u, v) spanning the circle's plane.
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, self.normal)) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        self.u = np.cross(self.normal, tmp)
        self.u /= np.linalg.norm(self.u)
        self.v = np.cross(self.normal, self.u)

    def point(self, s):
        theta = 2 * np.pi * s
        return self.center + self.radius * (np.cos(theta) * self.u + np.sin(theta) * self.v)

    def tangent(self, s):
        theta = 2 * np.pi * s
        t = -np.sin(theta) * self.u + np.cos(theta) * self.v
        return t / np.linalg.norm(t)

    def nearest_s(self, x, n_samples=360):
        s_grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        pts = np.array([self.point(s) for s in s_grid])
        d = np.linalg.norm(pts - x, axis=1)
        return float(s_grid[np.argmin(d)])


class ReactivePathFollower:
    def __init__(self, path, lam=1.02, rho_min=0.015, Ka=2.0, n_samples=360):
        self.path = path
        self.lam = lam
        self.rho_min = rho_min
        self.Ka = Ka
        self.n_samples = n_samples

    def _dist_to_path(self, x, s):
        return float(np.linalg.norm(self.path.point(s) - x))

    def next_goal(self, x):
        s_near = self.path.nearest_s(x, self.n_samples)
        d = self._dist_to_path(x, s_near)
        rho = self.lam * d if d >= self.rho_min else self.rho_min

        # Search s > s_near (wrapping around [0,1]), scanning forward in
        # increasing arc-length order, and take the FIRST point whose
        # distance to x reaches (or exceeds) the lookahead radius rho --
        # i.e. the first crossing of the virtual lookahead sphere, exactly
        # as in pure-pursuit path following (Coulter, CMU-RI-TR-92-01).
        #
        # NOTE (bug fixed here): a closed path's distance-from-x profile,
        # as a function of forward arc-length from s_near, is not
        # monotonic -- it rises to a maximum near the antipodal point and
        # falls back towards ~0 as s completes the lap and returns to
        # s_near. For a small rho, there are therefore generically TWO
        # points s where dist(x, P(s)) == rho: one a small step ahead of
        # s_near (the intended lookahead target), and one just before
        # completing a full lap (dist falling back through rho on the
        # *return* leg, representing essentially zero net forward
        # progress). Picking the GLOBAL argmin of |dist - rho| over the
        # full lap could lock onto the second (degenerate, near-zero-
        # progress) solution. Taking the first forward crossing removes
        # this ambiguity by construction.
        s_candidates = np.mod(s_near + np.linspace(1e-4, 1.0, self.n_samples), 1.0)
        best_s = s_candidates[-1]  # fallback: nearly a full lap ahead
        prev_d = d
        for s in s_candidates:
            cur_d = self._dist_to_path(x, s)
            if cur_d >= rho:
                best_s = s
                break
            prev_d = cur_d

        x_d = self.path.point(best_s)
        tangent = self.path.tangent(best_s)
        return x_d, tangent, best_s

    def robot_command(self, x):
        """Returns (v_r, tangent) for the proportional path-following
        control law v_r = Ka * (x_d - x)."""
        x_d, tangent, _ = self.next_goal(x)
        v_r = self.Ka * (x_d - x)
        return v_r, tangent
