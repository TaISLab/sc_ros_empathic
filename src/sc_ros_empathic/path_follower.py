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

Also exposes the robot command
    v_r = cruise_speed * tangent + Ka * (x_d - x)
and the path tangent vector needed by the directness performance
factor. cruise_speed defaults to 0 (the verbatim proportional-only law
from the offline package); a small positive value keeps the robot
advancing along the path on a tracing task, where the proportional
term alone can stall at near-zero cross-track error. The sampled-point
cache in CirclePath is a pure per-cycle-cost optimisation, no change
to the geometry.
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

        self._grid_n = 0
        self._s_grid = None
        self._grid_pts = None

    def point(self, s):
        theta = 2 * np.pi * s
        return self.center + self.radius * (np.cos(theta) * self.u + np.sin(theta) * self.v)

    def tangent(self, s):
        theta = 2 * np.pi * s
        t = -np.sin(theta) * self.u + np.cos(theta) * self.v
        return t / np.linalg.norm(t)

    def _ensure_grid(self, n):
        """Cache the sampled circle points once; nearest_s is called
        (twice) every control cycle, so re-sampling n points each time
        was a needless per-cycle cost."""
        if self._grid_n != n:
            self._s_grid = np.linspace(0.0, 1.0, n, endpoint=False)
            th = 2 * np.pi * self._s_grid
            self._grid_pts = (self.center
                              + self.radius * (np.cos(th)[:, None] * self.u
                                               + np.sin(th)[:, None] * self.v))
            self._grid_n = n

    def sampled(self, n_samples=360):
        """(s_grid, points) for the cached n-sample discretisation."""
        self._ensure_grid(n_samples)
        return self._s_grid, self._grid_pts

    def nearest_s(self, x, n_samples=360):
        self._ensure_grid(n_samples)
        d = np.linalg.norm(self._grid_pts - np.asarray(x, dtype=float), axis=1)
        return float(self._s_grid[int(np.argmin(d))])


class ReactivePathFollower:
    def __init__(self, path, lam=1.02, rho_min=0.015, Ka=2.0, n_samples=360,
                 cruise_speed=0.0, mode="lookahead"):
        self.path = path
        self.lam = lam
        self.rho_min = rho_min
        self.Ka = Ka
        self.n_samples = n_samples
        # Tangential feed-forward along the path (m/s). 0 -> the
        # verbatim proportional-only law; > 0 guarantees forward
        # progress on a tracing task even when the cross-track error
        # (and hence the proportional term) is ~0.
        self.cruise_speed = float(cruise_speed)
        # "lookahead": the virtual-sphere pure-pursuit law of [1],
        #   v_r = cruise*tangent(x_d) + Ka*(x_d - x). Steers along the
        #   chord to the lookahead point, so it CUTS CORNERS -- under
        #   loop lag it settles on a circle smaller than the reference,
        #   more so for a larger rho_min.
        # "crosstrack": v_r = cruise*tangent(s_near) + Ka*(P(s_near) - x)
        #   -- tangential feed-forward plus a pull to the NEAREST
        #   reference point. No corner-cutting: it tracks the reference
        #   radius. Not the [1] law; use it when radius fidelity matters.
        if mode not in ("lookahead", "crosstrack"):
            raise ValueError("mode must be 'lookahead' or 'crosstrack'")
        self.mode = mode

    def _dist_to_path(self, x, s):
        return float(np.linalg.norm(self.path.point(s) - x))

    def next_goal(self, x):
        # Same algorithm as the offline package -- nearest sample, then
        # the FIRST sample AHEAD (wrapping) whose distance to x reaches
        # the lookahead radius rho, i.e. the first forward crossing of
        # the virtual lookahead sphere (pure pursuit, Coulter
        # CMU-RI-TR-92-01). Taking the first FORWARD crossing (not the
        # global argmin of |dist - rho|) avoids locking onto the second,
        # near-zero-progress solution that a closed path's non-monotonic
        # distance profile also produces. Vectorised over the cached
        # sample grid so it costs ~10 us instead of a ~1.5 ms Python
        # loop -- no change to the geometry.
        x = np.asarray(x, dtype=float)
        s_grid, pts = self.path.sampled(self.n_samples)
        d_all = np.linalg.norm(pts - x, axis=1)
        i_near = int(np.argmin(d_all))
        d = float(d_all[i_near])
        rho = self.lam * d if d >= self.rho_min else self.rho_min

        n = len(s_grid)
        fwd = (i_near + 1 + np.arange(n)) % n          # indices ahead, wrapping
        d_fwd = d_all[fwd]
        crossings = np.flatnonzero(d_fwd >= rho)
        best_i = int(fwd[crossings[0]]) if crossings.size else int(fwd[-1])
        best_s = float(s_grid[best_i])

        x_d = self.path.point(best_s)
        tangent = self.path.tangent(best_s)
        return x_d, tangent, best_s

    def robot_command(self, x):
        """Returns (v_r, tangent). See __init__ for the two modes."""
        x = np.asarray(x, dtype=float)
        if self.mode == "crosstrack":
            s_near = self.path.nearest_s(x, self.n_samples)
            p_near = self.path.point(s_near)
            tangent = self.path.tangent(s_near)
            v_r = self.cruise_speed * tangent + self.Ka * (p_near - x)
            return v_r, tangent
        x_d, tangent, _ = self.next_goal(x)
        v_r = self.cruise_speed * tangent + self.Ka * (x_d - x)
        return v_r, tangent

    def progress(self, x):
        """Observable-only: the participant's current progress along the
        path, for lap counting and cross-track-error logging (NOT used
        by the control law). Returns (s_near, cross_track_dist) where
        s_near = argmin_s dist(x, P(s)) in [0, 1) and cross_track_dist
        is that minimum distance in metres.
        """
        s_near = self.path.nearest_s(x, self.n_samples)
        cross_track = self._dist_to_path(x, s_near)
        return s_near, cross_track
