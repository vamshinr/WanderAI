"""Bounded-memory egocentric occupancy mapping + frontier extraction.

The agent builds its OWN map from its own sensors as it moves — this module never
touches the environment's privileged occupancy grid or geodesic field. Three parts:

1. **Depth strip** — a 1-D range scan across the agent's field of view. In 2D it is
   simulated by ray-casting (`depth_strip`), exactly the information one row of a
   depth camera provides; in 3D it is decoded from the MuJoCo depth buffer
   (`depth_strip_from_image`), so the same mapper runs on real rendered pixels.
2. **`EgoMap`** — a sparse log-odds occupancy grid (Moravec & Elfes 1985; Thrun 2005
   inverse sensor model): cells along each ray accumulate free evidence, the hit
   cell accumulates occupied evidence. Memory is HARD-BOUNDED: when the number of
   stored cells exceeds `budget_cells`, the map coarsens (cell size doubles,
   occupied evidence dominates on merge). Long-horizon episodes degrade gracefully
   in map *resolution* instead of growing without bound.
3. **Frontiers** (Yamauchi 1997) — clusters of known-free cells that border unknown
   space: the classic "where to explore next" targets. `plan_path` runs A* over the
   agent's own map (unknown cells are optimistically traversable at a small
   penalty), so navigation to a frontier uses only what the agent has itself seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math

from .observation import (DEFAULT_FOV, DEFAULT_CLEARANCE_RANGE, cast_ray)
from .geometry import Pose
from .scene import Scene

DEFAULT_N_RAYS = 15
# Log-odds increments per observation (Thrun, Burgard & Fox 2005, ch. 9).
L_FREE = -0.7          # evidence a traversed cell is free
L_OCC = 1.8            # evidence the hit cell is occupied (stronger: hits are rare)
L_MIN, L_MAX = -4.0, 4.0
OCC_THRESHOLD = 0.6    # log-odds above which a cell counts as occupied
FREE_THRESHOLD = -0.3  # log-odds below which a cell counts as known-free


def depth_strip(scene: Scene, pose: Pose, n_rays: int = DEFAULT_N_RAYS,
                fov: float = DEFAULT_FOV,
                max_range: float = DEFAULT_CLEARANCE_RANGE):
    """Simulated 1-D range scan: `n_rays` bearings evenly spanning the FOV
    (left→right), each returning the range to the nearest surface (capped at
    `max_range`). This is the 2D stand-in for one horizon row of a depth camera —
    the same ray-caster the 3-ray symbolic observation already uses, densified."""
    out = []
    for k in range(n_rays):
        rel = fov / 2 - k * (fov / (n_rays - 1)) if n_rays > 1 else 0.0
        rng = cast_ray(scene, pose.x, pose.y, pose.heading + rel, max_range)
        out.append((rel, rng))
    return out


def depth_strip_from_image(depth, fov_x: float, max_depth: float,
                           n_rays: int = DEFAULT_N_RAYS,
                           band: tuple[float, float] = (0.35, 0.55)):
    """Decode the same 1-D range scan from a rendered depth image (HxW metres).

    Columns map to bearings through the pinhole model (bearing = -atan(ndc·tan(fov/2)));
    depth is sampled over a horizontal band near the horizon and divided by
    cos(bearing) to convert perpendicular depth to range along the ray. Returns
    [(relative_bearing, range)], left→right, capped at `max_depth`."""
    import numpy as np
    h, w = depth.shape
    r0, r1 = max(0, int(h * band[0])), min(h, int(h * band[1]))
    half_tan = math.tan(fov_x / 2.0)
    out = []
    for k in range(n_rays):
        # Even spacing in *bearing* (not in column) to match depth_strip().
        rel = fov_x / 2 - k * (fov_x / (n_rays - 1)) if n_rays > 1 else 0.0
        ndc = -math.tan(rel) / half_tan          # invert bearing = -atan(ndc·tan)
        col = int(round((ndc + 1.0) * (w - 1) / 2.0))
        col = min(max(col, 0), w - 1)
        column = depth[r0:r1, col]
        column = column[np.isfinite(column)]
        if column.size == 0:
            rng = max_depth
        else:
            # 10th percentile: nearest robust surface in the band.
            z = float(np.percentile(column, 10))
            rng = min(max_depth, z / max(math.cos(rel), 1e-6))
        out.append((rel, rng))
    return out


def obstacle_strip_from_depth(depth, *, fov_x: float, fov_y: float,
                              eye_height: float, pitch_rad: float,
                              max_range: float, n_rays: int = DEFAULT_N_RAYS,
                              z_min: float = 0.10, z_max: float = 1.5):
    """Height-aware range scan from a full depth image (the projection SemExp /
    Active Neural SLAM use): backproject every pixel to (bearing, horizontal
    range, height-above-floor), keep only points inside the agent's traversal
    band [z_min, z_max] — so floors and ceilings don't read as obstacles but
    LOW furniture below the horizon does — and take the nearest range per
    bearing bin. A single horizon row misses anything below eye level; this
    doesn't. Returns [(relative_bearing, range)], left→right."""
    import numpy as np
    h, w = depth.shape
    cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
    u_tan = (2.0 * (np.arange(w) + 0.5) / w - 1.0) * math.tan(fov_x / 2.0)
    v_tan = (2.0 * (np.arange(h) + 0.5) / h - 1.0) * math.tan(fov_y / 2.0)
    z = depth
    u = u_tan[None, :] * z                    # lateral (+right), camera frame
    v = v_tan[:, None] * z                    # down, camera frame
    # Camera pitched DOWN by pitch_rad: world height and horizontal-forward
    # components of each point (validated against rendered floors in tests).
    height = eye_height - v * cos_p - z * sin_p
    fwd = z * cos_p + v * sin_p               # horizontal distance ahead (+)
    rng = np.hypot(fwd, u)
    bearing = -np.arctan2(u, np.maximum(fwd, 1e-9))   # + = left
    valid = (np.isfinite(z) & (z > 1e-3) & (height > z_min) & (height < z_max)
             & (rng < max_range))
    half = fov_x / 2.0
    edges = np.linspace(half, -half, n_rays + 1)      # left → right bins
    out = []
    for k in range(n_rays):
        lo, hi = edges[k + 1], edges[k]               # lo < hi
        m = valid & (bearing >= lo) & (bearing < hi)
        if m.any():
            # 5th percentile: nearest robust obstacle in the bin.
            r = float(np.percentile(rng[m], 5))
        else:
            r = max_range
        out.append(((lo + hi) / 2.0, min(r, max_range)))
    return out


@dataclass
class FrontierCluster:
    centroid: tuple[float, float]   # world coordinates
    size: int                       # number of frontier cells


@dataclass
class EgoMap:
    """Sparse log-odds occupancy map with a hard memory budget.

    Cells are stored in a dict keyed by integer (i, j) = floor(world / cell_size);
    the map has no prior knowledge of room bounds. When the stored-cell count
    exceeds `budget_cells` the map re-bins itself at double the cell size —
    occupied evidence wins on merge (pessimistic, so planning never tunnels
    through a wall the finer map had seen)."""

    cell_size: float = 0.25
    budget_cells: int = 0            # 0 = unbounded
    cells: dict = field(default_factory=dict)   # (i, j) -> log-odds
    coarsen_count: int = 0
    updates: int = 0

    # --- indexing ---
    def key(self, x: float, y: float) -> tuple[int, int]:
        return (int(math.floor(x / self.cell_size)),
                int(math.floor(y / self.cell_size)))

    def center(self, ij: tuple[int, int]) -> tuple[float, float]:
        return ((ij[0] + 0.5) * self.cell_size, (ij[1] + 0.5) * self.cell_size)

    # --- classification ---
    def log_odds(self, ij) -> float | None:
        return self.cells.get(ij)

    def is_occupied(self, ij) -> bool:
        lo = self.cells.get(ij)
        return lo is not None and lo > OCC_THRESHOLD

    def is_free(self, ij) -> bool:
        lo = self.cells.get(ij)
        return lo is not None and lo < FREE_THRESHOLD

    def is_unknown(self, ij) -> bool:
        return ij not in self.cells

    # --- update ---
    def _bump(self, ij, delta: float):
        lo = self.cells.get(ij, 0.0) + delta
        self.cells[ij] = min(L_MAX, max(L_MIN, lo))

    def update(self, pose: Pose, strip, max_range: float = DEFAULT_CLEARANCE_RANGE):
        """Integrate one range scan taken at `pose` (inverse sensor model)."""
        self.updates += 1
        for rel, rng in strip:
            angle = pose.heading + rel
            dx, dy = math.cos(angle), math.sin(angle)
            hit = rng < max_range - 1e-6
            # March along the ray at half-cell resolution marking free space.
            step = self.cell_size * 0.5
            n = max(1, int(rng / step))
            for s in range(n):
                t = (s + 0.5) * step
                if t >= rng - step * 0.5:
                    break
                self._bump(self.key(pose.x + t * dx, pose.y + t * dy), L_FREE)
            # Pad the hit slightly INTO the surface so the occupied mark lands in
            # the wall's body, not on the boundary cell the agent may share.
            # Only the agent's own cell is exempt (it is free by definition) —
            # dropping close hits entirely leaves near walls unmapped, and the
            # planner then draws line-of-sight straight through them.
            if hit:
                t = rng + 0.3 * self.cell_size
                hit_key = self.key(pose.x + t * dx, pose.y + t * dy)
                if hit_key != self.key(pose.x, pose.y):
                    self._bump(hit_key, L_OCC)
        # The agent's own cell is free by definition (robot footprint clearing) —
        # assert it strongly so stray near-wall hits can't accumulate over it.
        self._bump(self.key(pose.x, pose.y), 3 * L_FREE)
        if self.budget_cells and len(self.cells) > self.budget_cells:
            self._coarsen()

    def _coarsen(self):
        """Double the cell size, re-binning 2x2 blocks. Merge keeps the most
        occupied evidence (max log-odds) so walls survive coarsening; ties of
        free evidence keep the strongest free value."""
        new: dict = {}
        for (i, j), lo in self.cells.items():
            nij = (i >> 1, j >> 1)
            cur = new.get(nij)
            if cur is None:
                new[nij] = lo
            else:
                # Occupied dominates; otherwise keep the more informative value.
                new[nij] = max(cur, lo) if max(cur, lo) > OCC_THRESHOLD \
                    else (cur if abs(cur) >= abs(lo) else lo)
        self.cells = new
        self.cell_size *= 2.0
        self.coarsen_count += 1

    # --- memory accounting ---
    def memory_cells(self) -> int:
        return len(self.cells)

    # --- frontiers ---
    def frontiers(self, min_cluster: int = 2) -> list[FrontierCluster]:
        """Known-free cells adjacent (4-connectivity) to unknown space, clustered
        by 8-connectivity. Sorted largest-first."""
        frontier = set()
        for ij, lo in self.cells.items():
            if lo >= FREE_THRESHOLD:
                continue
            i, j = ij
            for nb in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if nb not in self.cells:
                    frontier.add(ij)
                    break
        clusters: list[FrontierCluster] = []
        seen: set = set()
        for start in frontier:
            if start in seen:
                continue
            stack, comp = [start], []
            seen.add(start)
            while stack:
                i, j = stack.pop()
                comp.append((i, j))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        nb = (i + di, j + dj)
                        if nb in frontier and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
            if len(comp) >= min_cluster:
                cx = sum(c[0] for c in comp) / len(comp)
                cy = sum(c[1] for c in comp) / len(comp)
                clusters.append(FrontierCluster(
                    centroid=((cx + 0.5) * self.cell_size, (cy + 0.5) * self.cell_size),
                    size=len(comp)))
        clusters.sort(key=lambda c: -c.size)
        return clusters

    def _nearest_open(self, ij, radius: int = 3):
        """Closest non-occupied cell within `radius` (ring search), or None."""
        for r in range(1, radius + 1):
            best, best_d = None, math.inf
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    nb = (ij[0] + di, ij[1] + dj)
                    if not self.is_occupied(nb):
                        d = di * di + dj * dj
                        if d < best_d:
                            best, best_d = nb, d
            if best is not None:
                return best
        return None

    def line_of_sight(self, a_xy, b_xy) -> bool:
        """No occupied cell on the segment a→b (sampled at half-cell steps) —
        used for string-pulling smoothed path following."""
        ax, ay = a_xy
        bx, by = b_xy
        d = math.hypot(bx - ax, by - ay)
        n = max(1, int(d / (self.cell_size * 0.5)))
        for s in range(n + 1):
            t = s / n
            if self.is_occupied(self.key(ax + t * (bx - ax), ay + t * (by - ay))):
                return False
        return True

    # --- planning on the agent's own map ---
    def plan_path(self, start_xy, goal_xy, unknown_cost: float = 1.5,
                  max_expansions: int = 20000):
        """A* from start to goal over this map. Occupied cells are blocked;
        unknown cells are traversable at `unknown_cost`x (optimistic-with-penalty,
        the standard exploration-planning assumption). Returns a list of world
        waypoints (excluding start), or None."""
        start = self.key(*start_xy)
        goal = self.key(*goal_xy)
        if self.is_occupied(goal):
            # A frontier centroid can round into the wall it hugs — snap to the
            # nearest non-occupied cell instead of refusing to plan.
            goal = self._nearest_open(goal, radius=3)
            if goal is None:
                return None
        # The agent physically occupies the start — its immediate ring is
        # traversable no matter what stray hits say, or a near-wall robot can
        # never plan its way out of its own (mis)map.
        clear_ring = {(start[0] + di, start[1] + dj)
                      for di in (-1, 0, 1) for dj in (-1, 0, 1)}

        def h(ij):
            return math.hypot(ij[0] - goal[0], ij[1] - goal[1])

        g = {start: 0.0}
        came: dict = {}
        pq = [(h(start), start)]
        expansions = 0
        closed = set()
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == goal:
                path = [cur]
                while path[-1] in came:
                    path.append(came[path[-1]])
                path.reverse()
                return [self.center(ij) for ij in path[1:]] or [self.center(goal)]
            expansions += 1
            if expansions > max_expansions:
                return None
            ci, cj = cur
            for di, dj, w in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                              (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))):
                nb = (ci + di, cj + dj)
                if self.is_occupied(nb) and nb not in clear_ring:
                    continue
                cost = w * (unknown_cost if self.is_unknown(nb) else 1.0)
                ng = g[cur] + cost
                if ng < g.get(nb, math.inf):
                    g[nb] = ng
                    came[nb] = cur
                    heapq.heappush(pq, (ng + h(nb), nb))
        return None


def frontier_hint(egomap: EgoMap, pose: Pose,
                  fov: float = DEFAULT_FOV) -> dict | None:
    """Compact, in-context summary of the map for a learned/LLM policy: the
    direction (left/center/right/behind) and distance of the best frontier
    (largest cluster, distance-discounted). None if the map has no frontiers.

    This is the bridge between the mapping module and the text observation — a
    bounded piece of episodic map memory a language policy can actually consume."""
    clusters = egomap.frontiers()
    if not clusters:
        return None
    best, best_score = None, -math.inf
    for c in clusters:
        d = math.hypot(c.centroid[0] - pose.x, c.centroid[1] - pose.y)
        score = c.size / (1.0 + d)
        if score > best_score:
            best, best_score = c, score
    dx, dy = best.centroid[0] - pose.x, best.centroid[1] - pose.y
    rel = math.atan2(dy, dx) - pose.heading
    rel = math.atan2(math.sin(rel), math.cos(rel))
    if abs(rel) <= fov / 6:
        direction = "center"
    elif abs(rel) <= math.pi / 2:
        direction = "left" if rel > 0 else "right"
    else:
        direction = "behind"
    return {"direction": direction, "distance": math.hypot(dx, dy),
            "bearing": rel, "size": best.size}
