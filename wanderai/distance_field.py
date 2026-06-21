from __future__ import annotations
from dataclasses import dataclass
import heapq
import math
import numpy as np
from .occupancy import OccupancyGrid

_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


@dataclass
class DistanceField:
    """Shortest walkable (geodesic) distance from every free cell to the ball,
    computed by a wavefront (Dijkstra) outward from the ball cell. Unreachable
    cells hold np.inf. Continuous positions are queried by bilinear
    interpolation over finite neighbors."""

    dist: np.ndarray            # float meters, np.inf if unreachable
    grid: OccupancyGrid

    @classmethod
    def from_grid(cls, grid: OccupancyGrid, ball_xy: tuple[float, float]) -> "DistanceField":
        dist = np.full(grid.blocked.shape, np.inf)
        sr, sc = grid.world_to_cell(*ball_xy)
        if not grid.in_bounds(sr, sc) or grid.blocked[sr, sc]:
            raise ValueError("distance field goal must be in bounds and unblocked")
        cs = grid.cell_size
        dist[sr, sc] = 0.0
        pq = [(0.0, sr, sc)]
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r, c]:
                continue
            for dr, dc, w in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not _can_traverse(grid, r, c, dr, dc):
                    continue
                nd = d + w * cs
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
        return cls(dist, grid)

    def query(self, x: float, y: float) -> float:
        """Bilinear interpolation of the distance field at a continuous point,
        averaging only over neighbors with finite (reachable) distance."""
        g = self.grid
        if g.is_blocked_world(x, y):
            return math.inf
        fx = (x - g.origin[0]) / g.cell_size - 0.5
        fy = (y - g.origin[1]) / g.cell_size - 0.5
        c0, r0 = int(math.floor(fx)), int(math.floor(fy))
        tx, ty = fx - c0, fy - r0
        total_w, acc = 0.0, 0.0
        for dr, dc, w in [(0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                          (1, 0, (1 - tx) * ty), (1, 1, tx * ty)]:
            r, c = r0 + dr, c0 + dc
            if g.in_bounds(r, c) and math.isfinite(self.dist[r, c]):
                acc += w * self.dist[r, c]
                total_w += w
        if total_w == 0.0:
            return math.inf
        return acc / total_w

    def is_reachable(self, x: float, y: float) -> bool:
        return math.isfinite(self.query(x, y))


def _is_free_cell(grid: OccupancyGrid, r: int, c: int) -> bool:
    return grid.in_bounds(r, c) and not grid.blocked[r, c]


def _can_traverse(grid: OccupancyGrid, r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if not _is_free_cell(grid, nr, nc):
        return False
    if dr != 0 and dc != 0:
        return _is_free_cell(grid, r + dr, c) and _is_free_cell(grid, r, c + dc)
    return True
