from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .scene import Scene


@dataclass
class OccupancyGrid:
    blocked: np.ndarray            # bool [nrows, ncols], row=y, col=x
    cell_size: float
    origin: tuple[float, float]    # (min_x, min_y)

    @property
    def nrows(self) -> int:
        return self.blocked.shape[0]

    @property
    def ncols(self) -> int:
        return self.blocked.shape[1]

    @classmethod
    def from_scene(cls, scene: Scene, cell_size: float = 0.1) -> "OccupancyGrid":
        b = scene.bounds
        ncols = int(round((b.max_x - b.min_x) / cell_size))
        nrows = int(round((b.max_y - b.min_y) / cell_size))
        blocked = np.zeros((nrows, ncols), dtype=bool)
        for r in range(nrows):
            y = b.min_y + (r + 0.5) * cell_size
            for c in range(ncols):
                x = b.min_x + (c + 0.5) * cell_size
                blocked[r, c] = not scene.is_free(x, y)
        return cls(blocked, cell_size, (b.min_x, b.min_y))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.origin[0]) / self.cell_size)
        r = int((y - self.origin[1]) / self.cell_size)
        return r, c

    def cell_to_world(self, r: int, c: int) -> tuple[float, float]:
        x = self.origin[0] + (c + 0.5) * self.cell_size
        y = self.origin[1] + (r + 0.5) * self.cell_size
        return x, y

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def is_blocked_world(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        if not self.in_bounds(r, c):
            return True
        return bool(self.blocked[r, c])
