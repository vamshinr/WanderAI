from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # radians


@dataclass(frozen=True)
class AABB:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def inflate(self, r: float) -> "AABB":
        return AABB(self.min_x - r, self.min_y - r, self.max_x + r, self.max_y + r)


def wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    a = math.fmod(a, 2 * math.pi)
    if a <= -math.pi:
        a += 2 * math.pi
    elif a > math.pi:
        a -= 2 * math.pi
    return a


def segment_intersects_aabb(x0, y0, x1, y1, box: AABB) -> bool:
    """Liang-Barsky slab clipping; True if the segment touches the box."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - box.min_x, box.max_x - x0, y0 - box.min_y, box.max_y - y0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True
