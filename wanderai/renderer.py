from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np
from .scene import Scene
from .geometry import Pose, segment_intersects_aabb, wrap_angle


def ball_visible(scene: Scene, pose: Pose, fov: float, max_view: float) -> bool:
    """The ball is visible if it lies within the field-of-view half-angle and
    within max_view, with an unobstructed line of sight to the agent."""
    bx, by = scene.ball
    dx, dy = bx - pose.x, by - pose.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return True                      # standing on it
    if dist > max_view:
        return False
    bearing = wrap_angle(math.atan2(dy, dx) - pose.heading)
    if abs(bearing) > fov / 2:
        return False
    for ob in scene.obstacles:
        if segment_intersects_aabb(pose.x, pose.y, bx, by, ob):
            return False
    return True


class Renderer(ABC):
    @abstractmethod
    def render(self, scene: Scene, pose: Pose) -> np.ndarray:
        ...


class StubRenderer(Renderer):
    """Dependency-free stand-in for the Antim Labs renderer. Produces a flat
    gray image, drawing a red vertical band at the ball's bearing when the ball
    is visible. Occlusion and FOV match `ball_visible`."""

    def __init__(self, width: int = 64, height: int = 64,
                 fov: float = math.pi / 2, max_view: float = 8.0):
        self.width = width
        self.height = height
        self.fov = fov
        self.max_view = max_view

    def render(self, scene: Scene, pose: Pose) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 110, dtype=np.uint8)  # gray floor
        if ball_visible(scene, pose, self.fov, self.max_view):
            bx, by = scene.ball
            bearing = wrap_angle(math.atan2(by - pose.y, bx - pose.x) - pose.heading)
            col = int((bearing / self.fov + 0.5) * self.width)
            col = max(0, min(self.width - 1, col))
            dist = math.hypot(bx - pose.x, by - pose.y)
            half = max(2, int(self.width * 0.12 / max(dist, 0.5)))
            lo, hi = max(0, col - half), min(self.width, col + half + 1)
            img[:, lo:hi, 0] = 220   # red
            img[:, lo:hi, 1] = 30
            img[:, lo:hi, 2] = 30
        return img
