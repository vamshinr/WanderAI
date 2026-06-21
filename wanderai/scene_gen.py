"""Procedural scene generator. Produces varied, *solvable* rooms and a
deterministic train/test split — the substrate for the scene-agnostic claim:
train on one set of rooms, measure generalization on a held-out set."""

from __future__ import annotations
import math
import numpy as np
from .geometry import AABB, Pose
from .scene import Scene
from .occupancy import OccupancyGrid
from .distance_field import DistanceField


def _reachable(scene: Scene, cell_size: float = 0.1) -> bool:
    grid = OccupancyGrid.from_scene(scene, cell_size)
    field = DistanceField.from_grid(grid, scene.ball)
    return math.isfinite(field.query(scene.agent_start.x, scene.agent_start.y))


def _sample_free_point(scene_bounds: AABB, obstacles, agent_radius, rng, margin=0.3):
    probe = Scene(scene_bounds, obstacles, (0, 0), Pose(0, 0, 0), agent_radius)
    for _ in range(200):
        x = rng.uniform(scene_bounds.min_x + margin, scene_bounds.max_x - margin)
        y = rng.uniform(scene_bounds.min_y + margin, scene_bounds.max_y - margin)
        if probe.is_free(x, y):
            return x, y
    return None


def random_scene(rng: np.random.Generator, agent_radius: float = 0.2,
                 min_separation: float = 2.0, max_attempts: int = 60) -> Scene:
    """A random room (size, obstacles, ball, start) guaranteed free-placed,
    separated, and with the ball reachable from the start. Resamples until valid."""
    for _ in range(max_attempts):
        w = float(rng.uniform(5.0, 8.0))
        h = float(rng.uniform(5.0, 8.0))
        bounds = AABB(0, 0, w, h)

        n_obs = int(rng.integers(2, 5))
        obstacles = []
        for _ in range(n_obs):
            ow = float(rng.uniform(0.6, 1.8))
            oh = float(rng.uniform(0.6, 1.8))
            ox = float(rng.uniform(0.5, w - 0.5 - ow))
            oy = float(rng.uniform(0.5, h - 0.5 - oh))
            obstacles.append(AABB(ox, oy, ox + ow, oy + oh))

        ball = _sample_free_point(bounds, obstacles, agent_radius, rng)
        start = _sample_free_point(bounds, obstacles, agent_radius, rng)
        if ball is None or start is None:
            continue
        if math.hypot(ball[0] - start[0], ball[1] - start[1]) < min_separation:
            continue

        heading = float(rng.uniform(-math.pi, math.pi))
        scene = Scene(bounds, obstacles, ball, Pose(start[0], start[1], heading), agent_radius)
        if _reachable(scene):
            return scene

    # Fallback: an obstacle-light room is always solvable.
    bounds = AABB(0, 0, 6, 6)
    return Scene(bounds, [AABB(2.5, 2.5, 3.5, 3.5)], (5.2, 5.2),
                 Pose(0.6, 0.6, 0.0), agent_radius)


def make_split(n_train: int, n_test: int, seed: int = 0):
    """Deterministic disjoint train/test scene lists from a single seed."""
    rng = np.random.default_rng(seed)
    train = [random_scene(rng) for _ in range(n_train)]
    test = [random_scene(rng) for _ in range(n_test)]
    return train, test
