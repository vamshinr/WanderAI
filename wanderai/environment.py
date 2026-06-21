from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
from .scene import Scene
from .geometry import Pose, segment_intersects_aabb, wrap_angle
from .occupancy import OccupancyGrid
from .distance_field import DistanceField
from .renderer import Renderer, StubRenderer


class Action(IntEnum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2


@dataclass
class EnvConfig:
    cell_size: float = 0.1
    step_size: float = 0.25
    turn: float = math.radians(30)
    success_radius: float = 0.3
    max_steps: int = 200
    alpha: float = 1.0          # geodesic progress weight
    beta: float = 0.02          # per-step time penalty
    kappa: float = 0.1          # collision penalty
    goal_reward: float = 10.0   # terminal success bonus
    gamma: float = 0.99


class SceneSearchEnv:
    """Gym-style environment. The agent navigates continuous 2D space toward the
    ball; observations are egocentric RGB only. Reward is privileged: it is
    computed from the ground-truth geodesic distance the agent never sees.

      r_t = alpha * (d_{t-1} - d_t)   # geodesic progress (potential-based shaping)
            - beta                    # time penalty -> rewards speed
            - kappa * 1[collision]
            + goal_reward * 1[success]
    """

    def __init__(self, scene: Scene, renderer: Renderer | None = None,
                 config: EnvConfig | None = None):
        self.scene = scene
        self.config = config or EnvConfig()
        self.renderer = renderer or StubRenderer()
        self.grid: OccupancyGrid | None = None
        self.field: DistanceField | None = None
        self.pose: Pose = scene.agent_start
        self.steps = 0
        self.path_length = 0.0
        self.optimal = math.inf
        self._prev_d = math.inf
        self._done = False

    def reset(self):
        grid = OccupancyGrid.from_scene(self.scene, self.config.cell_size)
        start = self.scene.agent_start
        if grid.is_blocked_world(start.x, start.y):
            raise ValueError("agent start is blocked or out of bounds")
        if grid.is_blocked_world(*self.scene.ball):
            raise ValueError("goal is blocked or out of bounds")

        field = DistanceField.from_grid(grid, self.scene.ball)
        optimal = field.query(start.x, start.y)
        if not math.isfinite(optimal):
            raise ValueError("start and goal are unreachable")

        self.grid = grid
        self.field = field
        self.pose = start
        self.steps = 0
        self.path_length = 0.0
        self.optimal = optimal
        self._prev_d = optimal
        self._done = False
        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": self._prev_d, "optimal": self.optimal,
                "path_length": 0.0, "success": False, "collision": False, "steps": 0}
        return obs, info

    def step(self, action: int):
        if self.grid is None or self.field is None:
            raise RuntimeError("reset must be called before step")
        if self._done:
            raise RuntimeError("cannot step after episode is done")
        try:
            action = Action(action)
        except ValueError:
            raise ValueError(f"invalid action: {action}") from None

        cfg = self.config
        collision = False
        if action == Action.MOVE_FORWARD:
            nx = self.pose.x + cfg.step_size * math.cos(self.pose.heading)
            ny = self.pose.y + cfg.step_size * math.sin(self.pose.heading)
            if self._segment_blocked(self.pose.x, self.pose.y, nx, ny):
                collision = True
            else:
                self.path_length += math.hypot(nx - self.pose.x, ny - self.pose.y)
                self.pose = Pose(nx, ny, self.pose.heading)
        elif action == Action.TURN_LEFT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading + cfg.turn))
        elif action == Action.TURN_RIGHT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading - cfg.turn))

        d = self.field.query(self.pose.x, self.pose.y)
        if math.isfinite(d) and math.isfinite(self._prev_d):
            progress = self._prev_d - d
        else:
            progress = 0.0
        self._prev_d = d
        self.steps += 1

        success = math.isfinite(d) and d <= cfg.success_radius
        reward = cfg.alpha * progress - cfg.beta - (cfg.kappa if collision else 0.0)
        if success:
            reward += cfg.goal_reward
        done = success or self.steps >= cfg.max_steps
        self._done = done

        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": d, "optimal": self.optimal, "path_length": self.path_length,
                "success": success, "collision": collision, "steps": self.steps}
        return obs, reward, done, info

    def _segment_blocked(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        if self.grid is None:
            return True
        if self.grid.is_blocked_world(x1, y1):
            return True
        for obstacle in self.scene.obstacles:
            if segment_intersects_aabb(x0, y0, x1, y1, obstacle.inflate(self.scene.agent_radius)):
                return True
        return False
