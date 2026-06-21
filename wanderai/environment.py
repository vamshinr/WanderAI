from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
from .scene import Scene
from .geometry import Pose, wrap_angle
from .occupancy import OccupancyGrid
from .distance_field import DistanceField
from .renderer import Renderer, StubRenderer
from .observation import observe, observation_text, visit_key


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
    perception: str = "geometry"  # "geometry" (symbolic) or "vision" (RGB+depth)


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
        self.history: list[Action] = []
        self.visited: set = set()

    def text_observation(self) -> str:
        """Egocentric symbolic view as text — the observation the RFT text policy
        reads. Includes episodic (visited-areas) memory, in-context, because where
        the agent has been *this episode* cannot live in the model's weights.

        In ``perception="vision"`` mode the view is decoded from the renderer's
        RGB+depth frame (Phase B) instead of from privileged geometry; the text
        format is identical either way, so the policy is unchanged."""
        if self.config.perception == "vision":
            from .perception import perceive
            obs = perceive(self.renderer, self.scene, self.pose,
                           history=self.history, visited=self.visited)
        else:
            obs = observe(self.scene, self.pose, history=self.history,
                          visited=self.visited)
        return observation_text(obs)

    def reset(self):
        self.grid = OccupancyGrid.from_scene(self.scene, self.config.cell_size)
        self.field = DistanceField.from_grid(self.grid, self.scene.ball)
        self.pose = self.scene.agent_start
        self.steps = 0
        self.path_length = 0.0
        self.history = []
        self.visited = {visit_key(self.pose.x, self.pose.y)}
        self.optimal = self.field.query(self.pose.x, self.pose.y)
        self._prev_d = self.optimal
        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": self._prev_d, "optimal": self.optimal,
                "path_length": 0.0, "success": False, "collision": False, "steps": 0,
                "obs_text": self.text_observation()}
        return obs, info

    def step(self, action: int):
        cfg = self.config
        self.history.append(Action(action))
        collision = False
        if action == Action.MOVE_FORWARD:
            nx = self.pose.x + cfg.step_size * math.cos(self.pose.heading)
            ny = self.pose.y + cfg.step_size * math.sin(self.pose.heading)
            if self.grid.is_blocked_world(nx, ny):
                collision = True
            else:
                self.path_length += math.hypot(nx - self.pose.x, ny - self.pose.y)
                self.pose = Pose(nx, ny, self.pose.heading)
        elif action == Action.TURN_LEFT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading + cfg.turn))
        elif action == Action.TURN_RIGHT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading - cfg.turn))

        self.visited.add(visit_key(self.pose.x, self.pose.y))
        d = self.field.query(self.pose.x, self.pose.y)
        if math.isfinite(d) and math.isfinite(self._prev_d):
            progress = self._prev_d - d
        else:
            progress = 0.0
        self._prev_d = d
        self.steps += 1

        euclid_to_ball = math.hypot(self.scene.ball[0] - self.pose.x,
                                    self.scene.ball[1] - self.pose.y)
        success = euclid_to_ball <= cfg.success_radius
        reward = cfg.alpha * progress - cfg.beta - (cfg.kappa if collision else 0.0)
        if success:
            reward += cfg.goal_reward
        done = success or self.steps >= cfg.max_steps

        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": d, "optimal": self.optimal, "path_length": self.path_length,
                "success": success, "collision": collision, "steps": self.steps,
                "obs_text": self.text_observation()}
        return obs, reward, done, info
