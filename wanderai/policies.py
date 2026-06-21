from __future__ import annotations
import math
import numpy as np
from .environment import SceneSearchEnv, Action
from .geometry import wrap_angle
from .metrics import EpisodeResult


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env) -> Action:
        return Action(int(self.rng.integers(0, 3)))


class OraclePolicy:
    """Privileged baseline: descends the geodesic field by stepping toward the
    direction that most reduces distance to the ball. Proves the environment is
    solvable and provides an SPL upper bound. Validation only -- it peeks at the
    ground-truth field the learned policy never sees."""

    def act(self, obs, env: SceneSearchEnv) -> Action:
        p = env.pose
        best_dir, best_d = None, math.inf
        for k in range(8):
            ang = k * math.pi / 4
            tx = p.x + env.config.step_size * math.cos(ang)
            ty = p.y + env.config.step_size * math.sin(ang)
            if env.grid.is_blocked_world(tx, ty):
                continue
            d = env.field.query(tx, ty)
            if d < best_d:
                best_d, best_dir = d, ang
        if best_dir is None:
            return Action.TURN_LEFT
        err = wrap_angle(best_dir - p.heading)
        if abs(err) <= env.config.turn / 2:
            return Action.MOVE_FORWARD
        return Action.TURN_LEFT if err > 0 else Action.TURN_RIGHT


def run_episode(env: SceneSearchEnv, policy) -> EpisodeResult:
    obs, info = env.reset()
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, reward, done, info = env.step(action)
    return EpisodeResult(success=info["success"], optimal=info["optimal"],
                         path_length=info["path_length"], steps=info["steps"])
