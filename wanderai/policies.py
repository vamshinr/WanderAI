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
        current_d = env.field.query(p.x, p.y)
        fx = p.x + env.config.step_size * math.cos(p.heading)
        fy = p.y + env.config.step_size * math.sin(p.heading)
        if not _move_blocked(env, p.x, p.y, fx, fy):
            forward_d = env.field.query(fx, fy)
            if forward_d < current_d - 1e-9:
                return Action.MOVE_FORWARD

        best_dir, best_d = None, math.inf
        for k in range(8):
            ang = k * math.pi / 4
            tx = p.x + env.config.step_size * math.cos(ang)
            ty = p.y + env.config.step_size * math.sin(ang)
            if _move_blocked(env, p.x, p.y, tx, ty):
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


def _move_blocked(env: SceneSearchEnv, x0: float, y0: float, x1: float, y1: float) -> bool:
    if hasattr(env, "_segment_blocked"):
        return bool(env._segment_blocked(x0, y0, x1, y1))
    return bool(env.grid.is_blocked_world(x1, y1))


def run_episode(env: SceneSearchEnv, policy) -> EpisodeResult:
    obs, info = env.reset()
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, reward, done, info = env.step(action)
    return EpisodeResult(success=bool(info["success"]), optimal=float(info["optimal"]),
                         path_length=float(info["path_length"]), steps=int(info["steps"]))
