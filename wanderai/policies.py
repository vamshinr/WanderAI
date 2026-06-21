from __future__ import annotations
import math
import numpy as np
from .environment import SceneSearchEnv, Action
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
        p, step, turn = env.pose, env.config.step_size, env.config.turn

        def cell_d(angle: float) -> float:
            """Geodesic at the cell one step along `angle`; inf if blocked."""
            tx = p.x + step * math.cos(angle)
            ty = p.y + step * math.sin(angle)
            if env.grid.is_blocked_world(tx, ty):
                return math.inf
            return env.field.query(tx, ty)

        # One-step lookahead over the three actions. MOVE_FORWARD is only ever a
        # candidate when the cell directly ahead (at the actual heading) is free,
        # so the oracle never drives into a wall it merely turned toward.
        options = {
            Action.MOVE_FORWARD: cell_d(p.heading),
            Action.TURN_LEFT: cell_d(p.heading + turn),
            Action.TURN_RIGHT: cell_d(p.heading - turn),
        }
        best = min(options, key=lambda a: options[a])
        if math.isinf(options[best]):
            return Action.TURN_LEFT      # boxed in — rotate to find an opening
        return best


def run_episode(env: SceneSearchEnv, policy) -> EpisodeResult:
    obs, info = env.reset()
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, reward, done, info = env.step(action)
    return EpisodeResult(success=info["success"], optimal=info["optimal"],
                         path_length=info["path_length"], steps=info["steps"])
