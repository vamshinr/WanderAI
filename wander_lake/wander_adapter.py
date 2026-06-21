"""EnvironmentAdapter wrapping SceneSearchEnv for eval-protocol multi-turn RFT.

Each rollout = one full episode. The model calls move() each turn; the episode
reward is sparse-ish: 1.0 for reaching the ball, partial credit (fraction of
geodesic distance closed) on timeout. This is the episode-level signal single-step
RFT couldn't give — overshooting/oscillating now tanks the return."""
from __future__ import annotations
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from eval_protocol.mcp import EnvironmentAdapter

from wanderai.scene_gen import random_scene
from wanderai.environment import SceneSearchEnv, EnvConfig

_NAMES = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]


class WanderAdapter(EnvironmentAdapter):
    def get_default_config(self) -> Dict[str, Any]:
        return {"max_steps": 60}

    def create_environment(self, config: Optional[Dict[str, Any]] = None) -> SceneSearchEnv:
        config = config or self.get_default_config()
        seed = int(config.get("seed", 0))
        scene = random_scene(np.random.default_rng(seed))
        return SceneSearchEnv(scene, config=EnvConfig(max_steps=int(config.get("max_steps", 60))))

    def create_environment_with_seed(self, config: Optional[Dict[str, Any]] = None,
                                     seed: Optional[int] = None) -> Tuple[SceneSearchEnv, str, Dict]:
        cfg = dict(config or self.get_default_config())
        if seed is not None:
            cfg["seed"] = seed
        env = self.create_environment(cfg)
        _, info = env.reset()
        return env, env.text_observation(), info

    def reset_environment(self, env: SceneSearchEnv, seed: Optional[int] = None) -> Tuple[str, Dict]:
        _, info = env.reset()
        return env.text_observation(), info

    def step_environment(self, env: SceneSearchEnv, action: int):
        d_before = env._prev_d
        _, _, done, info = env.step(int(action))
        terminated = bool(info["success"])
        truncated = bool(done and not info["success"])
        opt, d_after = info["optimal"], info["geodesic"]
        # DENSE, telescoping reward = the fraction of the optimal path closed THIS
        # step. eval-protocol's get_total_reward SUMS per-step rewards, so this sums
        # to (d_start - d_final)/optimal = total fraction of distance closed, plus a
        # success bonus. The old reward only paid out on env *truncation* (max_steps
        # 60), but the rollout's step budget (~30) ends first, so truncation never
        # fired → every episode scored exactly 0.0 → zero reward variance → no GRPO
        # gradient. Dense shaping gives a signal every step, regardless of step cap.
        if opt > 0 and math.isfinite(d_before) and math.isfinite(d_after):
            reward = (d_before - d_after) / opt
        else:
            reward = 0.0
        if terminated:
            reward += 1.0
        return env.text_observation(), reward, terminated, truncated, info

    def close_environment(self, env: SceneSearchEnv) -> None:
        pass

    def parse_action(self, action_str: str) -> int:
        s = (action_str or "").strip().upper()
        for i, n in enumerate(_NAMES):
            if n in s:
                return i
        if s in ("0", "1", "2"):
            return int(s)
        raise ValueError(f"invalid action '{action_str}'; use MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT")

    def format_observation(self, observation: Any) -> Any:
        return observation
