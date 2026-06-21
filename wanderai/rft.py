"""Reinforcement fine-tuning (A5) core: the verifiable 0–1 reward and the
group-relative (GRPO) training signal it produces.

RFT loop = generate (model acts in the env) → score (this reward) → update
(push weights toward higher-reward trajectories). The weight update runs on
Fireworks (see `docs/rft_launch.md`); everything needed to *drive and score* it
lives here, and the GRPO preview shows the reward yields a real learning signal
without paying for training.

The reward is dense on purpose. Pure success is too sparse for an untrained model
that rarely reaches the ball — it would see all-zero reward and have no gradient.
So we give partial credit for *progress* (fraction of the geodesic distance
closed) plus an efficiency (SPL) bonus on success. All in [0, 1] as Fireworks
RFT requires."""

from __future__ import annotations
import math
from dataclasses import dataclass
from .environment import SceneSearchEnv, EnvConfig
from .scene import Scene

PROGRESS_WEIGHT = 0.7      # partial credit for closing geodesic distance
SPL_WEIGHT = 0.3          # efficiency bonus, only on success


@dataclass
class Rollout:
    reward: float          # in [0, 1] — the RFT training reward
    success: bool
    progress: float        # fraction of initial geodesic distance closed, [0,1]
    spl: float             # optimal / path if success else 0
    steps: int


def episode_reward(optimal: float, final_geodesic: float, path_length: float,
                   success: bool, steps: int) -> Rollout:
    """Map a finished episode to a 0–1 RFT reward."""
    if optimal > 0 and math.isfinite(final_geodesic):
        progress = max(0.0, min(1.0, (optimal - final_geodesic) / optimal))
    else:
        progress = 1.0 if success else 0.0
    spl = (optimal / max(path_length, optimal)) if (success and optimal > 0) else 0.0
    reward = PROGRESS_WEIGHT * progress + SPL_WEIGHT * spl
    if success:
        reward = max(reward, PROGRESS_WEIGHT)      # reaching the ball is never punished
    return Rollout(max(0.0, min(1.0, reward)), success, progress, spl, steps)


def run_scored(scene: Scene, policy, config: EnvConfig | None = None) -> Rollout:
    """Run one episode of `policy` on `scene` and score it for RFT."""
    env = SceneSearchEnv(scene, config=config or EnvConfig(max_steps=400))
    _, info = env.reset()
    done = False
    while not done:
        _, _, done, info = env.step(policy.act(None, env))
    return episode_reward(info["optimal"], info["geodesic"], info["path_length"],
                          info["success"], info["steps"])


def group_advantages(rewards: list[float]) -> list[float]:
    """GRPO advantage: standardize rewards within a group (mean 0, unit std).
    Positive => better than the group average => reinforced; negative => suppressed.
    Zero variance (all equal) => no signal, which is correct."""
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(var)
    if std < 1e-8:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def grpo_preview(scene: Scene, policy_factory, group_size: int = 4,
                 config: EnvConfig | None = None) -> dict:
    """Sample a group of trajectories from `policy_factory()` on one scene, score
    each, and compute GRPO advantages — the exact signal RFT trains on. Run with a
    stochastic policy (temperature > 0) so the group has reward variance."""
    rolls = [run_scored(scene, policy_factory(), config) for _ in range(group_size)]
    rewards = [r.reward for r in rolls]
    return {
        "rewards": rewards,
        "advantages": group_advantages(rewards),
        "mean": sum(rewards) / len(rewards),
        "successes": sum(r.success for r in rolls),
        "rollouts": rolls,
    }
