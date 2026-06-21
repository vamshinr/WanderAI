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
from .environment import SceneSearchEnv, EnvConfig, Action
from .geometry import AABB, Pose
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


def scene_to_dict(s: Scene) -> dict:
    return {"bounds": [s.bounds.min_x, s.bounds.min_y, s.bounds.max_x, s.bounds.max_y],
            "obstacles": [[o.min_x, o.min_y, o.max_x, o.max_y] for o in s.obstacles],
            "ball": list(s.ball), "agent_radius": s.agent_radius,
            "start": [s.agent_start.x, s.agent_start.y, s.agent_start.heading]}


def scene_from_dict(d: dict) -> Scene:
    return Scene(AABB(*d["bounds"]), [AABB(*o) for o in d["obstacles"]],
                 tuple(d["ball"]), Pose(*d["start"]), d["agent_radius"])


def _downhill_dir(field, x, y, step) -> float:
    """Direction (radians) that most reduces geodesic distance from (x, y) — the
    way the agent *should* be heading."""
    best_a, best_d = 0.0, math.inf
    for k in range(8):
        a = k * math.pi / 4
        d = field.query(x + step * math.cos(a), y + step * math.sin(a))
        if d < best_d:
            best_d, best_a = d, a
    return best_a


def single_step_reward(scene: Scene, pose, action: int, config: EnvConfig | None = None,
                       env: SceneSearchEnv | None = None) -> float:
    """Score ONE action from a pose, in [0, 1] — the RFT verifier. Crucially it is
    *direction-aware*, so it distinguishes turning toward vs. away from the goal:
      reach ball -> 1.0;  collide -> 0.0;
      MOVE_FORWARD -> 0.5 ± geodesic distance closed;
      TURN_*       -> 0.5 ± how much better it orients you toward the downhill
                      (toward-goal) direction.
    Without the orientation term, both turns would tie at 0.5 and the model could
    never learn which way to turn.

    Pass a pre-reset `env` (same scene) to reuse its occupancy/geodesic field — the
    field build dominates cost, so reuse makes scoring thousands of states cheap."""
    if env is None:
        env = SceneSearchEnv(scene, config=config or EnvConfig())
        env.reset()
    env.history = []
    env.steps = 0
    env.pose = Pose(*pose)
    field, step = env.field, env.config.step_size
    prev_d = field.query(env.pose.x, env.pose.y)
    env._prev_d = prev_d
    theta = _downhill_dir(field, env.pose.x, env.pose.y, step)
    align_before = math.cos(env.pose.heading - theta)

    _, _, _, info = env.step(action)
    if info["success"]:
        return 1.0
    if info["collision"]:
        return 0.0

    if action == Action.MOVE_FORWARD:
        d = info["geodesic"]
        if not (math.isfinite(prev_d) and math.isfinite(d)):
            return 0.5
        progress = (prev_d - d) / step                       # +1 ideal toward, -1 away
        return max(0.0, min(1.0, 0.5 + 0.5 * progress))
    # a turn: did it orient us better toward the downhill direction?
    align_after = math.cos(env.pose.heading - theta)
    return max(0.0, min(1.0, 0.5 + 1.5 * (align_after - align_before)))


def build_dataset(scenes, policies, max_steps: int = 40, config: EnvConfig | None = None):
    """Roll out `policies` across `scenes` and capture each visited state as a
    training row: the egocentric observation (prompt) + the serialized scene/pose
    (so the verifier can score any action the model proposes). Mixing oracle and
    random policies covers both good and bad states the model must handle."""
    config = config or EnvConfig(max_steps=max_steps)
    rows = []
    for i, scene in enumerate(scenes):
        policy = policies[i % len(policies)]
        env = SceneSearchEnv(scene, config=config)
        env.reset()
        done = False
        for _ in range(max_steps):
            rows.append({"obs": env.text_observation(),
                         "scene": scene_to_dict(scene),
                         "pose": [env.pose.x, env.pose.y, env.pose.heading]})
            _, _, done, _ = env.step(policy.act(None, env))
            if done:
                break
    return rows


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
