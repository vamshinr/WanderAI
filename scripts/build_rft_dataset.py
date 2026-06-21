"""Build the RFT dataset: observation prompts + a precomputed reward for each of
the 3 actions (the verifiable scorer). Storing all three rewards per row makes the
Fireworks-side evaluator a trivial self-contained lookup — no need to ship our
package to the training cluster.

Two modes:
  * 2D (default): procedural scenes, symbolic-geometry observations.
      python scripts/build_rft_dataset.py --scenes 30 --out data/rft_train.jsonl
  * 3D vision (v5): roll out in a MuJoCo scene; the prompt is decoded from the
    agent's rendered RGB+depth view (Phase B), while the reward label stays the
    privileged geometric verifier (it never needs pixels).
      python scripts/build_rft_dataset.py --mjcf examples/<train>.xml \
          --episodes 40 --out data/rft_train_v5.jsonl
"""
import argparse
import json
import math
import os
import numpy as np

from wanderai.scene_gen import make_split
from wanderai.policies import OraclePolicy, RandomPolicy
from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.geometry import Pose
from wanderai.rft import build_dataset, single_step_reward, scene_to_dict, scene_from_dict
from wanderai.llm_policy import SYSTEM_PROMPT


def _random_start(scene, field, rng, margin=0.4):
    """A random free, reachable pose — diversifies coverage of a single scene."""
    b = scene.bounds
    for _ in range(200):
        x = rng.uniform(b.min_x + margin, b.max_x - margin)
        y = rng.uniform(b.min_y + margin, b.max_y - margin)
        if scene.is_free(x, y) and math.isfinite(field.query(x, y)):
            return Pose(x, y, rng.uniform(-math.pi, math.pi))
    return scene.agent_start


def build_vision_states(mjcf, episodes, max_steps, seed, random_states=600):
    """Roll out oracle+random in the 3D scene; capture each state's *vision*
    observation (prompt) + serialized scene/pose (so rewards can be scored).

    Rollouts alone are forward-heavy (the oracle mostly drives straight), so we
    also sample standalone random poses with uniform headings — at a random
    heading the best move is just as often a left/right turn, which balances the
    three action classes the model must learn."""
    from wanderai.mujoco_renderer import load_mjcf_3d
    from wanderai.observation import visit_key

    scene, renderer = load_mjcf_3d(mjcf)
    print(f"    3D scene: {len(scene.obstacles)} obstacles, ball={tuple(round(c,1) for c in scene.ball)}")
    venv = SceneSearchEnv(scene, renderer=renderer,
                          config=EnvConfig(max_steps=max_steps, perception="vision"))
    venv.reset()
    rng = np.random.default_rng(seed)
    policies = [OraclePolicy(), RandomPolicy(seed=seed)]
    states = []

    def capture():
        states.append({"obs": venv.text_observation(),
                       "scene": scene_to_dict(scene),
                       "pose": [venv.pose.x, venv.pose.y, venv.pose.heading]})

    for ep in range(episodes):
        policy = policies[ep % len(policies)]
        start = _random_start(scene, venv.field, rng)
        venv.pose = start
        venv.history = []
        venv.visited = {visit_key(start.x, start.y)}
        venv._prev_d = venv.field.query(start.x, start.y)
        for _ in range(max_steps):
            capture()
            _, _, done, _ = venv.step(policy.act(None, venv))
            if done:
                break
        if (ep + 1) % 10 == 0:
            print(f"    rolled {ep+1}/{episodes} episodes, {len(states)} states")

    # uniform random-pose/heading coverage (balances turn-left / turn-right / forward)
    for _ in range(random_states):
        venv.pose = _random_start(scene, venv.field, rng)
        venv.history = []
        venv.visited = {visit_key(venv.pose.x, venv.pose.y)}
        capture()
    print(f"    + {random_states} random-pose states -> {len(states)} total")
    return states, scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--episodes", type=int, default=40, help="3D mode: rollouts per scene")
    ap.add_argument("--random-states", type=int, default=600,
                    help="3D mode: extra uniform random poses (balances turn classes)")
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mjcf", default=None, help="MuJoCo scene XML -> 3D vision dataset")
    ap.add_argument("--out", default="data/rft_train.jsonl")
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True,
                    help="keep the natural (forward-heavy) action distribution")
    args = ap.parse_args()

    reward_env = None
    if args.mjcf:
        print(f">>> 3D vision dataset from {args.mjcf}")
        states, scene = build_vision_states(args.mjcf, args.episodes, args.max_steps,
                                            args.seed, random_states=args.random_states)
        # One geometry env (occupancy + geodesic for `scene`) reused for every
        # reward lookup — the field build dominates cost.
        reward_env = SceneSearchEnv(scene, config=EnvConfig())
        reward_env.reset()
    else:
        train, _ = make_split(args.scenes, 0, seed=args.seed)
        # Oracle visits good states; random covers messy ones the model must recover from.
        states = build_dataset(train, [OraclePolicy(), RandomPolicy(seed=0)],
                               max_steps=args.max_steps)

    # Bucket rows by the action the reward prefers, so we can balance the classes.
    by_action = {a.name: [] for a in Action}
    for s in states:
        scene = scene_from_dict(s["scene"])
        env = reward_env if (reward_env is not None) else None
        rewards = {a.name: round(single_step_reward(scene, s["pose"], a, env=env), 4)
                   for a in Action}
        if len(set(rewards.values())) == 1:        # no signal — every action equal
            continue
        row = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["obs"] + "\nWhat is your next action?"},
            ],
            "ground_truth": json.dumps({"rewards": rewards}),
        }
        by_action[max(rewards, key=rewards.get)].append(row)

    rng = np.random.default_rng(args.seed)
    if args.balance:
        # Oversample minority classes (turns) up to the majority count, so the model
        # isn't biased toward MOVE_FORWARD just because forward is the plurality.
        target = max((len(v) for v in by_action.values()), default=0)
        rows = []
        for bucket in by_action.values():
            if bucket:
                rows.extend(bucket[i] for i in rng.integers(0, len(bucket), size=target))
    else:
        rows = [r for bucket in by_action.values() for r in bucket]
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    counts = {k: len(v) for k, v in by_action.items()}
    print(f"wrote {len(rows)} rows to {args.out} (balanced={args.balance}); "
          f"raw best-action counts: {counts}")


if __name__ == "__main__":
    main()
