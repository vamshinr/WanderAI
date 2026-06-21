"""Build the RFT dataset: observation prompts + a precomputed reward for each of
the 3 actions (the verifiable scorer). Storing all three rewards per row makes the
Fireworks-side evaluator a trivial self-contained lookup — no need to ship our
package to the training cluster.

Usage: python scripts/build_rft_dataset.py --scenes 30 --max-steps 40 --out data/rft_train.jsonl
"""
import argparse
import json
import os
import numpy as np
from wanderai.scene_gen import make_split
from wanderai.policies import OraclePolicy, RandomPolicy
from wanderai.environment import Action
from wanderai.rft import build_dataset, single_step_reward, scene_from_dict
from wanderai.llm_policy import SYSTEM_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="data/rft_train.jsonl")
    ap.add_argument("--no-balance", dest="balance", action="store_false", default=True,
                    help="keep the natural (forward-heavy) action distribution")
    args = ap.parse_args()

    train, _ = make_split(args.scenes, 0, seed=args.seed)
    # Oracle visits good states; random covers messy ones the model must recover from.
    states = build_dataset(train, [OraclePolicy(), RandomPolicy(seed=0)], max_steps=args.max_steps)

    # Bucket rows by the action the reward prefers, so we can balance the classes.
    by_action = {a.name: [] for a in Action}
    for s in states:
        scene = scene_from_dict(s["scene"])
        rewards = {a.name: round(single_step_reward(scene, s["pose"], a), 4) for a in Action}
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
