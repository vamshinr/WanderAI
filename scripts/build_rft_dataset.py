"""Build the RFT dataset: observation prompts + a precomputed reward for each of
the 3 actions (the verifiable scorer). Storing all three rewards per row makes the
Fireworks-side evaluator a trivial self-contained lookup — no need to ship our
package to the training cluster.

Usage: python scripts/build_rft_dataset.py --scenes 30 --max-steps 40 --out data/rft_train.jsonl
"""
import argparse
import json
import os
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
    args = ap.parse_args()

    train, _ = make_split(args.scenes, 0, seed=args.seed)
    # Oracle visits good states; random covers messy ones the model must recover from.
    states = build_dataset(train, [OraclePolicy(), RandomPolicy(seed=0)], max_steps=args.max_steps)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n = 0
    with open(args.out, "w") as fh:
        for s in states:
            scene = scene_from_dict(s["scene"])
            rewards = {a.name: round(single_step_reward(scene, s["pose"], a), 4) for a in Action}
            # Skip degenerate states where every action scores the same (no signal).
            if len(set(rewards.values())) == 1:
                continue
            row = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": s["obs"] + "\nWhat is your next action?"},
                ],
                "ground_truth": json.dumps({"rewards": rewards}),
            }
            fh.write(json.dumps(row) + "\n")
            n += 1
    print(f"wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
