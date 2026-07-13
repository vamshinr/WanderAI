"""2D benchmark: all locally-runnable policies over a held-out procedural split.

    python3 scripts/run_benchmark.py --rooms 25 --episodes 2 --seed 7 \
        --out-json docs/research/data/benchmark_2d.json \
        --out-md docs/research/benchmarks_2d.md

Policies: random (floor), FBE variants (nearest / cost-utility frontier
selection; memory-budget ablations), locally trained RL (if weights given),
and the privileged oracle (ceiling). All numbers are produced by this script —
nothing is hand-entered.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wanderai.benchmark import aggregate, run_policy, save_report, to_markdown
from wanderai.environment import EnvConfig
from wanderai.frontier_policy import FrontierPolicy, FrontierConfig
from wanderai.policies import OraclePolicy, RandomPolicy
from wanderai.scene_gen import make_split


def build_policies(args):
    policies = {
        "random": lambda seed: RandomPolicy(seed=seed),
        "fbe-nearest": lambda seed: FrontierPolicy(
            FrontierConfig(selection="nearest", seed=seed)),
        "fbe": lambda seed: FrontierPolicy(FrontierConfig(seed=seed)),
        "fbe-mem2000": lambda seed: FrontierPolicy(
            FrontierConfig(budget_cells=2000, seed=seed)),
        "fbe-mem500": lambda seed: FrontierPolicy(
            FrontierConfig(budget_cells=500, seed=seed)),
        "fbe-mem150": lambda seed: FrontierPolicy(
            FrontierConfig(budget_cells=150, seed=seed)),
    }
    if args.rl_weights and os.path.exists(args.rl_weights):
        from wanderai.rl_local import TrainedLocalPolicy
        policies["rl-local"] = (
            lambda seed, path=args.rl_weights: TrainedLocalPolicy.load(path))
    if args.rl_hint_weights and os.path.exists(args.rl_hint_weights):
        from wanderai.rl_local import TrainedLocalPolicy
        policies["rl-local-hint"] = (
            lambda seed, path=args.rl_hint_weights: TrainedLocalPolicy.load(path))
    policies["oracle (privileged)"] = lambda seed: OraclePolicy()
    return policies


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=25, help="held-out rooms")
    ap.add_argument("--episodes", type=int, default=2, help="episodes per room")
    ap.add_argument("--seed", type=int, default=7, help="scene-split seed")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--rl-weights", default="", help="rl_local weights JSON")
    ap.add_argument("--rl-hint-weights", default="",
                    help="rl_local weights JSON trained with --hint")
    ap.add_argument("--out-json", default="docs/research/data/benchmark_2d.json")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    _, test = make_split(0, args.rooms, seed=args.seed)
    cfg = EnvConfig(max_steps=args.max_steps)

    rows = []
    for name, factory in build_policies(args).items():
        results = run_policy(factory, test, cfg,
                             episodes_per_scene=args.episodes)
        row = aggregate(name, results)
        rows.append(row)
        print(f"{name:22s} SR {row['success_rate']:.3f}  SPL {row['spl']:.3f}  "
              f"SoftSPL {row['soft_spl']:.3f}  DTS {row['dts']:.2f}")

    meta = {"benchmark": "2d-heldout", "rooms": args.rooms,
            "episodes_per_room": args.episodes, "split_seed": args.seed,
            "max_steps": args.max_steps,
            "env": {"step_size": cfg.step_size, "turn_deg": 30,
                    "success_radius": cfg.success_radius}}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_report(args.out_json, rows, meta)
    caption = (f"2D held-out benchmark: {args.rooms} unseen procedural rooms "
               f"x {args.episodes} episodes (split seed {args.seed}, "
               f"max {args.max_steps} steps)")
    md = to_markdown(rows, caption)
    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
        with open(args.out_md, "w") as fh:
            fh.write(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
