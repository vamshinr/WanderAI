"""Show the RFT training signal on the real model — no training, no cost beyond
inference. Samples a group of LLM trajectories on one scene, scores each with the
geodesic reward, and prints the GRPO advantages RFT would learn from.

Usage:  python scripts/grpo_preview.py --group 4 --max-steps 14 --temperature 0.8
"""
import argparse
from wanderai.scene_gen import make_split
from wanderai.environment import EnvConfig
from wanderai.llm_policy import LLMPolicy
from wanderai.rft import grpo_preview


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=4, help="trajectories per group")
    ap.add_argument("--max-steps", type=int, default=14, help="steps per trajectory (keep small — each is LLM calls)")
    ap.add_argument("--temperature", type=float, default=0.8, help="sampling temp (>0 for variance)")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    _, test = make_split(0, 1, seed=args.seed)
    scene = test[0]
    cfg = EnvConfig(max_steps=args.max_steps)
    print(f"GRPO preview · group={args.group} · max_steps={args.max_steps} · temp={args.temperature}")
    print("sampling trajectories from gpt-oss-20b (this calls the model many times)...")

    out = grpo_preview(scene, lambda: LLMPolicy(temperature=args.temperature),
                       group_size=args.group, config=cfg)

    print(f"\nmean reward {out['mean']:.3f} · successes {out['successes']}/{args.group}")
    print(f"{'traj':>4}  {'reward':>7}  {'advantage':>9}")
    for i, (r, a) in enumerate(zip(out["rewards"], out["advantages"])):
        mark = "↑ reinforce" if a > 0.05 else ("↓ suppress" if a < -0.05 else "· neutral")
        print(f"{i:>4}  {r:>7.3f}  {a:>+9.3f}  {mark}")
    spread = max(out["advantages"]) - min(out["advantages"]) if out["advantages"] else 0
    print(f"\nadvantage spread {spread:.3f} — nonzero means a real learning signal exists.")


if __name__ == "__main__":
    main()
