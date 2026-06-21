"""Evaluate a policy across a set of scenes — the scene-agnostic test.

Train (eventually) on one set of rooms; report SPL/success on a *held-out* set.
For A4 we report baselines (random floor, oracle ceiling, untrained LLM) so A5's
RFT has a number to beat."""

from __future__ import annotations
from .environment import SceneSearchEnv, EnvConfig
from .policies import run_episode
from .metrics import summarize


def evaluate(policy, scenes, config: EnvConfig | None = None) -> dict:
    """Run `policy` once per scene; return summary metrics + per-episode results."""
    config = config or EnvConfig(max_steps=400)
    results = [run_episode(SceneSearchEnv(s, config=config), policy) for s in scenes]
    summary = summarize(results)
    summary["n"] = len(results)
    return {"summary": summary, "results": results}


def _main():
    import argparse
    from .scene_gen import make_split
    from .policies import RandomPolicy, OraclePolicy

    ap = argparse.ArgumentParser(description="Held-out SPL eval on generated rooms.")
    ap.add_argument("--test", type=int, default=6, help="held-out scene count")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--llm", action="store_true", help="also eval the Fireworks LLM (slow)")
    ap.add_argument("--llm-max-steps", type=int, default=60)
    args = ap.parse_args()

    _, test = make_split(0, args.test, seed=args.seed)
    cfg = EnvConfig(max_steps=args.max_steps)

    print(f"Held-out eval on {len(test)} unseen rooms (seed {args.seed}):")
    print(f"  {'random':10s}", evaluate(RandomPolicy(seed=0), test, cfg)["summary"])
    print(f"  {'oracle':10s}", evaluate(OraclePolicy(), test, cfg)["summary"])
    if args.llm:
        from .llm_policy import LLMPolicy
        llm_cfg = EnvConfig(max_steps=args.llm_max_steps)
        print(f"  {'llm':10s}", evaluate(LLMPolicy(), test, llm_cfg)["summary"], "(untrained baseline)")


if __name__ == "__main__":
    _main()
