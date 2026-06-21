"""HUD-eval a NATIVE tool-calling model (our qwen3-4b multi-turn RFT models) on a
HUD task, driving the move() tool via real function calls.

The `hud eval ... openai_compatible` CLI routes the model string through a provider
parser (splits "accounts/..." → unknown provider). This bypasses that: it builds an
OpenAIChatConfig with an EXPLICIT Fireworks base_url + api_key (so the model id is
sent verbatim) and runs the stock OpenAIChatAgent (native tool calls).

  WANDER_TRAINED_MODEL='<model>#<deployment>' WANDER_TASK_SLUG=find-ball-3d-sym-0 \
  FIREWORKS_API_KEY=... HUD_API_KEY=... python run_native.py
"""
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

from hud import Taskset
from hud.agents.types import OpenAIChatConfig
from hud.agents.openai_compatible.agent import OpenAIChatAgent


async def main():
    model = os.environ["WANDER_TRAINED_MODEL"]
    cfg = OpenAIChatConfig(
        model=model,
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
    )
    for k, v in (("max_steps", int(os.environ.get("WANDER_MAX_STEPS", "80"))),):
        try:
            setattr(cfg, k, v)
        except Exception:
            pass
    try:
        # temp>0 to break the deterministic oscillation single-step-RFT models fall
        # into at greedy decoding; /no_think matches the model's training (it was
        # trained with /no_think so it acts directly instead of reasoning).
        cfg.completion_kwargs = {"max_tokens": int(os.environ.get("WANDER_MAX_TOKENS", "1024")),
                                 "temperature": float(os.environ.get("WANDER_TEMP", "0.7"))}
        cfg.system_prompt = "/no_think"
    except Exception:
        pass

    agent = OpenAIChatAgent(config=cfg)
    ts = Taskset.from_module("tasks.py")
    only = os.environ.get("WANDER_TASK_SLUG")
    kept = [t for slug, t in ts.items() if slug == only] if only else [t for _, t in ts.items()]
    if not kept:
        raise SystemExit(f"no task '{only}'; have {[s for s,_ in ts.items()]}")
    ts = Taskset(tasks=kept)
    print(f"running {len(kept)} task(s) with our model:\n  {model}\n")
    job = await ts.run(agent, group=1, max_concurrent=1)

    rewards = [float(getattr(r, "reward", 0.0) or 0.0) for r in (getattr(job, "runs", []) or [])]
    print("\n=== RESULT ===")
    print("  rewards:", rewards)
    print("  mean reward:", round(sum(rewards) / len(rewards), 3) if rewards else getattr(job, "reward", None))
    print("  job id:", getattr(job, "id", None))


if __name__ == "__main__":
    asyncio.run(main())
