"""Run our model on the HUD scene-search env via the bridge agent (programmatic,
so no interactive TUI). Produces a real HUD trace + reward for OUR model.

  WANDER_TRAINED_MODEL='accounts/.../models/wander-rft-vN#accounts/.../deployments/<id>' \\
  FIREWORKS_API_KEY=... HUD_API_KEY=... python run_bridge.py
"""
import asyncio
import os

from hud import Taskset
from hud.agents.types import OpenAIChatConfig

from bridge_agent import WanderBridgeAgent


async def main():
    model = os.environ["WANDER_TRAINED_MODEL"]
    cfg = OpenAIChatConfig(
        model=model,
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url="https://api.fireworks.ai/inference/v1",
    )
    agent = WanderBridgeAgent(config=cfg, wander_model=model)

    ts = Taskset.from_module("tasks.py")
    print(f"running {len(list(ts.items()))} task(s) with our model:\n  {model}\n")
    job = await ts.run(agent, group=1, max_concurrent=1)

    print("\n=== RESULT ===")
    for attr in ("name", "id", "url", "reward", "mean_reward", "success_rate"):
        if hasattr(job, attr):
            print(f"  {attr}: {getattr(job, attr)}")
    print("  (full job object):", repr(job)[:300])


if __name__ == "__main__":
    asyncio.run(main())
