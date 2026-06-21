# WanderAI Hackathon Feasibility and Winnability Report

Date: June 21, 2026

## Executive Summary

WanderAI is feasible for the HUD x YC Frontier RL Environments Hackathon as a v1 submission. The core environment is already implemented and tested: it has scene fixtures, occupancy rasterization, geodesic distance fields, egocentric stub rendering, dense privileged reward, SPL metrics, baseline policies, and a HUD v6 wrapper with MCP tools.

The project is winnable for a special-category or strong finalist-style submission if the team can show a live HUD trace and a crisp demo story. It is less likely to win the overall top prize unless the final pitch includes at least one of: real Antim rendering, a launched training run, model improvement evidence, or a larger held-out scene suite.

## Feasibility

Overall feasibility: **High**

Why it is feasible:
- The environment is deterministic, fast, and dependency-light.
- Reward is mathematically grounded in geodesic distance, avoiding Euclidean reward traps around obstacles.
- The agent receives only egocentric observations; privileged state is hidden and used only for reward.
- HUD v6 files are now present: `env.py`, `tasks.py`, `Dockerfile.hud`, and package dependencies.
- Local checks pass: `pytest`, HUD task listing, HUD task start, and HUD task grading.

Current implementation status:
- Simulator correctness: hardened against thin-wall tunneling, diagonal corner-cutting, invalid starts/goals, unreachable scenes, invalid actions, and post-done stepping.
- HUD integration: exposes `look`, `move_forward`, `turn_left`, `turn_right`, and `episode_status` via MCP.
- Evaluation: four deterministic task rows cover default, open, turning, and occluded scenes.
- Packaging: `Dockerfile.hud` serves `env:env` on port `8765`.

Remaining external blockers:
- `hud eval tasks.py claude ...` needs `ANTHROPIC_API_KEY` or a configured HUD gateway/model route.
- `hud deploy` needs `HUD_API_KEY`.
- Docker image verification was not run because local deploy credentials were missing; build should be checked once credentials/runtime are ready.

## Winnability

Overall winnability: **Medium**

Best-fit prize angles:
- **Best Design:** strong reward design, clean architecture, and clear simulator/renderer separation.
- **Most Creative:** embodied object search with occlusion and scene generalization is easy to explain.
- **Most Utopian:** credible path toward scene-agnostic physical AI behavior.
- **Overall finalist:** possible if paired with a live HUD trace and evidence that the task generates training signal.

Strengths for judging:
- Clear problem: find a red ball from egocentric observations in unseen rooms.
- Clear verifier: SPL and success are objective, repeatable, and hard to fake.
- Clean learning signal: geodesic reward gives progress without exposing coordinates to the agent.
- Strong demo contrast: oracle succeeds, random policy has lower SPL, leaving visible room for a learned policy.
- Good engineering shape: core simulator can run without Antim, while Antim can slot in behind `Renderer`.

Risks for judging:
- Stub renderer may read as less impressive than a real 3D Antim-rendered scene.
- No fine-tuned policy evidence yet.
- The taskset is small; “scene-agnostic” is currently a direction, not fully demonstrated.
- HUD deployment and model eval require credentials before final submission.

## Recommended Final-Hour Plan

Priority 1: unlock HUD traces.
- Set `HUD_API_KEY`.
- Set `ANTHROPIC_API_KEY` or route through HUD Gateway.
- Run:
  ```bash
  hud eval tasks.py claude --full --group 3 --max-steps 100 -y
  hud deploy
  hud sync tasks wanderai-scene-search
  ```

Priority 2: collect proof of signal.
- Run multiple rollouts per task and record reward spread.
- Show oracle vs random baseline.
- Capture one trace where the agent uses `look`, turns, moves, and gets graded.

Priority 3: strengthen the pitch.
- Lead with “Euclidean reward fails around obstacles; geodesic reward fixes it.”
- Explain that observations remain egocentric and reward remains privileged.
- Show the interface seam for Antim rendering and Fireworks training.

Priority 4: stretch only if the above is done.
- Add a simple generated-scene fixture function for more held-out rooms.
- Add real image payload support for the MCP `look` tool.
- Swap in Antim rendering if available without destabilizing the HUD wrapper.

## Bottom Line

WanderAI is submission-feasible now. To become highly competitive, the team needs live HUD traces and one clear training/eval story before judging. With those in place, it has a credible shot at design/creative/utopian recognition and an outside shot at top-10 finalist consideration.
