# WanderAI — Handoff

_Last updated: 2026-06-20 · HUD Frontier/RSI RL Environments Hackathon_

## TL;DR

WanderAI is a **scene-agnostic egocentric object-search RL environment**: an agent navigates
continuous 2D space using only first-person RGB images and must **find the red ball**. The bet:
train on a handful of rooms, generalize to unseen ones. Targets the **Robotics / Physical-AI (VLA)**
angle, delivered as a **verifiable RL environment** (geodesic reward + SPL metric).

The **core environment is fully built, tested, and pushed.** The remaining work is wiring the three
sponsor tools into the clean seams we left.

## Status snapshot

| | State |
|---|---|
| Core env (`wanderai/`) | ✅ Done — **44 tests pass** |
| Branch | `feat/scene-search-env` |
| Baselines | Oracle **SPL 1.0** / 22 steps · Random SPL 0.23 / 176 steps |
| **Goal** | **RFT / RLVR for a scene-agnostic policy** (not SFT). Hybrid: symbolic-text first, vision stretch |
| A1 symbolic observation | ✅ Done (`observation.py`) |
| A2 procedural scenes + split | ✅ Done (`scene_gen.py`) |
| A3 env text-obs mode | ✅ Done (`obs_text` in info) |
| Browser visualizer | ✅ Done (`serve.py` + `ui/index.html`) — `python serve.py` |
| Antim/Gizmo client + MJCF importer | ✅ Done (`antim.py`, `antim_import.py`) — UI "Load MJCF". Gizmo gen is slow/stalls; flow is pre-generate+cache+import |
| A4 LLM text policy + held-out eval | ✅ Done (`llm_policy.py`, `evaluate.py`) — Fireworks `gpt-oss-20b`, "Run LLM" in UI. Untrained baseline wanders |
| A5 RFT reward + GRPO signal | ✅ Built+verified (`rft.py`, `scripts/grpo_preview.py`). Live signal: advantages spread 2.41 |
| Visited-areas episodic memory | ✅ In the observation (in-context per-room exploration map) |
| A5 RFT training launch | ✅ Pipeline runs end-to-end (dataset+evaluator+`scripts/launch_rft.py`); model `llama-v3p1-8b-instruct`. **Blocked only on Fireworks payment method** (`400 payment method is required`, even with $501 credits). Add a card → `python scripts/launch_rft.py` |
| MuJoCo renderer (vision) | ⬜ Phase B stretch |

## The pipeline (and where each sponsor tool plugs in)

```
Antim (Gizmo) ──gen──> MuJoCo ──render──> SceneSearchEnv ──rollouts──> HUD ──data──> Fireworks SFT policy
   scene variety        RGB frames         reward + SPL                trace/eval      vision→action student
```

| Service | Role | Seam it plugs into |
|---|---|---|
| **Antim Labs** | Generates varied 3D scenes (the "10 houses → unseen rooms" engine) | scene source feeding the renderer |
| **MuJoCo** | Renders egocentric RGB from a pose | `Renderer` interface (replaces `StubRenderer`) |
| **HUD** | Hosts rollouts, logs rewards/traces, GRPO grouping | wraps `SceneSearchEnv` as an MCP env |
| **Fireworks** | Fine-tuned Qwen-VL = the learned policy | `policy.act()` (replaces `RandomPolicy`) |

## Core concepts (read before touching the code)

- **Geodesic distance field** (`distance_field.py`) — a map `D(x,y)` = shortest *walkable* distance
  from any point to the ball, routed **around** obstacles (Dijkstra wavefront from the ball, computed
  once per episode). This is the single source of truth. Straight-line distance is deliberately
  rejected — it creates a reward trap at every object between agent and ball.
- **Privileged oracle** (`OraclePolicy`, `policies.py:17`) — a ~15-line *deterministic planner*
  (NOT a model, NOT trained). It reads the geodesic field and steps toward the downhill direction →
  optimal paths (SPL 1.0). "Privileged" = it reads `env.field`, the ground-truth map the real agent
  never sees. Not deployable (needs the map); used as a **teacher** and an **upper bound**.
- **The geodesic function has three jobs:** (1) **reward** shaping `α·(d_{t-1}−d_t)` in
  `environment.py` — matters for RFT; (2) the **oracle's brain** — picks teacher actions → SFT labels;
  (3) **eval** — `D(start)` is the optimal path length, the numerator of **SPL**.
- **SFT (the planned training path) has no reward in the loop.** Roll out the oracle → record
  `(camera image, oracle action)` pairs → train the Qwen-VL student with cross-entropy to copy the
  action from pixels alone. The geodesic field makes the *labels*; it never touches the student's
  gradients. (Pattern = "Learning by Cheating", Chen et al. CoRL 2019.)
- **RFT** would instead train the student directly on the geodesic reward — but Fireworks RFT looks
  **text-policy-only** (no vision-policy training found), so it's future-work, not the demo path.

## Run it

```bash
pip install -e .            # numpy
pip install pytest mujoco   # pytest now; mujoco for the next task
pytest -q                   # 25 tests
python -m scripts.run_episode   # prints oracle vs random metrics
```

## Repo layout

```
wanderai/
  geometry.py        Pose, AABB, segment–box intersection, angle wrap
  scene.py           Scene (floor, obstacles, ball, start) + default_scene()
  occupancy.py       Rasterize scene → occupancy grid (cell-RECT overlap, not center-sample)
  distance_field.py  Geodesic field (wavefront Dijkstra) + bilinear query  ← the heart
  renderer.py        Renderer ABC + StubRenderer (FOV + occlusion)         ← MuJoCo goes here
  environment.py     SceneSearchEnv: reset/step, Action, EnvConfig, reward
  metrics.py         EpisodeResult, spl(), summarize()
  policies.py        RandomPolicy, OraclePolicy, run_episode()
scripts/run_episode.py   demo runner
tests/                   one test file per module
docs/superpowers/specs/  design spec
docs/superpowers/plans/  implementation plan (task-by-task)
```

## Key decisions & rationale

1. **Antim is a scene GENERATOR, not a renderer.** Gizmo's REST API does prompt→3D-scene
   (export USD/MJCF/SDF) — there is no `render(scene,pose)→RGB` and no coordinate control. So Antim
   can't be our `Renderer`; it's the scene-variety engine. (Original "rendering sorted by Antim"
   assumption was wrong.)
2. **MuJoCo is the renderer.** `mujoco.Renderer(model).render()` returns the `HxWx3 uint8` our
   `Renderer` interface already expects, with full pose control. Decision locked.
3. **SFT-distill the oracle** is the policy path (fast, reliable, clean narrative). RFT deferred.
4. **HUD v6 is protocol-based, not gym.** No reset/step interface — wrap our env as a `FastMCP`
   server (`@server.tool` for reset/step) exposed as an `mcp` capability on a `hud.Environment`,
   plus an `@env.template()` that yields a 0–1 reward.

## Integration cheatsheet (from API research)

**HUD** — `pip install hud-python` (import `hud`, Python 3.11/3.12). Env var `HUD_API_KEY` (`sk-hud-…`).
Build `hud.Environment` + `@env.template()` + an `mcp` capability wrapping `reset()`/`step()`.
CLI: `hud eval tasks.py <agent>`. Rewards must be normalized to **0.0–1.0**. Docs: docs.hud.ai (v6).

**Fireworks** — model `accounts/fireworks/models/qwen3-vl-8b-instruct` (vision + tunable, free <16B).
Env var `FIREWORKS_API_KEY` (`fw_…`). SFT data = JSONL chat, images as **base64 data URLs** (remote
URLs NOT allowed for training). Launch: `firectl dataset create …` then `firectl sftj create
--base-model … --dataset … --output-model …`. Serve on a dedicated GPU (LoRA can't go serverless),
call via OpenAI-compatible endpoint `https://api.fireworks.ai/inference/v1`. Docs: docs.fireworks.ai.

**Antim Labs (Gizmo)** — REST `https://api.gizmo.antimlabs.com/v1`, Bearer `gzm_k1_…`. `POST /v1/scenes`
(prompt → job), poll `GET /v1/jobs/{id}`, `POST /v1/scenes/{id}/export` (`mjcf`/`usd`/`sdf`; **SDF is
the only export with camera sensors**). No Python SDK, access gated — email viswajit@antimlabs.com /
Discord for a key + credits on-site.

## Next steps (prioritized) — RFT path

1. **A4 · LLM text policy + held-out eval** — `LLMPolicy.act` reads `info["obs_text"]`, calls a
   Fireworks text model (OpenAI-compatible endpoint), parses one of `{MOVE_FORWARD, TURN_LEFT,
   TURN_RIGHT}`. Eval SPL/success on held-out scenes from `make_split`. Establish baselines:
   random (floor), oracle (ceiling), untrained model. **Demo lands here** even pre-training.
2. **A5 · RFT loop** — wrap the env for Fireworks RFT (Eval Protocol `RemoteRolloutProcessor` /
   `firectl rftj create`) or HUD GRPO rollouts; reward = normalized geodesic return (RLVR). Train,
   then show **held-out SPL rising** — the scene-agnostic result.
3. **HUD wrapper** — expose `SceneSearchEnv` as an MCP environment (`@server.tool` reset/step,
   `@env.template` yielding 0–1 reward) for traces + grouped rollouts.
4. **Antim scene variety** — optionally feed Gizmo-generated rooms in alongside `scene_gen`.
5. **Phase B (vision stretch)** — MuJoCo renderer behind the `Renderer` seam + a VLM policy, with
   self-hosted RL (verl/TRL + GRPO on Modal), since Fireworks managed RFT looks text-only.
6. **(Stretch) agent memory** — visited-cell memory in the observation for smarter search.

Add `LLMPolicy` to the visualizer ("Run LLM" button) once A4 lands — it already drives
oracle/random through `/api/policy_action`.

## Gotchas

- **Secrets:** real keys live in gitignored `.env` (`thoughts.txt` also gitignored). `.env.example`
  is the committed template. **Rotate both keys after the event** — they were shared in plaintext.
- **HUD reward range is 0–1** — our raw rewards/SPL must be normalized before yielding to HUD.
- **Fireworks training images must be base64**, not URLs. Serving a LoRA needs a dedicated GPU
  (~$7/hr, scale-to-zero) — tear it down between sessions.
- **Occupancy uses cell-rectangle overlap, not center-sampling** (`occupancy.py:38`) — don't
  "simplify" it back to center-sampling or thin walls leak (a real bug we already fixed).

## Pointers

- Design spec: `docs/superpowers/specs/2026-06-20-wanderai-scene-agnostic-search-design.md`
- Build plan: `docs/superpowers/plans/2026-06-20-scene-search-env.md`
- Remote: https://github.com/vamshinr/WanderAI (branch `feat/scene-search-env`)
