---
title: WanderAI
emoji: 🔴
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

Team members:
Aditya Gouroju
Krish Garg
Rohith K
Vamshi Nagireddy

# WanderAI

**Scene-agnostic egocentric object-search.** An agent is dropped into a room it has
never seen and must **find a red ball** using only a first-person view — then
generalize that search behavior to new rooms. Built for the HUD Frontier/RSI RL
Environments Hackathon, using **Fireworks** (reinforcement fine-tuning + model
serving), **HUD** (standardized evaluation), and **Antim Labs / Gizmo** (3D scene
generation).

The agent never gets the map. Each step it sees a compact **symbolic observation**
— whether the ball is in line of sight (and its bearing/distance), the open space to
its left/center/right, which directions it has already explored — and picks one of
three actions: `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`. It only learns where the
ball is once it actually comes into view, so it has to **search**, not beeline.

---

## For judges — try it in the UI

**Prereq:** put `FIREWORKS_API_KEY=...` in `.env` (the trained models are served on
Fireworks; the app calls them server-side — the key never reaches the browser).

```bash
python3 serve.py
# → opens http://localhost:8000
# (auto-relaunches under the MuJoCo venv .venv-hud when you load a 3D scene)
```

### Test the 2D scene
1. Click **Default scene** (or **New scene** with a random seed for an unseen room).
2. Click **▶ Trained**.
3. Watch it search first-person: it turns to scan, advances into open space, and
   homes in once the ball enters its line of sight. On the map you'll see:
   - the **amber trail** — the exact path it walked;
   - **green cells** — ground it remembers visiting (it favors NEW ground);
   - the side panel — the live symbolic observation it reads (`Red ball: VISIBLE …`
     appears *only* when the ball is genuinely in view).

### Test the 3D test scene
1. Click **Load 3D Test scene** — a real Gizmo-exported room rendered in MuJoCo.
2. Top-right shows the agent's **first-person RGB + depth** camera; the canvas shows
   the top-down map.
3. Click **▶ Trained**.
4. Same honest search inside the 3D room — amber trail + green explored cells on the
   map as it navigates toward the ball.

### What to look for (and what makes it legit)
- It **explores** — it can't see the ball through walls, so it doesn't beeline.
- It **avoids re-treading** — green marks visited ground; it steers toward new area.
- It **never cheats** — the trained policy uses *only* what it perceives (clearance
  + the ball's bearing once visible). It does **not** read the geodesic field, the
  obstacle map, or the ball's hidden location.
- Compare the buttons: **Oracle** = privileged shortest-path (the ceiling, proves the
  room is solvable); **Random** = the floor; **Trained** = our model in between,
  searching honestly.

> Each step is a reasoning-model call, so the agent advances roughly one move every
> moment or two — watch the trail grow. Scenes allow up to 3000 steps.

---

## The reward — why it works with obstacles in the way

The agent is rewarded by **geodesic distance** to the ball — the shortest *walkable*
path that routes **around** obstacles — not straight-line (Euclidean) distance.
Euclidean distance rewards walking straight at the ball, so the agent presses into
whatever object sits between it and the goal and stalls in a reward trap. The geodesic
field bends around every obstacle, so the reward gradient always points along a route
the agent can actually walk.

Per step, with `d_t` = geodesic distance from the agent to the ball:

```
r_t = alpha * (d_{t-1} - d_t)     # geodesic progress  (closer => positive)
      - beta                       # per-step time penalty  -> rewards speed
      - kappa * 1[collision]       # bumped a blocked cell
      + R     * 1[success]         # reached within success_radius of the ball
```

- The progress term is **potential-based reward shaping** (`Phi = -d`); by
  Ng–Harada–Russell (1999) it does not change the optimal policy versus a sparse
  goal reward — it just makes learning tractable.
- The reward is **privileged**: the environment computes `d_t` from ground truth, but
  the policy only sees the egocentric observation. This is the train/eval signal —
  the deployed agent never sees it. Occlusion is real: a ball behind a box is simply
  absent from the observation until the agent rounds the corner.

## How distance is computed

1. Project obstacle geometry onto the floor; inflate each footprint by the agent
   radius (configuration-space expansion).
2. Rasterize into an occupancy grid (cell-rectangle overlap, so thin walls aren't missed).
3. Run a wavefront (Dijkstra) from the ball across free cells → a distance-to-goal
   field, once per episode.
4. Each step looks up `d_t` by bilinear interpolation. O(1).

## The trained models (RFT on Fireworks)

The policy is a small LLM fine-tuned with **reinforcement fine-tuning (RFT)** on
Fireworks: the model proposes actions, the env scores them with the geodesic reward,
and weights move toward higher return.

- **Single-turn RFT** (`wander-rft-*`): trained on one-step decisions (snapshot →
  best action). Works, but myopic — it can't see the consequences of a *sequence*, so
  it tends to oscillate.
- **Multi-turn / episodic RFT** (`wander-rft-2dmt-q`, `wander-rft-3dmt-q`, on
  `qwen3-4b`): trained on *whole episodes* via an MCP-Gym environment — the model
  drives the room turn-by-turn and is scored on the whole-episode return, so it learns
  to actually search and reach rather than dither. These are the models the UI's
  **Trained** button uses, **scene-aware**: the 2D model for 2D rooms, the 3D model
  for MuJoCo rooms.

A note on the 3D scene: the policy reads the *same symbolic observation* as 2D
(computed from the room's geometry), so a 3D room is no different to the policy — the
RGB+depth view is shown for the demo. (A "decode the observation from rendered pixels"
vision mode also exists but is noisier; the UI uses the clean geometric observation.)

## Honest navigation — no oracle, no cheating

The UI's Trained policy (`GuidedLLMPolicy`) drives the agent from the observation
alone. The only assist is **clearance-based obstacle avoidance** — built entirely from
information already in the observation (ray-cast clearance, recent moves, NEW/explored
flags): don't walk into a wall, don't spin forever in place, and prefer unexplored
ground over re-treading. It **never** reads the geodesic field, the obstacle map, the
oracle, or the ball's location while it's out of sight. (Earlier a geodesic
"safety-net" was wired in for demos; it was removed — that was privileged information
the agent shouldn't have.)

## Evaluation — SPL and a HUD leaderboard

Headline metric is **SPL** = `(1/N) Σ S_i · ℓ_i / max(p_i, ℓ_i)` (1.0 = perfect
shortest-path navigation). The scene-agnostic test: train on N rooms, report on
held-out rooms.

The environment is also wrapped as a **HUD** environment with an eval suite
(`wander-hud/run_eval_suite.py`) that runs several agents over the *same* held-out
rooms and produces one leaderboard (`docs/eval_report.md`):

```
oracle (upper bound)  ~1.0     ← proves the rooms are solvable
ours  (RFT model)     in between, searching honestly
random (floor)        ~chance
```

### Multi-turn RFT — trained ≥1 epoch and evaluated end-to-end on HUD

Both episodic models train via the `wander_lake/` McpGym env (model drives `move()`
turn-by-turn; GRPO scores the whole-episode return) and are then run through the HUD
env via native tool-calls (`wander-hud/run_native.py`, model served on Fireworks):

| Model (qwen3-4b, 1 epoch) | Train score | HUD eval (end-to-end) |
|---|---|---|
| `wander-rft-3dmt-q` (3D MuJoCo scene) | **0.861** | **reached the ball — reward 1.0** (seed 0), 0.556 (seed 1) |
| `wander-rft-2dmt-q` (2D procedural) | 0.617 | runs end-to-end; weak on held-out (quick 6-room/1-epoch run) |

The HUD harness itself is validated independently (Claude scores **1.0 / 100%** on the
same 3D task). Getting episodic RFT to run took fixing four real bugs — a
non-idempotent dataset adapter, an all-zero reward (rollout step-cap below the env's),
a base model that couldn't tool-call (→ `qwen3-4b`), and a fixed-port rollout-server
collision (→ `server_mode="shared"`) — see [HANDOFF.md](HANDOFF.md) / commit history.

**Reproduce** (datasets are 6 rooms with `/no_think` so qwen3-4b runs full 30-step
episodes; 2D and 3D share the evaluator name → launch **sequentially**):

```bash
# 1. Train (each is a 1-epoch episodic RFT job on Fireworks; base must be qwen3-4b)
export WANDER_BASE_MODEL=accounts/fireworks/models/qwen3-4b WANDER_EPOCHS=1 WANDER_MAX_TOKENS=1024
WANDER_OUTPUT_MODEL=accounts/<acct>/models/wander-rft-2dmt-q bash scripts/launch_rft_v4.sh           # 2D
WANDER_OUTPUT_MODEL=accounts/<acct>/models/wander-rft-3dmt-q \
  WANDER_DATASET_JSONL=wander_lake/data/wander_dataset_3d.jsonl bash scripts/launch_rft_v4.sh        # 3D

# 2. Deploy (qwen3 is NOT supported on A100, and H100 was capacity-exhausted → H200)
WANDER_ACCELERATOR=NVIDIA_H200_141GB python scripts/deploy_trained.py wander-rft-3dmt-q

# 3. HUD-eval our model end-to-end (native tool-calls; the `hud eval openai_compatible`
#    CLI mis-routes "accounts/..." as a provider, so use run_native.py)
cd wander-hud && WANDER_TRAINED_MODEL='<model>#<deployment>' \
  WANDER_TASK_SLUG=find-ball-3d-sym-0 python run_native.py     # → reward 1.0 (reached the ball)

# Teardown the H200 deployments (they bill per hour):
WANDER_ACCELERATOR=NVIDIA_H200_141GB python scripts/deploy_trained.py wander-rft-3dmt-q --delete
```

The 3D model trains/evals on the real Gizmo MuJoCo scene's **geometry** (symbolic obs):
Fireworks' rollout sandbox is headless, so vision is decoded only in the local UI/HUD,
which can render. The matching HUD task is `find-ball-3d-sym-*` (`wander-hud/tasks.py`).

## Architecture

| Module | Responsibility |
|---|---|
| `wanderai/geometry.py` | Pose, AABB, segment–box intersection, angle wrap |
| `wanderai/scene.py` / `scene_gen.py` | `Scene` + procedural room generation |
| `wanderai/occupancy.py` | Rasterize a scene into an occupancy grid |
| `wanderai/distance_field.py` | Geodesic distance field (wavefront + bilinear query) |
| `wanderai/observation.py` | Egocentric **symbolic** observation + visited-areas memory |
| `wanderai/perception.py` / `mujoco_renderer.py` | 3D RGB+depth view + (optional) decode |
| `wanderai/environment.py` | `SceneSearchEnv` (gym-style `reset`/`step`, reward) |
| `wanderai/policies.py` | `RandomPolicy`, privileged `OraclePolicy`, `run_episode` |
| `wanderai/llm_policy.py` | `LLMPolicy` (Fireworks) + honest `GuidedLLMPolicy` |
| `serve.py` + `ui/index.html` | Zero-dependency browser visualizer |
| `wander-hud/` | HUD environment + eval suite; `run_native.py` HUD-evals our tool-calling models |
| `wander_lake/` | Multi-turn (episodic) RFT env (eval-protocol McpGym); `scene_3d` loads a real 3D scene |
| `scripts/launch_rft_v4.sh` / `deploy_trained.py` | Launch episodic RFT (qwen3-4b) / deploy a model (any accelerator) |

## The visualizer (`serve.py`)

Zero extra dependencies (stdlib only). Generate scenes (seeded/random), drive the
agent with buttons / arrow keys, or run **Oracle / Random / LLM (base) / Trained**.
The canvas shows the geodesic heatmap, obstacles, ball, the agent's FOV, the **amber
traversal trail**, and **green explored cells**; the side panel shows the live
symbolic observation. For 3D scenes it also shows the first-person RGB + depth view.
`python3 serve.py` auto-relaunches under `.venv-hud` (Python 3.12 + MuJoCo) when 3D is
needed, so the one command serves both 2D and 3D.

## Antim Labs / Gizmo

`wanderai/antim.py` is a client for Gizmo's REST API (prompt → 3D scene → export
MJCF); `wanderai/antim_import.py` parses an exported MJCF into our `Scene`. Gizmo is a
scene *generator*, so the flow is **pre-generate + cache + import**. Set `GIZMO_API_KEY`
in `.env`. The 3D test/train rooms in `examples/*.xml` are real Gizmo exports.

## Status

- ✅ Geodesic environment, symbolic observation, visited-areas memory, SPL eval.
- ✅ Single-turn and **multi-turn (episodic) RFT** models trained (≥1 epoch) + served on Fireworks.
- ✅ **Multi-turn 3D model evaluated end-to-end on HUD — reached the ball (reward 1.0).**
- ✅ HUD environment + eval-suite leaderboard (random < ours < oracle).
- ✅ Browser UI with honest navigation, traversal trail, and explored-cells overlay,
  for both 2D and 3D rooms.
- ✅ Antim/Gizmo 3D scene import (MuJoCo RGB+depth view).

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for the design and build plan.
