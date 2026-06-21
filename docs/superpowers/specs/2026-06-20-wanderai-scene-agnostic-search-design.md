# WanderAI — Scene-Agnostic Object Search (Design Spec)

**Date:** 2026-06-20
**Context:** HUD Frontier/RSI RL Environments Hackathon. 2-person, ~24h. Idea from Antim Labs.
**Goal (this prototype):** A single 3D room with multiple objects and one goal object (a red ball). An egocentric agent navigates continuous 2D space to *reach* the ball. Reward is a deterministic, mathematical function of the agent's geodesic (around-obstacle) distance to the goal plus a speed incentive. Antim Labs renders the scene; Fireworks fine-tunes the image→action policy. Long-term goal: a **scene-agnostic** search policy that generalizes to unseen rooms.

## Locked decisions

- **Movement:** continuous 2D on the floor plane (`x, y, heading`). Scene is 3D; navigation is on the floor.
- **Success condition:** proximity — agent within threshold `τ` of the ball.
- **Obstacles:** block both movement (can't pass through) and sight (occlude the ball in the rendered image).
- **Observation:** egocentric 2D RGB image only. The agent never sees coordinates or distance.
- **Reward:** privileged — the environment computes it from ground-truth positions the agent cannot see.

## Architecture (units, each independently testable)

1. **Scene** — holds floor bounds, obstacle footprints (2D polygons/AABBs projected from 3D), agent radius `r_a`, ball position `g`, agent start pose. Source-agnostic: produced by a scene generator (Antim Labs) or a hand-authored fixture.
2. **OccupancyGrid** — rasterizes the floor to cells (e.g. 5 cm). Marks a cell blocked if it overlaps any obstacle footprint inflated by `r_a` (configuration-space expansion). Pure function of Scene.
3. **DistanceField** — wavefront (Dijkstra/BFS) from the ball's cell across free cells → `D(cell)` = shortest walkable distance to the ball for every reachable free cell (`∞` if unreachable). Computed **once per episode**. Query `D(p)` for continuous `p` via bilinear interpolation. Also yields `D(start)` = optimal start→goal distance (used by reward and SPL).
4. **Renderer (interface)** — `render(scene, pose) -> RGB image`. Two implementations: `StubRenderer` (synthetic placeholder, no external deps, for tests/CI) and `AntimRenderer` (real). The environment depends only on the interface.
5. **Environment** (gym-style: `reset()`, `step(action)`) — owns Scene + DistanceField + Renderer. Applies actions with collision handling, computes reward, checks termination, emits observations. This is the RL environment deliverable.
6. **Metrics/Eval** — per-episode success, steps-to-goal, and **SPL** (success weighted by path length). Eval runs a policy over held-out scenes.
7. **Policy (test harness)** — a random / simple scripted policy to exercise the env end-to-end before any model is attached. Fireworks fine-tuning plugs in here later.

## Action space

Discrete, low-cardinality (easy for an LLM-style policy to emit): `MOVE_FORWARD` (fixed step `Δ`), `TURN_LEFT` (fixed angle `θ`), `TURN_RIGHT`. `MOVE_FORWARD` into a blocked cell is rejected (pose unchanged) and flagged as a collision for the reward.

## Reward function

Per step `t`, agent at `p_t`, static ball `g`, `d_t = D(p_t)`:

```
r_t = α·(d_{t-1} − d_t)        # geodesic progress (closer ⇒ positive); potential-based shaping
      − β                       # per-step time penalty ⇒ rewards speed
      − κ·𝟙[collision_t]         # tried to enter a blocked cell
      + R·𝟙[d_t ≤ τ]           # terminal success bonus (proximity)
```

Defaults: `α=1.0`, `β=0.02`, `κ=0.1`, `R=10`, `τ≈0.3 m`, step `Δ≈0.25 m`, turn `θ=30°`, `T_max≈200`. Episode ends on success or `T_max`.

**Multiple objects in the way:** the wavefront flows around *all* obstacles simultaneously, so `D` always encodes the true shortest walkable route regardless of object count. The reward formula is unchanged; only the precomputed field differs. Euclidean distance is explicitly rejected — it ignores obstacles and creates a reward trap at every object between agent and ball.

**Rigor notes:**
- The progress term is potential-based reward shaping with `Φ(p) = −D(p)`. By Ng–Harada–Russell (1999) it does not change the optimal policy vs. the sparse goal reward; it only densifies the gradient.
- Occlusion lives entirely in the Renderer (a hidden ball is simply absent from the image until the agent rounds the obstacle). It makes the task hard but never enters the reward — clean separation.

## Evaluation

Headline metric: **SPL** = `(1/N) Σ_i S_i · ℓ_i / max(p_i, ℓ_i)`, where `ℓ_i = D(start)` (optimal), `p_i` = path length walked, `S_i` = success indicator. Also report success rate and mean steps-to-goal. **Scene-agnostic test:** train on N rooms, evaluate SPL on held-out rooms.

## Scope / YAGNI

**In scope now:** Scene + fixtures, OccupancyGrid, DistanceField, StubRenderer, Environment, reward, SPL eval, random-policy harness, tests. This is a fully runnable, verifiable RL environment with no external dependencies.

**Out of scope now (interfaces left clean):** AntimRenderer integration, Fireworks fine-tuning, agent memory of visited regions, multi-goal / colored-distractor variants. Each plugs into an existing seam.

## Risks

- **Antim render latency/availability** during the event — mitigated by StubRenderer so the core never blocks on it.
- **Geodesic field resolution** vs. compute — tune cell size; field is per-episode O(cells).
- **Privileged dense shaping vs. generalization** — dense `D`-shaping could over-guide; if held-out SPL is weak, anneal shaping toward sparse or add visit-count exploration. Noted, not built yet.
