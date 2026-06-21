# WanderAI

A **scene-agnostic egocentric object-search** RL environment. An agent navigates
continuous 2D space using only egocentric RGB images and must **find the red ball**.
Built for the HUD Frontier/RSI RL Environments Hackathon (idea seeded by Antim Labs).

The long-term bet: train a search policy on a handful of rooms and have it generalize
to unseen ones — a small step toward scene-agnostic embodied behavior.

## The reward — why it works with obstacles in the way

The agent is rewarded by **geodesic distance** to the ball — the shortest *walkable*
path that routes **around** obstacles — not straight-line (Euclidean) distance.
Euclidean distance rewards walking straight at the ball, so the agent presses into
whatever object sits between it and the goal and stalls in a reward trap. The geodesic
field bends around every obstacle, so the reward gradient always points along a route
the agent can actually walk. Add more objects and nothing changes: the wavefront flows
around all of them at once.

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
- The reward is **privileged**: the environment computes `d_t` from ground-truth
  positions, but the policy only ever sees the egocentric image. Occlusion lives
  entirely in the renderer (a ball hidden behind a box is simply absent from the
  image until the agent rounds the corner).

## How distance is computed

1. Project 3D obstacle geometry onto the floor; inflate each footprint by the agent
   radius (configuration-space expansion).
2. Rasterize the floor into an occupancy grid (cell-rectangle overlap, so thin walls
   are never missed).
3. Run a wavefront (Dijkstra) outward from the ball across free cells → a
   distance-to-goal field, computed **once per episode**.
4. Each step looks up `d_t` by bilinear interpolation. O(1).

## Architecture

| Module | Responsibility |
|---|---|
| `wanderai/geometry.py` | Pose, AABB, segment–box intersection, angle wrap |
| `wanderai/scene.py` | `Scene` (floor, obstacles, ball, start) + `default_scene()` |
| `wanderai/occupancy.py` | Rasterize a scene into an occupancy grid |
| `wanderai/distance_field.py` | Geodesic distance field (wavefront + bilinear query) |
| `wanderai/renderer.py` | `Renderer` interface + `StubRenderer` (FOV + occlusion) |
| `wanderai/environment.py` | `SceneSearchEnv` (gym-style `reset`/`step`, reward) |
| `wanderai/metrics.py` | SPL (Success weighted by Path Length) + summaries |
| `wanderai/policies.py` | `RandomPolicy`, privileged `OraclePolicy`, `run_episode` |

`StubRenderer` is a dependency-free stand-in so the environment runs in CI; the
**Antim Labs** renderer drops in behind the same `Renderer` interface. A learned
policy (e.g. **Fireworks** fine-tuning) replaces `RandomPolicy` at the same seam.

## Evaluation

Headline metric is **SPL** = `(1/N) Σ S_i · ℓ_i / max(p_i, ℓ_i)`, where `ℓ_i` is the
optimal geodesic distance from start to ball and `p_i` is the path the agent walked.
SPL = 1.0 is perfect shortest-path navigation. The **scene-agnostic test** is to
train on N rooms and report SPL on held-out rooms.

## Quickstart

```bash
pip install -e .          # numpy
pip install pytest        # tests
pytest -q                 # full suite
python -m scripts.run_episode
```

## Visualizer (browser UI)

```bash
python serve.py           # zero extra deps (stdlib only) → http://localhost:8000
```

Generate scenes (seeded or random), drive the agent with buttons / arrow keys, or
run the oracle / random policy. The canvas shows the geodesic heatmap (bright = close
to the ball) and the agent's FOV; the side panel shows live reward, efficiency, and
the **symbolic observation** the RFT text policy actually reads. The agent never sees
the map — only that text.

**Load an Antim/Gizmo (or any MuJoCo) scene:** click **Load MJCF file** and pick
`examples/gizmo_sample_room.xml` (or any MJCF) — the importer extracts the floor,
box obstacles, and red ball into a solvable `Scene` and the agent searches it.

## Antim Labs / Gizmo

`wanderai/antim.py` is a client for Gizmo's REST API (prompt → 3D scene → export
MJCF/USD/SDF); `wanderai/antim_import.py` parses an exported MJCF into our `Scene`.
Gizmo is a scene *generator*, not a renderer, and generation takes minutes, so the
intended flow is **pre-generate + cache + import**, not live. Set `GIZMO_API_KEY` in
`.env`. Example:

```python
from wanderai.antim import GizmoClient
from wanderai.antim_import import mjcf_zip_to_scene
path = GizmoClient().generate_export("a room with three boxes and a red ball", fmt="mjcf")
scene = mjcf_zip_to_scene(path)
```

Current baselines on `default_scene` (privileged oracle vs. random):

```
oracle {'success_rate': 1.0, 'spl': 1.0,  'mean_steps': 22}
random {'success_rate': 0.4, 'spl': 0.23, 'mean_steps': 176}
```

The oracle (geodesic descent) hits SPL 1.0, confirming the environment is solvable and
the distance field is correct; the wide oracle-vs-random gap is the room a learned
policy has to close.

## Roadmap

- Antim Labs renderer behind `Renderer` (real 3D egocentric frames).
- Fireworks fine-tuning of an image→action policy; eval on held-out scenes.
- Procedural scene generation (train on 10 rooms, test on unseen rooms).
- Agent memory of visited regions for efficient search.

See `docs/superpowers/specs/` and `docs/superpowers/plans/` for the design and build plan.
