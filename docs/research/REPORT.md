# WanderAI: Scene-Agnostic Object-Goal Search with Language-Model Policies, Geodesic Reward Shaping, and Bounded-Memory Mapping

**Technical report — 2026-07-13.** All quantitative claims in this report are
produced by scripts in this repository (`scripts/run_benchmark.py`,
`scripts/run_benchmark_3d.py`, `scripts/run_gizmo_transfer.py`,
`scripts/train_local_rl.py`) with fixed seeds; the raw outputs, including git
revision and configuration metadata, are versioned under
`docs/research/data/`. The companion literature review with verified citations
is `docs/research/literature_review.md`.

---

## 1. Problem statement

An embodied agent is placed at a random pose in a room it has never seen and
must reach a goal object (a red ball) using only egocentric sensing. The
research question is **scene-agnostic generalization**: does a policy trained
on a small set of procedurally generated rooms transfer its *search behavior*
— not a memorized route — to rooms it has never observed?

This is an instance of object-goal navigation (ObjectNav). WanderAI's variant
deliberately differs from the standard Habitat formulation in three ways:

1. **A single, visually distinctive goal object.** The red ball removes the
   semantic-prior sub-problem ("TVs are near sofas") and isolates the *search
   and control* sub-problem. A consequence, confirmed by our benchmarks, is
   that geometric frontier exploration is a much stronger baseline here than
   in semantic ObjectNav suites — so any learned-policy claim must beat, or
   explain its position against, classical exploration rather than only a
   random floor.
2. **A compact symbolic-text observation as the primary interface**, so that
   small language models can be trained as policies with episodic
   reinforcement fine-tuning (RFT). A pixels-only mode (rendered RGB-D)
   exercises the same task with real perception.
3. **A privileged geodesic training signal with an honest deployment
   interface.** The environment computes rewards from ground truth the policy
   can never read; every deployable policy in this repo senses the world only
   through its observation channels (Section 3.4).

## 2. Task formalization

The task is a POMDP. The state is the agent pose $(x, y, \theta)$ in a scene
$S$ = (floor bounds, obstacle footprints, ball position $g$); the agent's
actions are `MOVE_FORWARD` (0.25 m, rejected on collision), `TURN_LEFT`,
`TURN_RIGHT` (30°). An episode succeeds when the agent is within 0.3 m
(Euclidean) of the ball; episodes cap at $T_{max}$ steps (400 in the 2D/3D
benchmarks, 1500 in the 20×15 m transfer room).

**Observation.** The symbolic observation is a text rendering of: ball
visibility (with bearing/distance only while a clear line of sight exists
within a 90° FOV and 8 m range — occlusion is real), left/center/right
ray-cast clearance, per-direction explored/NEW flags from a visited-cell
memory, and the last four actions. The vision mode replaces privileged
geometry with rendered pixels: ball detection by color segmentation on RGB,
ranges and mapping from the depth buffer (Section 6).

**Sensors available to honest policies.** Pose (the GPS+Compass sensor that is
part of the standard ObjectNav specification), the observation above, and — for
map-building policies — a 1-D depth strip across the FOV (a simulated range
scan in 2D; decoded from the real MuJoCo depth image in 3D). No policy reads
the environment's occupancy grid, geodesic field, or the ball's position while
out of view. The `FrontierPolicy` test suite includes a guard test that
replaces `env.field` with an object that raises on any attribute access.

## 3. Environment

### 3.1 Geodesic distance field

Obstacle footprints, inflated by the agent radius (configuration-space
expansion), are rasterized to a 0.1 m occupancy grid using cell-rectangle
overlap (so thin walls cannot be missed). A Dijkstra wavefront from the ball
cell yields $D(x,y)$: the shortest *walkable* distance from every free cell to
the goal, computed once per episode and queried by bilinear interpolation.

### 3.2 Reward

With $d_t = D(x_t, y_t)$:

$$r_t = \alpha\,(d_{t-1} - d_t)\; -\; \beta\; -\; \kappa\,\mathbb{1}[\text{collision}]\; +\; R\,\mathbb{1}[\text{success}]$$

($\alpha{=}1$, $\beta{=}0.02$, $\kappa{=}0.1$, $R{=}10$.) The progress term is
potential-based reward shaping with potential $\Phi = -D$; by the
Ng–Harada–Russell policy-invariance theorem it does not change the optimal
policy relative to the sparse goal reward — it densifies the gradient. The
geodesic (rather than Euclidean) potential is the essential choice: a Euclidean
potential creates a local optimum against every obstacle that lies between the
agent and the goal, while the geodesic field's gradient always points along a
walkable route. This mirrors the standard distance-to-goal reward used to
train PointNav/ObjectNav agents in Habitat, with the same slack penalty role
played by $\beta$.

**The reward is privileged; the policy is not.** $D$ is used for training
signal, for the oracle upper bound, and for evaluation ($D(\text{start})$ is
the SPL numerator). It is never an input to any deployable policy.

### 3.3 Procedural scenes and splits

`scene_gen.random_scene` samples rooms 5–8 m on a side with 2–4 axis-aligned
obstacles, resampling until the ball is reachable from the start
(wavefront-verified) and at least 2 m away. `make_split(n_train, n_test,
seed)` yields deterministic, disjoint train/test lists from one seed.

Two split-hygiene hazards surfaced during this work and are now guarded in
code: (i) *stream reuse* — `make_split(0, N, seed)` draws the same room stream
as `make_split(n_train, ·, seed)`, so a benchmark that reuses an RL training
seed evaluates the trained policy partly on its own training rooms (the
benchmark CLI now defaults to a disjoint seed and documents the hazard);
(ii) *pseudo-replication* — evaluating a deterministic policy twice per room
duplicates outcomes and narrows bootstrap confidence intervals by $\sqrt{2}$
(final tables use one episode per room across more rooms).

### 3.4 Matched 3D rooms

`scene_mjcf.scene_to_mjcf` compiles any procedural scene into a MuJoCo room —
floor plane at $z{=}0$, perimeter walls whose inner faces lie exactly on the
scene bounds, one box per obstacle with a deterministic pseudo-random height
in [0.45 m, 1.9 m] (some furniture sits below the camera horizon; some
occludes), and the emissive red ball injected by the same helper the
Gizmo-import path uses. Because the rendered geometry and the collision
substrate are *identical*, any behavioral gap between the geometry-sensing and
pixel-sensing runs of the same policy in the same rooms is attributable to
perception — not to world mismatch. This matched-world property is what makes
the vision benchmark interpretable, and it is scalable: every procedural room
is also a 3D room.

## 4. Policies

**Random** (floor) and **Oracle** (privileged ceiling: one-step lookahead
descent of the geodesic field). The oracle proves each room solvable and
bounds SPL; note it is *not* a perfect ceiling under a 30° discrete turn —
on one of 50 held-out rooms greedy descent oscillates and fails, which we
report rather than hide.

**Frontier-based exploration (FBE).** The classical zero-training baseline:
build a map from your own sensing, go to the boundary between known-free and
unknown space, repeat; once the ball enters view, servo on its bearing.
Selection is cost–utility (cluster size discounted by distance) with
nearest-frontier as an ablation. Modern zero-shot ObjectNav systems are
FBE backbones with semantic frontier scoring; with a semantically neutral
goal, plain FBE is the honest classical comparator. Implementation details
that mattered (each fixed a measured failure mode): string-pulled waypoint
following (steering at raw A* grid waypoints makes a 30°-turn agent zigzag
indefinitely), best-first replanning across frontier candidates (the top
cluster alone can be unplannable), sticky obstacle-skirting, collision-inferred
map updates, and goal-position memory (Section 6).

**Episodic RFT of a small LLM (Fireworks).** The repository's headline
training path: qwen3-4b drives whole episodes through an MCP-Gym interface
and is fine-tuned with GRPO on whole-episode return (`wander_lake/`,
`scripts/launch_rft_v4.sh`). This report does not re-run those cloud
trainings; their end-to-end results (e.g. the 3D multi-turn model reaching
the ball with reward 1.0 on the HUD harness) are documented in the README.
What this report adds is the *local, reproducible mirror* of that loop:

**Local episodic RL (`rl_local.py`).** A ~19-feature encoding of the symbolic
observation (ball visibility/bearing/distance, clearances, explored flags,
last action, optional map hint) feeds a 32-unit MLP trained with
group-relative REINFORCE — the GRPO advantage estimator (per-scene groups of
G=8 episodes; advantage = group-standardized episodic return) with Adam and an
entropy bonus, in pure numpy. It trains in minutes on CPU with no API keys, so
the claim "episodic RL on the geodesic reward produces transferable search
behavior" is verifiable by anyone. The deployed policy is argmax with the same
two unstuck reflexes the FBE follower uses (collision inference, anti-dither),
applied identically to trained and untrained weights so measured deltas are
attributable to learning. The untrained comparator is evaluated
*stochastically* — argmax over a zero-initialized head is a tie broken
constantly to `MOVE_FORWARD`, i.e. a drive-forward heuristic rather than the
uniform policy training starts from (an evaluation artifact we found and
fixed during review).

**Map-hint variant.** The agent maintains its own `EgoMap` (Section 5) and
appends a compact frontier hint (direction bucket + distance of the best
frontier cluster) to the feature vector — a bounded, in-context piece of map
memory of exactly the kind a language-model policy can consume as text. This
is the bridge between classical mapping and promptable policies: the same
hint string can be injected into the RFT model's observation.

**VLM policy (`vision_policy.py`).** An image-native policy interface:
egocentric RGB (stdlib PNG encoder, no PIL) plus an episodic-memory text
scaffold (recent moves, explored flags — deliberately *not* ball bearing) in
an OpenAI-format multimodal prompt against any compatible endpoint (default
Fireworks). Transport-injectable and fully unit-tested offline; running it
against a live VLM requires only `FIREWORKS_API_KEY`.

## 5. Bounded-memory mapping (`mapping.py`)

The agent's own map is a sparse log-odds occupancy grid (Moravec–Elfes
inverse sensor model): cells traversed by a range ray accumulate free
evidence ($\ell_{free}{=}-0.7$), the hit cell accumulates occupied evidence
($\ell_{occ}{=}+1.8$), clamped to $\pm 4$. Three implementation details are
load-bearing, each traced to a measured failure:

- **Hits are padded ~a third of a cell into the surface** and only the
  agent's own cell is exempt from occupied marks. Dropping sub-cell hits
  entirely (our first "don't wall yourself in" rule) left *near* walls
  unmapped: line-of-sight smoothing then aimed straight through real walls
  and the follower entered a two-step turn limit cycle.
- **Robot-footprint clearing:** the agent's cell is strongly re-freed each
  scan, and the planner treats the start cell's immediate ring as traversable
  — otherwise stray near-wall returns wall the robot into its own map.
- **A trust threshold for motion gating** (`TRUSTED_OCC`, asserted to lie
  strictly between one and two occupied hits): a single stray return may not
  freeze the robot; real walls (multiple hits, or one inferred collision at
  +4.0) must.

**Memory bound.** The map has a hard cell budget; on overflow it re-bins at
double the cell size, occupied evidence dominating merges so walls survive
coarsening. Long-horizon episodes degrade gracefully in map *resolution*
instead of growing without bound — the mechanism is ablated in the benchmarks
(Section 8.1: a 500-cell budget is free in 5–8 m rooms; 150 cells costs ~14
points of success rate).

**Frontiers and planning.** Frontier cells (known-free adjacent to unknown)
are clustered (8-connectivity); A* runs over the agent's own map with unknown
cells optimistically traversable at 1.5× cost, occupied goals snapped to the
nearest open cell.

**Vision-mode sensing.** In 3D the strip is *not* one depth row: a single
horizon band sees over knee-high furniture (measured: a 0.35 m-tall obstacle
3.3 m ahead read as free space). `obstacle_strip_from_depth` backprojects the
full depth image to (bearing, horizontal range, height-above-floor), keeps
points inside the agent's traversal band [0.10 m, 1.5 m] — floors and
overhead structure are not obstacles; low furniture is — and takes a robust
per-bearing-bin minimum. This is the height-thresholded point-cloud
projection modular ObjectNav systems use, reduced to a strip. Two camera-model
bugs were found by testing against ground truth and are worth recording: the
image-down axis has forward component $-\sin(\text{pitch})$ (a $+$ sign
overestimated ranges to below-horizon obstacles by up to ~21%), and the depth
buffer stores perpendicular $z$, requiring a $\cos(\text{bearing})$ correction
to yield range along the ray. After both fixes, strip fidelity against
un-inflated ray-cast ground truth at matched bearings is ~0.08 m MAE.

## 6. Goal-position memory and the vertical-FOV blind zone

A floor-level ball leaves a pitched camera's frame at close range: with eye
height 1.4 m, pitch 8°, and fov$_y \approx 74°$, the ball drops below the
bottom edge inside ~1.1 m — *before* the 0.3 m success radius. A purely
reactive vision policy therefore homes flawlessly until it goes blind, then
wanders off (we traced exactly this loop). The fix is the standard one from
map-based systems: while the ball is visible, keep a world-frame estimate of
its position (bearing + sensed range + own pose); when it drops out of frame
nearby, dead-reckon to the remembered point, discarding the belief if arrival
reveals nothing. This is honest (own perception + own odometry) and it is the
reason the pixels-only success rate is close to the geometry-mode rate rather
than near zero.

## 7. Planar-surface extraction (`planes.py`)

From one depth image: backproject to a world-frame point cloud (camera model
validated against analytic floors/walls to <0.02 m), estimate normals from
the organized point image, and run sequential RANSAC with signed
normal-agreement gating; classify each plane as **floor** (horizontal, at
floor height), **wall** (vertical), elevated horizontal **actionable
surface** (tabletop-like), or other. Floor-plane pixels project to a
navigable-space estimate — free space derived from pixels alone, with no
semantic labels, suitable as an alternative free-space evidence source for
the mapper and as the geometric substrate for future "which surfaces can I
act on" queries in 3D scenes.

Measured on 20 matched rooms (start-pose view): floor recovered where visible
(13/20 views) with height error ~2×10⁻⁷ m, wall verticality error ~0.001°,
and navigable-point precision 94% against raw obstacle footprints (validated
against *un-inflated* geometry — scoring against the configuration-space
`is_free` would count the 0.2 m inflation ring around furniture as false
positives, a harness artifact we corrected during review).

## 8. Experiments

*(Tables below are generated by the named scripts; JSON with git revision and
full config in `docs/research/data/`.)*

### 8.1 2D held-out generalization (50 unseen rooms, 1 episode each)

`python3 scripts/run_benchmark.py --rooms 50 --episodes 1 --seed 1234` —
the split seed is disjoint from the RL training stream (seed 7), max 400
steps, bootstrap 95% CIs over episodes.

| Policy | SR [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Steps (succ.) | Collisions |
|---|---|---|---|---|---|---|
| random | 0.120 [0.040, 0.220] | 0.043 [0.010, 0.090] | 0.074 | 3.85 | 246 | 51.7 |
| fbe-nearest | 0.720 [0.600, 0.840] | 0.522 [0.421, 0.629] | 0.531 | 1.34 | 44 | 0.5 |
| **fbe (cost–utility)** | **0.860 [0.760, 0.940]** | 0.613 [0.525, 0.701] | 0.592 | 0.67 | 49 | 0.4 |
| fbe, 2000-cell map budget | 0.860 [0.760, 0.940] | 0.613 [0.525, 0.701] | 0.592 | 0.67 | 49 | 0.4 |
| fbe, 500-cell map budget | 0.840 [0.740, 0.940] | 0.600 [0.508, 0.692] | 0.579 | 0.84 | 49 | 0.5 |
| fbe, 150-cell map budget | 0.720 [0.600, 0.840] | 0.543 [0.438, 0.652] | 0.559 | 1.14 | 43 | 0.4 |
| **rl-local (episodic RL, no map)** | 0.720 [0.600, 0.840] | **0.636 [0.516, 0.749]** | 0.627 | 1.64 | 80 | 0.5 |
| rl-local-hint (episodic RL + own-map hint) | 0.620 [0.500, 0.740] | 0.574 [0.449, 0.692] | **0.713** | 0.88 | 59 | 0.4 |
| oracle (privileged) | 0.980 [0.940, 1.000] | 0.966 [0.916, 0.999] | 0.921 | 0.03 | 26 | 0.0 |

Readings, stated conservatively:

- **Episodic RL on the geodesic reward produces genuinely transferable search
  behavior.** A 19-feature reactive policy trained on 12 rooms reaches SR
  0.72 on 50 disjoint unseen rooms — six times the random floor — and its
  SPL (0.636) is at least competitive with full map-building frontier
  exploration (0.613), i.e. when it succeeds it takes *more direct* paths
  than frontier-chasing. Its SR remains below FBE's 0.860 (CIs touch); a
  memoryless policy cannot systematically cover space, which is the expected
  and observed failure mode (per-room training SR is bimodal: near 1.0 on
  reactive-solvable rooms, near 0 on rooms requiring systematic search).
- **Frontier selection matters:** cost–utility beats nearest-frontier by 14
  points of SR — consistent with the exploration literature.
- **Bounded map memory degrades gracefully:** a 500-cell budget is
  essentially free in 5–8 m rooms; forcing coarsening at 150 cells costs ~14
  points of SR and leaves the policy still 6× the random floor. Memory can be
  traded smoothly for competence — the mechanism intended for long-horizon
  scaling.
- **The map hint helps progress-efficiency, not success, at this training
  budget:** the hint variant posts the best non-oracle SoftSPL (0.713) and a
  DTS half of the no-hint variant's, but lower SR — it commits toward
  frontiers and under-explores when the hint is stale between map updates.
  We report this honestly as a mixed result; the hint's value for a
  *language-model* policy (as in-context text) is the follow-up experiment,
  not a claim made here.

### 8.2 3D pixels-only navigation (20 matched rooms)

`MUJOCO_GL=osmesa python3 scripts/run_benchmark_3d.py --rooms 20 --seed 42` —
the same rooms rendered as MuJoCo worlds whose geometry equals the collision
substrate exactly; the same `FrontierPolicy` runs with symbolic ray sensing
(geometry) vs pixels only (RGB ball detection + height-aware depth mapping +
goal-position memory).

| Policy | SR [95% CI] | SPL [95% CI] | SoftSPL | DTS (m) | Collisions |
|---|---|---|---|---|---|
| fbe-geometry | 0.900 [0.750, 1.000] | 0.598 [0.464, 0.734] | 0.575 | 0.23 | 0.1 |
| **fbe-vision (pixels only)** | 0.700 [0.500, 0.900] | 0.532 [0.356, 0.709] | 0.534 | 1.29 | 2.8 |
| random | 0.050 [0.000, 0.150] | 0.039 [0.000, 0.116] | 0.079 | 3.86 | 49.2 |
| oracle (privileged) | 1.000 [1.000, 1.000] | 0.981 [0.944, 1.000] | 0.931 | 0.00 | 0.0 |

The **perception cost is 20 points of SR and 0.07 SPL** under matched worlds
— attributable to sensing alone: depth-strip fidelity is 0.14 m MAE against
un-inflated ray-cast ground truth at matched bearings, and the residual
failures trace to close-range detection loss and map noise near clutter.
Collisions rise from ~0 to 2.8/episode, the direct signature of sensed vs
exact clearance.

**Planar-surface extraction** on the same 20 start views: the floor plane is
recovered in every view where a sufficient floor region is visible (13/20)
with height error ~2×10⁻⁷ m; wall verticality error ~0.001°; floor-derived
navigable points have **98.0% precision** against raw obstacle footprints.
Pixels alone yield a near-exact actionable-space estimate in these worlds.

### 8.3 Transfer to a real Gizmo export (20×15 m room)

`MUJOCO_GL=osmesa python3 scripts/run_gizmo_transfer.py` — max 1500 steps.

| Policy | Outcome |
|---|---|
| oracle (privileged) | success, 87 steps (SPL term 1.0) |
| fbe-geometry (3 seeds) | success, 104 steps, SPL term 0.90, 0 collisions — all seeds |
| fbe-vision (3 seeds) | failure at 1500 steps — all seeds |
| random | failure at 1500 steps, 125 collisions |

The geometry-sensing policy transfers to a real exported room 6× larger than
any training-distribution room with near-oracle path efficiency. The
pixels-only policy does not — and the *measured reason* is world mismatch, not
perception: Gizmo exports carry furniture as position-only bodies (meshes
external), so the camera sees real clutter that does not exist in the
imported collision world. The start-pose depth strip disagrees with the
collision substrate by **2.93 m** mean absolute deviation (vs 0.14 m in
matched rooms — a 20× gap); the agent's map fills with visually-real,
collision-absent obstacles until no path exists. We report this as a property
of the export format with a clear remediation (full-mesh exports), and it is
precisely why the matched-room benchmark of Section 8.2 exists.

### 8.4 Local episodic training dynamics

`python3 scripts/train_local_rl.py --iters 400 --group 8` (curves in
`docs/research/data/rl_curve_*.json`): group-mean return rises from ≈ −1.3 to
+14.7 on solved rooms within 400 iterations (~minutes on CPU); entropy anneals
from 1.02 to ~0.15. Per-room returns are bimodal (see 8.1). The comparator
protocol evaluates untrained weights stochastically (the uniform policy
training actually starts from) after review found the argmax-of-zero-logits
"untrained" baseline was a deterministic drive-forward artifact.

## 9. Related work and positioning

<!-- RELATED:BEGIN -->
<!-- summary of docs/research/literature_review.md once the verified review lands -->
<!-- RELATED:END -->

## 10. Limitations

- **Simulation simplicity.** Rooms are single-space with axis-aligned box
  obstacles; there are no doorways, multi-room topologies, or clutter
  geometry beyond boxes. Claims are about *relative* policy behavior under
  matched conditions, not about absolute difficulty parity with HM3D/MP3D.
- **Gizmo exports are visually richer than their collision import.**
  Furniture arrives position-only (meshes external), so the pixels-only
  pipeline perceives real clutter the collision world lacks and walls itself
  in; we quantify the mismatch and treat the matched procedural rooms as the
  controlled vision benchmark (`docs/research/data/gizmo_transfer.json`).
  Full-mesh exports would close this gap.
- **The oracle is not a perfect ceiling** under 30° discrete turns (0.98 SR on
  the 50-room split); SPL normalization uses the true geodesic optimum, which
  is unaffected.
- **Local RL is a mirror, not the artifact.** The numpy trainer demonstrates
  the training loop's soundness reproducibly; the language-model RFT results
  on Fireworks are the project's deployment path and are documented in the
  README rather than re-run here.
- **Known deferred engineering:** costmap inflation of the agent's own map
  would subsume three separate near-wall safeguards; the FBE follower and the
  RL deployment wrapper share unstuck reflexes by convention rather than by a
  common helper; vision-mode stepping renders more frames than necessary.
  These are documented in the review findings and none affect reported
  numbers.

## 11. Future work

1. **Map-conditioned RFT.** Inject the `frontier_hint` text into the episodic
   RFT observation (the plumbing exists end-to-end) and test whether a small
   LLM learns to *use* a map summary the way the local RL probe suggests.
2. **VLM policies on matched rooms.** Run `VLMPolicy` on the matched 3D
   benchmark (needs only an API key) against the pixels-only FBE row.
3. **Richer procedural worlds.** Doorways/multi-room topologies in
   `scene_gen` + `scene_mjcf`; the matched-world property is preserved by
   construction.
4. **Plane-fed mapping.** Use floor-plane inliers as free-space evidence and
   non-floor planes as occupied evidence in `EgoMap`, unifying Sections 5
   and 7 into a single pixels-to-map pipeline.
5. **Full-mesh scene exports** to close the Gizmo visual/collision gap.
