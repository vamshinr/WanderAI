# Scene-Agnostic Object Search Environment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable, tested RL environment where an egocentric agent navigates continuous 2D space to reach a red ball, rewarded by geodesic (around-obstacle) distance plus a speed incentive.

**Architecture:** Pure-Python `wanderai` package. A `Scene` (floor + obstacle footprints + ball) feeds an `OccupancyGrid`, which feeds a wavefront `DistanceField` (shortest walkable distance to the ball, computed once per episode). A `Renderer` interface (StubRenderer now, Antim later) produces egocentric RGB. `SceneSearchEnv` ties them together with gym-style `reset()/step()`, computing the potential-based reward from privileged ground-truth distance. `metrics` computes SPL; `policies` provides random + oracle policies for end-to-end validation.

**Tech Stack:** Python 3.10+, numpy, pytest. No RL framework dependency (gym-style API hand-rolled to avoid version friction; gymnasium wrapper is a later seam).

## Global Constraints

- Python 3.10+; dependencies limited to `numpy` and `pytest`.
- Distances in meters; angles in radians internally (degrees only at config boundaries).
- Reward is privileged: computed from ground-truth state; never exposed in the observation.
- Geodesic distance only — Euclidean distance is explicitly forbidden for reward.
- All randomness takes an explicit `numpy.random.Generator` seed for determinism.
- Package import root is `wanderai`; tests live under `tests/`.

---

### Task 1: Scaffold + geometry primitives

**Files:**
- Create: `pyproject.toml`, `wanderai/__init__.py`, `wanderai/geometry.py`
- Test: `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `Pose(x: float, y: float, heading: float)` dataclass (heading radians).
  - `AABB(min_x, min_y, max_x, max_y)` dataclass with `.inflate(r) -> AABB`, `.contains(x, y) -> bool`.
  - `segment_intersects_aabb(x0,y0,x1,y1, box: AABB) -> bool` (slab method).
  - `wrap_angle(a: float) -> float` → (-π, π].

- [ ] **Step 1: Write failing tests**

```python
# tests/test_geometry.py
import math
from wanderai.geometry import Pose, AABB, segment_intersects_aabb, wrap_angle

def test_aabb_contains_and_inflate():
    b = AABB(0, 0, 2, 2)
    assert b.contains(1, 1)
    assert not b.contains(3, 1)
    big = b.inflate(0.5)
    assert big.min_x == -0.5 and big.max_x == 2.5
    assert big.contains(-0.4, 1)

def test_segment_intersects_aabb():
    b = AABB(1, 1, 2, 2)
    assert segment_intersects_aabb(0, 1.5, 3, 1.5, b)      # passes through
    assert not segment_intersects_aabb(0, 0, 0.5, 0.5, b)  # misses
    assert segment_intersects_aabb(1.5, 0, 1.5, 3, b)      # vertical through

def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi), math.pi)
    assert math.isclose(wrap_angle(-3 * math.pi), math.pi)
    assert math.isclose(wrap_angle(0.5), 0.5)
```

- [ ] **Step 2: Run, verify fail** — `pytest tests/test_geometry.py -v` → FAIL (import error).

- [ ] **Step 3: Implement**

```python
# pyproject.toml
[project]
name = "wanderai"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["numpy"]
[project.optional-dependencies]
dev = ["pytest"]
[tool.setuptools.packages.find]
include = ["wanderai*"]
```

```python
# wanderai/__init__.py
"""WanderAI: scene-agnostic egocentric object-search RL environment."""
```

```python
# wanderai/geometry.py
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float  # radians

@dataclass(frozen=True)
class AABB:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def inflate(self, r: float) -> "AABB":
        return AABB(self.min_x - r, self.min_y - r, self.max_x + r, self.max_y + r)

def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a, 2 * math.pi)
    if a <= -math.pi:
        a += 2 * math.pi
    elif a > math.pi:
        a -= 2 * math.pi
    return a

def segment_intersects_aabb(x0, y0, x1, y1, box: AABB) -> bool:
    """Liang-Barsky slab clipping; True if segment touches the box."""
    dx, dy = x1 - x0, y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - box.min_x, box.max_x - x0, y0 - box.min_y, box.max_y - y0]
    t0, t1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_geometry.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: geometry primitives + scaffold"`

---

### Task 2: Scene + default fixture

**Files:**
- Create: `wanderai/scene.py`
- Test: `tests/test_scene.py`

**Interfaces:**
- Consumes: `Pose`, `AABB`.
- Produces:
  - `Scene(bounds: AABB, obstacles: list[AABB], ball: tuple[float,float], agent_start: Pose, agent_radius: float)` dataclass.
  - `Scene.is_free(x, y) -> bool` (inside bounds, outside any obstacle inflated by `agent_radius`).
  - `default_scene() -> Scene` — a 6×6 m room, 3 box obstacles, ball in a corner behind one obstacle, agent start opposite.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scene.py
from wanderai.scene import Scene, default_scene
from wanderai.geometry import AABB, Pose

def test_is_free_respects_bounds_and_obstacles():
    s = Scene(bounds=AABB(0,0,6,6), obstacles=[AABB(2,2,4,4)],
              ball=(5.5,5.5), agent_start=Pose(0.5,0.5,0.0), agent_radius=0.2)
    assert s.is_free(0.5, 0.5)         # open floor
    assert not s.is_free(3, 3)         # inside obstacle
    assert not s.is_free(1.9, 3)       # within agent_radius of obstacle
    assert not s.is_free(-0.1, 3)      # outside bounds

def test_default_scene_valid():
    s = default_scene()
    assert s.is_free(s.agent_start.x, s.agent_start.y)
    assert s.is_free(*s.ball)
    assert s.bounds.contains(*s.ball)
```

- [ ] **Step 2: Run, verify fail** — FAIL (import error).

- [ ] **Step 3: Implement**

```python
# wanderai/scene.py
from __future__ import annotations
from dataclasses import dataclass, field
from .geometry import AABB, Pose

@dataclass(frozen=True)
class Scene:
    bounds: AABB
    obstacles: list[AABB]
    ball: tuple[float, float]
    agent_start: Pose
    agent_radius: float = 0.2

    def is_free(self, x: float, y: float) -> bool:
        if not self.bounds.contains(x, y):
            return False
        for ob in self.obstacles:
            if ob.inflate(self.agent_radius).contains(x, y):
                return False
        return True

def default_scene() -> Scene:
    return Scene(
        bounds=AABB(0, 0, 6, 6),
        obstacles=[AABB(2.0, 2.0, 3.0, 4.5),   # vertical divider
                   AABB(3.8, 1.0, 4.6, 2.2),   # box near ball
                   AABB(1.0, 4.8, 4.0, 5.2)],  # top wall stub
        ball=(5.4, 1.4),
        agent_start=Pose(0.6, 0.6, 0.0),
        agent_radius=0.2,
    )
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: Scene + default fixture"`

---

### Task 3: OccupancyGrid

**Files:**
- Create: `wanderai/occupancy.py`
- Test: `tests/test_occupancy.py`

**Interfaces:**
- Consumes: `Scene`.
- Produces:
  - `OccupancyGrid.from_scene(scene, cell_size=0.1) -> OccupancyGrid`.
  - Attributes: `.blocked: np.ndarray[bool]` shape `(nrows, ncols)` (row=y, col=x), `.cell_size`, `.origin=(min_x,min_y)`, `.nrows`, `.ncols`.
  - `.world_to_cell(x, y) -> tuple[int,int]` (row, col); `.cell_to_world(row, col) -> tuple[float,float]` (cell center).
  - `.is_blocked_world(x, y) -> bool` (out-of-bounds counts as blocked).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_occupancy.py
import numpy as np
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.occupancy import OccupancyGrid

def _scene():
    return Scene(AABB(0,0,4,4), [AABB(1,1,2,2)], (3.5,3.5), Pose(0.5,0.5,0), 0.0)

def test_grid_shape_and_roundtrip():
    g = OccupancyGrid.from_scene(_scene(), cell_size=0.5)
    assert g.blocked.shape == (8, 8)
    r, c = g.world_to_cell(0.25, 0.25)
    assert (r, c) == (0, 0)
    x, y = g.cell_to_world(0, 0)
    assert abs(x-0.25) < 1e-9 and abs(y-0.25) < 1e-9

def test_obstacle_cells_blocked():
    g = OccupancyGrid.from_scene(_scene(), cell_size=0.5)
    assert g.is_blocked_world(1.5, 1.5)     # inside obstacle
    assert not g.is_blocked_world(0.25, 0.25)
    assert g.is_blocked_world(-1, 2)        # out of bounds
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/occupancy.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .scene import Scene

@dataclass
class OccupancyGrid:
    blocked: np.ndarray   # bool [nrows, ncols], row=y, col=x
    cell_size: float
    origin: tuple[float, float]

    @property
    def nrows(self) -> int: return self.blocked.shape[0]
    @property
    def ncols(self) -> int: return self.blocked.shape[1]

    @classmethod
    def from_scene(cls, scene: Scene, cell_size: float = 0.1) -> "OccupancyGrid":
        b = scene.bounds
        ncols = int(round((b.max_x - b.min_x) / cell_size))
        nrows = int(round((b.max_y - b.min_y) / cell_size))
        blocked = np.zeros((nrows, ncols), dtype=bool)
        for r in range(nrows):
            for c in range(ncols):
                x = b.min_x + (c + 0.5) * cell_size
                y = b.min_y + (r + 0.5) * cell_size
                blocked[r, c] = not scene.is_free(x, y)
        return cls(blocked, cell_size, (b.min_x, b.min_y))

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.origin[0]) / self.cell_size)
        r = int((y - self.origin[1]) / self.cell_size)
        return r, c

    def cell_to_world(self, r: int, c: int) -> tuple[float, float]:
        x = self.origin[0] + (c + 0.5) * self.cell_size
        y = self.origin[1] + (r + 0.5) * self.cell_size
        return x, y

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.nrows and 0 <= c < self.ncols

    def is_blocked_world(self, x: float, y: float) -> bool:
        r, c = self.world_to_cell(x, y)
        if not self.in_bounds(r, c):
            return True
        return bool(self.blocked[r, c])
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: OccupancyGrid rasterization"`

---

### Task 4: DistanceField (wavefront geodesic)

**Files:**
- Create: `wanderai/distance_field.py`
- Test: `tests/test_distance_field.py`

**Interfaces:**
- Consumes: `OccupancyGrid`, ball world position.
- Produces:
  - `DistanceField.from_grid(grid, ball_xy) -> DistanceField` — Dijkstra from ball cell over free cells (8-connected, costs 1.0 / √2 × cell_size).
  - `.dist: np.ndarray[float]` (np.inf where unreachable/blocked).
  - `.query(x, y) -> float` — bilinear interpolation over finite neighbors; np.inf if no finite support.
  - `.is_reachable(x, y) -> bool`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_distance_field.py
import math, numpy as np
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.occupancy import OccupancyGrid
from wanderai.distance_field import DistanceField

def _empty_grid():
    s = Scene(AABB(0,0,4,4), [], (3.5,0.5), Pose(0.5,0.5,0), 0.0)
    return OccupancyGrid.from_scene(s, 0.5), (3.5, 0.5)

def test_distance_zero_at_ball():
    g, ball = _empty_grid()
    df = DistanceField.from_grid(g, ball)
    assert df.query(*ball) < 0.5

def test_open_field_matches_euclidean():
    g, ball = _empty_grid()
    df = DistanceField.from_grid(g, ball)
    d = df.query(0.5, 0.5)
    euclid = math.hypot(3.5-0.5, 0.5-0.5)
    assert abs(d - euclid) < 0.6   # 8-connected approx, coarse grid

def test_geodesic_exceeds_euclidean_with_wall():
    # Wall from y=0..3 at x≈2 forces a detour around the top.
    s = Scene(AABB(0,0,4,4), [AABB(1.9,0.0,2.1,3.0)], (3.5,0.5), Pose(0.5,0.5,0), 0.0)
    g = OccupancyGrid.from_scene(s, 0.25)
    df = DistanceField.from_grid(g, (3.5,0.5))
    geo = df.query(0.5, 0.5)
    euclid = math.hypot(3.0, 0.0)
    assert geo > euclid + 1.0      # must detour around the wall

def test_unreachable_is_inf():
    # Ball sealed in a box.
    s = Scene(AABB(0,0,4,4),
              [AABB(2,0,2.2,4), AABB(0,2,4,2.2)], (3.5,3.5), Pose(0.5,0.5,0), 0.0)
    g = OccupancyGrid.from_scene(s, 0.2)
    df = DistanceField.from_grid(g, (3.5,3.5))
    assert math.isinf(df.query(0.5,0.5))
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/distance_field.py
from __future__ import annotations
from dataclasses import dataclass
import heapq, math
import numpy as np
from .occupancy import OccupancyGrid

_NEIGHBORS = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
              (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),
              (1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]

@dataclass
class DistanceField:
    dist: np.ndarray            # float meters, np.inf if unreachable
    grid: OccupancyGrid

    @classmethod
    def from_grid(cls, grid: OccupancyGrid, ball_xy: tuple[float,float]) -> "DistanceField":
        dist = np.full(grid.blocked.shape, np.inf)
        sr, sc = grid.world_to_cell(*ball_xy)
        if not grid.in_bounds(sr, sc) or grid.blocked[sr, sc]:
            sr, sc = _nearest_free(grid, sr, sc)
        cs = grid.cell_size
        dist[sr, sc] = 0.0
        pq = [(0.0, sr, sc)]
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r, c]:
                continue
            for dr, dc, w in _NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not grid.in_bounds(nr, nc) or grid.blocked[nr, nc]:
                    continue
                nd = d + w * cs
                if nd < dist[nr, nc]:
                    dist[nr, nc] = nd
                    heapq.heappush(pq, (nd, nr, nc))
        return cls(dist, grid)

    def query(self, x: float, y: float) -> float:
        g = self.grid
        fx = (x - g.origin[0]) / g.cell_size - 0.5
        fy = (y - g.origin[1]) / g.cell_size - 0.5
        c0, r0 = int(math.floor(fx)), int(math.floor(fy))
        tx, ty = fx - c0, fy - r0
        total_w, acc = 0.0, 0.0
        for (dr, dc, w) in [(0,0,(1-tx)*(1-ty)), (0,1,tx*(1-ty)),
                            (1,0,(1-tx)*ty),     (1,1,tx*ty)]:
            r, c = r0 + dr, c0 + dc
            if g.in_bounds(r, c) and math.isfinite(self.dist[r, c]):
                acc += w * self.dist[r, c]
                total_w += w
        if total_w == 0.0:
            return math.inf
        return acc / total_w

    def is_reachable(self, x: float, y: float) -> bool:
        return math.isfinite(self.query(x, y))

def _nearest_free(grid: OccupancyGrid, r: int, c: int) -> tuple[int,int]:
    best, bestd = (r, c), math.inf
    for rr in range(grid.nrows):
        for cc in range(grid.ncols):
            if not grid.blocked[rr, cc]:
                d = (rr-r)**2 + (cc-c)**2
                if d < bestd:
                    bestd, best = d, (rr, cc)
    return best
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: geodesic DistanceField via wavefront Dijkstra"`

---

### Task 5: Renderer interface + StubRenderer

**Files:**
- Create: `wanderai/renderer.py`
- Test: `tests/test_renderer.py`

**Interfaces:**
- Consumes: `Scene`, `Pose`, `segment_intersects_aabb`, `wrap_angle`.
- Produces:
  - `Renderer` ABC: `.render(scene: Scene, pose: Pose) -> np.ndarray` (H×W×3 uint8).
  - `StubRenderer(width=64, height=64, fov=math.pi/2, max_view=8.0)`.
  - `ball_visible(scene, pose, fov, max_view) -> bool` — ball within FOV half-angle, within `max_view`, and line-of-sight to ball clear of all obstacle AABBs.
  - StubRenderer draws a red vertical band at the ball's bearing column when visible; otherwise a flat gray image.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_renderer.py
import math, numpy as np
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.renderer import StubRenderer, ball_visible

def test_visible_when_facing_clear():
    s = Scene(AABB(0,0,6,6), [], (5,3), Pose(1,3,0.0), 0.0)  # ball straight ahead (+x)
    assert ball_visible(s, Pose(1,3,0.0), math.pi/2, 8.0)
    assert not ball_visible(s, Pose(1,3,math.pi), math.pi/2, 8.0)  # facing away

def test_occluded_by_obstacle():
    s = Scene(AABB(0,0,6,6), [AABB(2.8,2.5,3.2,3.5)], (5,3), Pose(1,3,0.0), 0.0)
    assert not ball_visible(s, Pose(1,3,0.0), math.pi/2, 8.0)  # box between agent & ball

def test_render_shape_and_red_when_visible():
    r = StubRenderer()
    s = Scene(AABB(0,0,6,6), [], (5,3), Pose(1,3,0.0), 0.0)
    img = r.render(s, Pose(1,3,0.0))
    assert img.shape == (64,64,3) and img.dtype == np.uint8
    assert img[:, :, 0].max() > 150 and img[:, :, 1].max() < 120  # has red, little green
    blank = r.render(s, Pose(1,3,math.pi))
    assert blank[:, :, 0].max() < 130   # no red band when facing away
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/renderer.py
from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np
from .scene import Scene
from .geometry import Pose, segment_intersects_aabb, wrap_angle

def ball_visible(scene: Scene, pose: Pose, fov: float, max_view: float) -> bool:
    bx, by = scene.ball
    dx, dy = bx - pose.x, by - pose.y
    dist = math.hypot(dx, dy)
    if dist > max_view or dist < 1e-6:
        return dist < 1e-6  # standing on it counts as visible
    bearing = wrap_angle(math.atan2(dy, dx) - pose.heading)
    if abs(bearing) > fov / 2:
        return False
    for ob in scene.obstacles:
        if segment_intersects_aabb(pose.x, pose.y, bx, by, ob):
            return False
    return True

class Renderer(ABC):
    @abstractmethod
    def render(self, scene: Scene, pose: Pose) -> np.ndarray: ...

class StubRenderer(Renderer):
    def __init__(self, width=64, height=64, fov=math.pi/2, max_view=8.0):
        self.width, self.height, self.fov, self.max_view = width, height, fov, max_view

    def render(self, scene: Scene, pose: Pose) -> np.ndarray:
        img = np.full((self.height, self.width, 3), 110, dtype=np.uint8)  # gray floor
        if ball_visible(scene, pose, self.fov, self.max_view):
            bx, by = scene.ball
            bearing = wrap_angle(math.atan2(by - pose.y, bx - pose.x) - pose.heading)
            col = int((bearing / self.fov + 0.5) * self.width)
            col = max(0, min(self.width - 1, col))
            dist = math.hypot(bx - pose.x, by - pose.y)
            half = max(2, int(self.width * 0.12 / max(dist, 0.5)))
            lo, hi = max(0, col - half), min(self.width, col + half + 1)
            img[:, lo:hi, 0] = 220   # red
            img[:, lo:hi, 1] = 30
            img[:, lo:hi, 2] = 30
        return img
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: Renderer interface + StubRenderer with FOV/occlusion"`

---

### Task 6: SceneSearchEnv (reset/step + reward)

**Files:**
- Create: `wanderai/environment.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Consumes: `Scene`, `OccupancyGrid`, `DistanceField`, `Renderer`/`StubRenderer`, `Pose`, `wrap_angle`.
- Produces:
  - `Action` IntEnum: `MOVE_FORWARD=0, TURN_LEFT=1, TURN_RIGHT=2`.
  - `EnvConfig` dataclass: `cell_size=0.1, step_size=0.25, turn=math.radians(30), success_radius=0.3, max_steps=200, alpha=1.0, beta=0.02, kappa=0.1, goal_reward=10.0, gamma=0.99`.
  - `SceneSearchEnv(scene, renderer=None, config=EnvConfig())`.
  - `.reset() -> (obs: np.ndarray, info: dict)` — builds grid + distance field, resets pose/step count, `info` has `geodesic`, `optimal` (= geodesic at start), `path_length=0`.
  - `.step(action) -> (obs, reward, done, info)` — applies action with collision rejection; reward per spec; `done` on success or max_steps; `info` has `geodesic`, `success`, `collision`, `path_length`, `optimal`, `steps`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_environment.py
import math, numpy as np
from wanderai.scene import Scene, default_scene
from wanderai.geometry import AABB, Pose
from wanderai.environment import SceneSearchEnv, EnvConfig, Action

def test_reset_returns_obs_and_optimal():
    env = SceneSearchEnv(default_scene())
    obs, info = env.reset()
    assert obs.shape[2] == 3
    assert info["optimal"] > 0 and math.isfinite(info["optimal"])
    assert info["path_length"] == 0.0

def test_progress_reward_positive_when_closer():
    # Open room, ball ahead along +x; moving forward must reduce geodesic distance.
    s = Scene(AABB(0,0,6,6), [], (5.0,1.0), Pose(1.0,1.0,0.0), 0.2)
    env = SceneSearchEnv(s, config=EnvConfig(cell_size=0.1))
    env.reset()
    _, reward, _, info = env.step(Action.MOVE_FORWARD)
    assert reward > 0           # progress term beats time penalty
    assert not info["collision"]

def test_collision_blocks_and_penalizes():
    # Wall directly ahead; forward move is rejected, pose unchanged, collision flagged.
    s = Scene(AABB(0,0,6,6), [AABB(1.3,0.0,1.6,6.0)], (5,1), Pose(1.0,1.0,0.0), 0.2)
    env = SceneSearchEnv(s, config=EnvConfig(step_size=0.25))
    env.reset()
    before = (env.pose.x, env.pose.y)
    _, reward, _, info = env.step(Action.MOVE_FORWARD)
    assert info["collision"] and reward < 0
    assert (env.pose.x, env.pose.y) == before

def test_turn_changes_heading_only():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(turn=math.radians(30)))
    env.reset()
    h0 = env.pose.heading
    env.step(Action.TURN_LEFT)
    assert abs(env.pose.heading - (h0 + math.radians(30))) < 1e-6

def test_success_terminates_with_bonus():
    s = Scene(AABB(0,0,6,6), [], (1.3,1.0), Pose(1.0,1.0,0.0), 0.2)
    env = SceneSearchEnv(s, config=EnvConfig(step_size=0.25, success_radius=0.3, goal_reward=10.0))
    env.reset()
    done = False
    for _ in range(5):
        _, reward, done, info = env.step(Action.MOVE_FORWARD)
        if done:
            break
    assert done and info["success"] and reward > 5
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/environment.py
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
from .scene import Scene
from .geometry import Pose, wrap_angle
from .occupancy import OccupancyGrid
from .distance_field import DistanceField
from .renderer import Renderer, StubRenderer

class Action(IntEnum):
    MOVE_FORWARD = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2

@dataclass
class EnvConfig:
    cell_size: float = 0.1
    step_size: float = 0.25
    turn: float = math.radians(30)
    success_radius: float = 0.3
    max_steps: int = 200
    alpha: float = 1.0
    beta: float = 0.02
    kappa: float = 0.1
    goal_reward: float = 10.0
    gamma: float = 0.99

class SceneSearchEnv:
    def __init__(self, scene: Scene, renderer: Renderer | None = None, config: EnvConfig | None = None):
        self.scene = scene
        self.config = config or EnvConfig()
        self.renderer = renderer or StubRenderer()
        self.grid: OccupancyGrid | None = None
        self.field: DistanceField | None = None
        self.pose: Pose = scene.agent_start
        self.steps = 0
        self.path_length = 0.0
        self.optimal = math.inf
        self._prev_d = math.inf

    def reset(self):
        self.grid = OccupancyGrid.from_scene(self.scene, self.config.cell_size)
        self.field = DistanceField.from_grid(self.grid, self.scene.ball)
        self.pose = self.scene.agent_start
        self.steps = 0
        self.path_length = 0.0
        self.optimal = self.field.query(self.pose.x, self.pose.y)
        self._prev_d = self.optimal
        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": self._prev_d, "optimal": self.optimal,
                "path_length": 0.0, "success": False, "collision": False, "steps": 0}
        return obs, info

    def step(self, action: int):
        cfg = self.config
        collision = False
        if action == Action.MOVE_FORWARD:
            nx = self.pose.x + cfg.step_size * math.cos(self.pose.heading)
            ny = self.pose.y + cfg.step_size * math.sin(self.pose.heading)
            if self.grid.is_blocked_world(nx, ny):
                collision = True
            else:
                self.path_length += math.hypot(nx - self.pose.x, ny - self.pose.y)
                self.pose = Pose(nx, ny, self.pose.heading)
        elif action == Action.TURN_LEFT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading + cfg.turn))
        elif action == Action.TURN_RIGHT:
            self.pose = Pose(self.pose.x, self.pose.y, wrap_angle(self.pose.heading - cfg.turn))

        d = self.field.query(self.pose.x, self.pose.y)
        progress = (self._prev_d - d) if math.isfinite(d) and math.isfinite(self._prev_d) else 0.0
        self._prev_d = d
        self.steps += 1

        euclid_to_ball = math.hypot(self.scene.ball[0]-self.pose.x, self.scene.ball[1]-self.pose.y)
        success = euclid_to_ball <= cfg.success_radius
        reward = cfg.alpha * progress - cfg.beta - (cfg.kappa if collision else 0.0)
        if success:
            reward += cfg.goal_reward
        done = success or self.steps >= cfg.max_steps

        obs = self.renderer.render(self.scene, self.pose)
        info = {"geodesic": d, "optimal": self.optimal, "path_length": self.path_length,
                "success": success, "collision": collision, "steps": self.steps}
        return obs, reward, done, info
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: SceneSearchEnv with geodesic reward"`

---

### Task 7: Metrics (SPL)

**Files:**
- Create: `wanderai/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `EpisodeResult(success: bool, optimal: float, path_length: float, steps: int)` dataclass.
  - `spl(results: list[EpisodeResult]) -> float` — `(1/N) Σ S·ℓ/max(p,ℓ)`; failed or zero-path episodes contribute 0; ℓ=0 guarded.
  - `summarize(results) -> dict` with `success_rate`, `spl`, `mean_steps` (successful episodes only; 0.0 if none).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metrics.py
import math
from wanderai.metrics import EpisodeResult, spl, summarize

def test_spl_perfect_path():
    r = EpisodeResult(success=True, optimal=4.0, path_length=4.0, steps=16)
    assert math.isclose(spl([r]), 1.0)

def test_spl_penalizes_detour():
    r = EpisodeResult(success=True, optimal=4.0, path_length=8.0, steps=32)
    assert math.isclose(spl([r]), 0.5)

def test_failed_episode_zero():
    r = EpisodeResult(success=False, optimal=4.0, path_length=2.0, steps=200)
    assert spl([r]) == 0.0

def test_summarize():
    rs = [EpisodeResult(True,4,4,16), EpisodeResult(False,4,3,200)]
    out = summarize(rs)
    assert math.isclose(out["success_rate"], 0.5)
    assert math.isclose(out["spl"], 0.5)
    assert out["mean_steps"] == 16
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/metrics.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class EpisodeResult:
    success: bool
    optimal: float
    path_length: float
    steps: int

def spl(results: list[EpisodeResult]) -> float:
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.success and r.optimal > 0:
            total += r.optimal / max(r.path_length, r.optimal)
    return total / len(results)

def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"success_rate": 0.0, "spl": 0.0, "mean_steps": 0.0}
    succ = [r for r in results if r.success]
    return {
        "success_rate": len(succ) / n,
        "spl": spl(results),
        "mean_steps": sum(r.steps for r in succ) / len(succ) if succ else 0.0,
    }
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -am "feat: SPL + episode metrics"`

---

### Task 8: Policies + runnable demo

**Files:**
- Create: `wanderai/policies.py`, `scripts/run_episode.py`
- Test: `tests/test_policies.py`

**Interfaces:**
- Consumes: `SceneSearchEnv`, `Action`, `DistanceField`, `EpisodeResult`, `summarize`.
- Produces:
  - `RandomPolicy(seed: int)` with `.act(obs, env) -> Action`.
  - `OraclePolicy()` with `.act(obs, env) -> Action` — peeks at `env.field`/`env.pose` (privileged, for validation only): turns toward the neighbor direction that most reduces geodesic distance, else moves forward. Proves the env is solvable and yields high SPL.
  - `run_episode(env, policy) -> EpisodeResult`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_policies.py
from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode

def test_oracle_solves_default_scene():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=400))
    res = run_episode(env, OraclePolicy())
    assert res.success
    assert res.path_length <= res.optimal * 1.6   # near-optimal route

def test_random_policy_runs_without_error():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=50))
    res = run_episode(env, RandomPolicy(seed=0))
    assert res.steps <= 50
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
# wanderai/policies.py
from __future__ import annotations
import math
import numpy as np
from .environment import SceneSearchEnv, Action
from .geometry import wrap_angle
from .metrics import EpisodeResult

class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
    def act(self, obs, env) -> Action:
        return Action(int(self.rng.integers(0, 3)))

class OraclePolicy:
    """Privileged: descends the geodesic field. Validation/upper-bound only."""
    def act(self, obs, env: SceneSearchEnv) -> Action:
        p = env.pose
        best_dir, best_d = None, math.inf
        for k in range(8):
            ang = k * math.pi / 4
            tx = p.x + env.config.step_size * math.cos(ang)
            ty = p.y + env.config.step_size * math.sin(ang)
            if env.grid.is_blocked_world(tx, ty):
                continue
            d = env.field.query(tx, ty)
            if d < best_d:
                best_d, best_dir = d, ang
        if best_dir is None:
            return Action.TURN_LEFT
        err = wrap_angle(best_dir - p.heading)
        if abs(err) <= env.config.turn / 2:
            return Action.MOVE_FORWARD
        return Action.TURN_LEFT if err > 0 else Action.TURN_RIGHT

def run_episode(env: SceneSearchEnv, policy) -> EpisodeResult:
    obs, info = env.reset()
    done = False
    while not done:
        action = policy.act(obs, env)
        obs, reward, done, info = env.step(action)
    return EpisodeResult(success=info["success"], optimal=info["optimal"],
                         path_length=info["path_length"], steps=info["steps"])
```

```python
# scripts/run_episode.py
"""Run policies on the default scene and print metrics. Usage: python -m scripts.run_episode"""
from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode
from wanderai.metrics import summarize

def main():
    for name, policy in [("oracle", OraclePolicy()), ("random", RandomPolicy(seed=0))]:
        results = [run_episode(SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=400)), policy)
                   for _ in range(5)]
        print(name, summarize(results))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_policies.py -v` then `python -m scripts.run_episode` (oracle success_rate≈1.0, high SPL).
- [ ] **Step 5: Commit** — `git commit -am "feat: random + oracle policies, demo runner"`

---

## Self-Review

**Spec coverage:** Scene/fixtures (T2), OccupancyGrid (T3), DistanceField geodesic (T4), StubRenderer + occlusion (T5), Environment + reward formula α·progress − β − κ·collision + R·success (T6), SPL eval (T7), random+oracle policies & demo (T8). Antim renderer / Fireworks / memory are intentionally out of scope per spec — left as the `Renderer` interface and policy seams. ✓

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `info` dict keys (`geodesic`, `optimal`, `path_length`, `success`, `collision`, `steps`) consistent across T6→T8; `EpisodeResult` fields consistent T7→T8; `DistanceField.query`/`grid.is_blocked_world` names consistent T4→T6→T8. ✓

**Note on success check:** success uses straight-line distance to the ball within `success_radius` (physically reaching it), while the *reward shaping* uses geodesic distance — consistent with the spec (proximity success, geodesic shaping).
