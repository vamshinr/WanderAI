"""Frontier-based exploration policy (Yamauchi 1997) — the classical, zero-training
baseline that modern zero-shot ObjectNav systems (CoW, ESC, VLFM) build on.

Honesty contract — the policy senses the world ONLY through:
  * its own pose (`env.pose`) — the GPS+Compass sensor that is part of the standard
    ObjectNav task specification (Batra et al. 2020);
  * the egocentric observation (ball visibility/bearing once genuinely in view,
    with occlusion — same channel every other WanderAI policy reads);
  * a 1-D depth strip across its FOV (simulated ray-cast in 2D; decoded from the
    rendered MuJoCo depth buffer in vision mode).
It never reads the environment's occupancy grid, geodesic field, or the ball's
hidden position. The map it plans over is built online by `EgoMap` from its own
scans, so exploration quality is exactly as good as its own mapping.

Two modes:
  * search — pick a frontier (cost–utility: cluster size discounted by travel
    distance, Gonzalez-Banos & Latombe 2002), A* to it over the agent's own map,
    follow waypoints; replan when the plan goes stale or a collision is inferred.
  * home — once the ball is visible, servo on its bearing and skirt obstacles
    using depth clearance only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .environment import Action, SceneSearchEnv
from .geometry import Pose
from .mapping import (EgoMap, depth_strip, obstacle_strip_from_depth,
                      DEFAULT_N_RAYS)
from .observation import observe, DEFAULT_FOV, DEFAULT_CLEARANCE_RANGE


@dataclass
class FrontierConfig:
    n_rays: int = DEFAULT_N_RAYS
    max_range: float = DEFAULT_CLEARANCE_RANGE
    map_cell: float = 0.25
    budget_cells: int = 0            # 0 = unbounded; set to ablate memory
    replan_every: int = 12           # steps between forced replans
    waypoint_radius: float = 0.35    # waypoint considered reached within this
    selection: str = "cost_utility"  # or "nearest"
    seed: int = 0


class FrontierPolicy:
    """`act(obs, env) -> Action`; self-resets whenever `env.steps == 0`."""

    def __init__(self, config: FrontierConfig | None = None):
        self.cfg = config or FrontierConfig()
        self._reset()

    def _reset(self):
        self.map = EgoMap(cell_size=self.cfg.map_cell,
                          budget_cells=self.cfg.budget_cells)
        self.plan: list[tuple[float, float]] = []
        self.steps_since_plan = 0
        self.last_pose: Pose | None = None
        self.last_action: Action | None = None
        self.consecutive_turns = 0
        self.banned: set = set()          # (x, y, heading)-buckets where forward collided
        self.skirt_side: Action | None = None   # sticky side while skirting (homing)
        self.explore_skirt: Action | None = None  # sticky side while skirting (search)
        self.ball_belief: tuple[float, float] | None = None  # last estimated ball position
        self.rng = np.random.default_rng(self.cfg.seed)

    # --- sensing (the only channels the policy is allowed) ---
    def _sense(self, env: SceneSearchEnv):
        pose = env.pose
        if env.config.perception == "vision" and hasattr(env.renderer, "render_rgb_depth"):
            from .perception import perceive
            rgb, depth = env.renderer.render_rgb_depth(env.scene, pose)
            obs = perceive(env.renderer, env.scene, pose, history=env.history,
                           visited=env.visited, frame=(rgb, depth))
            # Height-aware projection of the whole depth image (not one horizon
            # band, which sees over low furniture), capped at the same sensor
            # range the 2D ray-cast uses so "range == max_range" means "no
            # obstacle within range" in both modes.
            r = env.renderer
            strip = obstacle_strip_from_depth(
                depth, fov_x=float(getattr(r, "fov_x", DEFAULT_FOV)),
                fov_y=float(getattr(r, "fov_y", DEFAULT_FOV * 0.75)),
                eye_height=float(getattr(r, "eye_height", 1.4)),
                pitch_rad=math.radians(float(getattr(r, "pitch_deg", 0.0))),
                max_range=self.cfg.max_range, n_rays=self.cfg.n_rays)
        else:
            obs = observe(env.scene, pose, history=env.history, visited=env.visited)
            strip = depth_strip(env.scene, pose, n_rays=self.cfg.n_rays,
                                max_range=self.cfg.max_range)
        return obs, strip

    # --- helpers ---
    @staticmethod
    def _wrap(a: float) -> float:
        return math.atan2(math.sin(a), math.cos(a))

    def _turn_toward(self, rel_bearing: float) -> Action:
        return Action.TURN_LEFT if rel_bearing > 0 else Action.TURN_RIGHT

    def _center_clear(self, strip) -> float:
        mid = min(range(len(strip)), key=lambda i: abs(strip[i][0]))
        return strip[mid][1]

    def _pose_bucket(self, pose: Pose, ecfg) -> tuple[int, int, int]:
        return (int(round(pose.x / 0.1)), int(round(pose.y / 0.1)),
                int(round(pose.heading / ecfg.turn)))

    def _forward_ok(self, pose: Pose, strip, ecfg, margin: float = 1.2) -> bool:
        """Moving forward is sensible: the depth strip shows room, our own map
        doesn't mark the destination occupied, and this exact (pose, heading)
        hasn't already produced a collision (the env's occupancy raster is a
        little more conservative than exact geometry, so we learn its edges
        from experience instead of assuming our sensor is the last word)."""
        if self._pose_bucket(pose, ecfg) in self.banned:
            return False
        if self._center_clear(strip) <= ecfg.step_size * margin:
            return False
        nx = pose.x + ecfg.step_size * math.cos(pose.heading)
        ny = pose.y + ecfg.step_size * math.sin(pose.heading)
        # Gate only on STRONG occupancy evidence: a single stray sensor hit
        # (one L_OCC) must not freeze the robot — if the cell really is a wall
        # the collision ban catches it on the next step and marks it hard.
        lo = self.map.log_odds(self.map.key(nx, ny))
        return lo is None or lo <= 2.0

    def _choose_frontier(self, pose: Pose):
        clusters = self.map.frontiers()
        if not clusters:
            return None
        if self.cfg.selection == "nearest":
            return min(clusters, key=lambda c: math.hypot(
                c.centroid[0] - pose.x, c.centroid[1] - pose.y)).centroid
        best, best_score = None, -math.inf
        for c in clusters:
            d = math.hypot(c.centroid[0] - pose.x, c.centroid[1] - pose.y)
            score = c.size / (1.0 + d)          # cost–utility trade-off
            if score > best_score:
                best, best_score = c.centroid, score
        return best

    def _replan(self, pose: Pose):
        self.plan = []
        target = self._choose_frontier(pose)
        if target is None:
            return
        path = self.map.plan_path((pose.x, pose.y), target)
        if path:
            self.plan = path
        self.steps_since_plan = 0

    # --- policy interface ---
    def act(self, obs, env: SceneSearchEnv) -> Action:
        if env.steps == 0:
            self._reset()
        pose = env.pose
        cfg, ecfg = self.cfg, env.config

        # Infer a collision: we asked to move forward and the pose didn't change.
        if (self.last_action == Action.MOVE_FORWARD and self.last_pose is not None
                and math.hypot(pose.x - self.last_pose.x,
                               pose.y - self.last_pose.y) < 1e-9):
            bx = pose.x + ecfg.step_size * math.cos(pose.heading)
            by = pose.y + ecfg.step_size * math.sin(pose.heading)
            self.map._bump(self.map.key(bx, by), 4.0)   # blocked, from experience
            self.banned.add(self._pose_bucket(pose, ecfg))
            self.plan = []

        sym, strip = self._sense(env)
        self.map.update(pose, strip, max_range=cfg.max_range)

        action = self._decide(sym, strip, pose, ecfg)

        # Anti-dither: if we've spun a full circle, force progress when possible.
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            self.consecutive_turns += 1
            if (self.consecutive_turns > int(2 * math.pi / ecfg.turn) and
                    self._forward_ok(pose, strip, ecfg, margin=2.0)):
                action = Action.MOVE_FORWARD
        if action == Action.MOVE_FORWARD:
            self.consecutive_turns = 0

        self.last_pose, self.last_action = pose, action
        return action

    def _decide(self, sym, strip, pose: Pose, ecfg) -> Action:
        # Goal-position memory: while the ball is in view, keep a world-frame
        # estimate of it (bearing + sensed distance + own odometry). A camera
        # with a finite vertical FOV loses a floor-level ball inside ~1m — the
        # remembered position lets the agent finish the approach dead-reckoned,
        # exactly as map-based agents navigate to logged goal detections.
        if sym.ball_visible and sym.ball_bearing is not None and sym.ball_distance:
            a = pose.heading + sym.ball_bearing
            self.ball_belief = (pose.x + sym.ball_distance * math.cos(a),
                                pose.y + sym.ball_distance * math.sin(a))

        target = None                      # (bearing, distance) to steer at
        if sym.ball_visible and sym.ball_bearing is not None:
            target = (sym.ball_bearing, sym.ball_distance or math.inf)
        elif self.ball_belief is not None:
            dx, dy = self.ball_belief[0] - pose.x, self.ball_belief[1] - pose.y
            d = math.hypot(dx, dy)
            if d < 0.25:
                # We're standing on the remembered spot and see nothing — the
                # estimate was wrong; drop it and go back to searching.
                self.ball_belief = None
            elif d < 2.5:
                target = (self._wrap(math.atan2(dy, dx) - pose.heading), d)
            # A far-away stale belief is ignored (search will re-sight it).

        # --- home: on the ball in view, or its remembered position ---
        if target is not None:
            bearing, _ = target
            if self.skirt_side is None and abs(bearing) > ecfg.turn / 2:
                return self._turn_toward(bearing)
            if self._forward_ok(pose, strip, ecfg):
                self.skirt_side = None
                return Action.MOVE_FORWARD
            # Obstacle between us and the ball: commit to one side and keep
            # turning that way until forward opens up (wall-follow, not dither).
            if self.skirt_side is None:
                left = max(r for rel, r in strip if rel > 0)
                right = max(r for rel, r in strip if rel < 0)
                self.skirt_side = (Action.TURN_LEFT if left >= right
                                   else Action.TURN_RIGHT)
            return self.skirt_side
        self.skirt_side = None

        # --- search: frontier exploration on our own map ---
        self.steps_since_plan += 1
        if not self.plan or self.steps_since_plan >= self.cfg.replan_every:
            self._replan(pose)
        while self.plan and math.hypot(self.plan[0][0] - pose.x,
                                       self.plan[0][1] - pose.y) < self.cfg.waypoint_radius:
            self.plan.pop(0)
        if self.plan:
            # String-pulling: steer at the FARTHEST waypoint we can see on our
            # own map. Steering at each 0.25m grid waypoint makes a 30deg-turn
            # agent zigzag forever on diagonal segments.
            target = self.plan[0]
            for wp in reversed(self.plan):
                if self.map.line_of_sight((pose.x, pose.y), wp):
                    target = wp
                    break
            rel = self._wrap(math.atan2(target[1] - pose.y,
                                        target[0] - pose.x) - pose.heading)
            if abs(rel) > ecfg.turn * 0.6:
                return self._turn_toward(rel)
            if self._forward_ok(pose, strip, ecfg):
                self.explore_skirt = None
                return Action.MOVE_FORWARD
            self.plan = []          # blocked along the plan — force a replan

        # No plan (or plan blocked): open-space walk with a sticky skirt side,
        # so a wall-adjacent robot commits to one turning direction instead of
        # flip-flopping between left and right every step.
        if self._forward_ok(pose, strip, ecfg, margin=1.5):
            self.explore_skirt = None
            return Action.MOVE_FORWARD
        if self.explore_skirt is None:
            left = max(r for rel, r in strip if rel > 0)
            right = max(r for rel, r in strip if rel < 0)
            if abs(left - right) < 1e-6:
                self.explore_skirt = (Action.TURN_LEFT if self.rng.random() < 0.5
                                      else Action.TURN_RIGHT)
            else:
                self.explore_skirt = (Action.TURN_LEFT if left > right
                                      else Action.TURN_RIGHT)
        return self.explore_skirt
