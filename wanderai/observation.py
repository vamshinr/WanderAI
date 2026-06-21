"""Symbolic, egocentric observation for the text policy (Phase A of RFT).

The agent never gets the map. It gets a compact, partial view computed from the
scene geometry: whether the red ball is currently visible (and where), how much
free space is ahead/left/right (ray-cast clearance), and its own recent moves.
Partial observability is deliberate — the ball only appears when it is genuinely
in line of sight, so the policy must learn to *search*, not just home in."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from .scene import Scene
from .geometry import Pose, AABB, ray_aabb, wrap_angle
from .renderer import ball_visible

# Defaults shared with the StubRenderer so text and image views agree.
DEFAULT_FOV = math.pi / 2
DEFAULT_MAX_VIEW = 8.0
DEFAULT_CLEARANCE_RANGE = 6.0
DEFAULT_HISTORY = 4


def cast_ray(scene: Scene, x: float, y: float, angle: float, max_range: float) -> float:
    """Distance from (x,y) along `angle` to the nearest blocking surface — an
    obstacle face or a room wall — capped at max_range."""
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_range
    # Room boundary: the agent is inside, so the wall is the ray's exit distance.
    hit = ray_aabb(x, y, dx, dy, scene.bounds)
    if hit is not None:
        tmin, tmax = hit
        exit_t = tmax if tmin < 0 else tmin
        if 0 < exit_t < best:
            best = exit_t
    # Obstacles: entry distance from outside.
    for ob in scene.obstacles:
        hit = ray_aabb(x, y, dx, dy, ob)
        if hit is not None:
            tmin, tmax = hit
            entry_t = tmin if tmin > 0 else 0.0
            if 0 < entry_t < best:
                best = entry_t
    return best


@dataclass
class Observation:
    ball_visible: bool
    ball_bearing: float | None      # radians, relative to heading (+ = left)
    ball_distance: float | None     # meters
    clearance: dict                 # {"left", "center", "right"} -> meters
    recent_actions: list            # action names, oldest..newest


def observe(scene: Scene, pose: Pose, history=None,
           fov: float = DEFAULT_FOV, max_view: float = DEFAULT_MAX_VIEW,
           clearance_range: float = DEFAULT_CLEARANCE_RANGE) -> Observation:
    visible = ball_visible(scene, pose, fov, max_view)
    bearing = distance = None
    if visible:
        bx, by = scene.ball
        bearing = wrap_angle(math.atan2(by - pose.y, bx - pose.x) - pose.heading)
        distance = math.hypot(bx - pose.x, by - pose.y)
    clearance = {
        "left": cast_ray(scene, pose.x, pose.y, pose.heading + fov / 2, clearance_range),
        "center": cast_ray(scene, pose.x, pose.y, pose.heading, clearance_range),
        "right": cast_ray(scene, pose.x, pose.y, pose.heading - fov / 2, clearance_range),
    }
    recent = [getattr(a, "name", str(a)) for a in (history or [])][-DEFAULT_HISTORY:]
    return Observation(visible, bearing, distance, clearance, recent)


def observation_text(obs: Observation) -> str:
    """Render the observation as the prompt the text policy reads."""
    if obs.ball_visible:
        deg = math.degrees(obs.ball_bearing)
        side = "ahead" if abs(deg) < 1 else (f"{abs(deg):.0f}deg left" if deg > 0
                                             else f"{abs(deg):.0f}deg right")
        ball = f"Red ball: VISIBLE, {side}, distance {obs.ball_distance:.1f}m."
    else:
        ball = "Red ball: not visible."
    c = obs.clearance
    clear = (f"Clearance — left {c['left']:.1f}m, center {c['center']:.1f}m, "
             f"right {c['right']:.1f}m.")
    moves = ", ".join(obs.recent_actions) if obs.recent_actions else "none"
    return f"{ball} {clear} Recent moves: {moves}."
