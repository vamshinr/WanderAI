"""Compile a WanderAI `Scene` into a MuJoCo MJCF room — matched 3D worlds.

Gizmo exports carry furniture as position-only bodies (their meshes live
outside the export), so an imported scene's *visual* world and *collision*
world can disagree — the camera sees sofas the 2D occupancy never knew about.
That is fine for a demo but poisonous for measuring a perception pipeline:
mapping errors become indistinguishable from world mismatch.

This module goes the other way: it renders OUR procedural scenes in 3D, so the
rendered geometry matches the collision/geodesic substrate *exactly* — floor
plane at z=0, four perimeter walls whose inner faces sit on `scene.bounds`,
one box per obstacle AABB, and the emissive red ball. Every discrepancy a
vision policy shows against its geometry-mode twin is then attributable to
perception, not to the scene. It also scales: any `random_scene()` room
becomes a 3D room, which is how the vision benchmark gets N rooms instead of
one hand-exported fixture.

Obstacle heights are seeded deterministically from the obstacle's footprint so
rooms are stable across runs: some furniture is knee-high (hard: below the
horizon band), some is head-high (easy: occludes and blocks sight).
"""
from __future__ import annotations

import hashlib
import math

from .scene import Scene
# Single source of truth for the goal object's appearance: perception's
# red-blob detector is tuned to THIS ball. Forking the values here would let
# procedural rooms and imported rooms drift apart, confounding vision results.
from .mujoco_renderer import BALL_RADIUS, BALL_RGBA

WALL_HEIGHT = 2.6
WALL_THICKNESS = 0.15
MIN_OBSTACLE_H = 0.45
MAX_OBSTACLE_H = 1.9


def _obstacle_height(ob) -> float:
    """Deterministic pseudo-random height from the footprint (stable per room)."""
    key = f"{ob.min_x:.3f},{ob.min_y:.3f},{ob.max_x:.3f},{ob.max_y:.3f}".encode()
    u = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") / 2**32
    return MIN_OBSTACLE_H + u * (MAX_OBSTACLE_H - MIN_OBSTACLE_H)


def scene_to_mjcf(scene: Scene, *, ball_radius: float = BALL_RADIUS,
                  wall_height: float = WALL_HEIGHT) -> str:
    """MJCF XML whose geometry matches `scene` exactly (floor at z=0)."""
    b = scene.bounds
    w, h = b.max_x - b.min_x, b.max_y - b.min_y
    cx, cy = b.min_x + w / 2, b.min_y + h / 2
    t = WALL_THICKNESS

    geoms = [f'<geom name="floor" type="plane" pos="{cx} {cy} 0" '
             f'size="{w / 2} {h / 2} .1" rgba="0.58 0.56 0.53 1"/>']
    # Perimeter walls: inner faces exactly on the bounds (thickness outward),
    # so ray-casts against `bounds` and rendered depth agree at the perimeter.
    for name, px, py, sx, sy in (
            ("wall_s", cx, b.min_y - t / 2, w / 2 + t, t / 2),
            ("wall_n", cx, b.max_y + t / 2, w / 2 + t, t / 2),
            ("wall_w", b.min_x - t / 2, cy, t / 2, h / 2 + t),
            ("wall_e", b.max_x + t / 2, cy, t / 2, h / 2 + t)):
        geoms.append(f'<geom name="{name}" type="box" '
                     f'pos="{px} {py} {wall_height / 2}" '
                     f'size="{sx} {sy} {wall_height / 2}" rgba="0.85 0.84 0.80 1"/>')
    for i, ob in enumerate(scene.obstacles):
        oh = _obstacle_height(ob)
        px, py = (ob.min_x + ob.max_x) / 2, (ob.min_y + ob.max_y) / 2
        sx, sy = (ob.max_x - ob.min_x) / 2, (ob.max_y - ob.min_y) / 2
        shade = 0.30 + 0.06 * (i % 4)
        geoms.append(f'<geom name="obstacle_{i}" type="box" pos="{px} {py} {oh / 2}" '
                     f'size="{sx} {sy} {oh / 2}" '
                     f'rgba="{shade + 0.08} {shade} {shade - 0.05} 1"/>')
    room = f"""<mujoco model="wander_room">
  <visual><global offwidth="640" offheight="480"/><quality shadowsize="0"/></visual>
  <worldbody>
    <light directional="true" pos="{cx} {cy} 6" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <light directional="true" pos="{b.min_x} {b.min_y} 4" dir="0.4 0.4 -1" diffuse="0.35 0.35 0.35"/>
    {chr(10).join('    ' + g for g in geoms)}
  </worldbody>
</mujoco>"""
    # The ball is injected by the same helper the Gizmo-import path uses, so
    # its material/name/emission can never fork between the two 3D pipelines.
    from .mujoco_renderer import inject_red_ball
    return inject_red_ball(room, scene.ball, ball_radius, radius=ball_radius)


def scene_renderer_3d(scene: Scene, *, width: int = 192, height: int = 144,
                      **renderer_kwargs):
    """(scene, MuJoCoRenderer) for a procedurally generated room — the 3D twin
    of the 2D scene, sharing its exact coordinates (world_offset 0, floor 0)."""
    import mujoco
    from .mujoco_renderer import MuJoCoRenderer
    model = mujoco.MjModel.from_xml_string(scene_to_mjcf(scene))
    diag = math.hypot(scene.bounds.max_x - scene.bounds.min_x,
                      scene.bounds.max_y - scene.bounds.min_y)
    renderer_kwargs.setdefault("max_depth", float(max(8.0, round(diag) + 1)))
    renderer = MuJoCoRenderer(model, world_offset=(0.0, 0.0), floor_z=0.0,
                              width=width, height=height, **renderer_kwargs)
    return scene, renderer
