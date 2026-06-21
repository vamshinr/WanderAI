"""MuJoCo egocentric renderer — Phase B vision (RGB + depth).

This is the real renderer behind the `Renderer` seam (replaces `StubRenderer`):
it loads a MuJoCo MJCF scene (e.g. a Gizmo/Antim export), drops our red ball into
it, and renders a first-person RGB image **and a depth buffer** from the agent's
pose. The depth buffer is what gives the agent genuine *distance perception* — the
perception layer reads object distances and wall clearances straight out of it
instead of from privileged geometry.

`mujoco` is imported lazily (only when 3D vision is actually used) so the 2D
symbolic pipeline keeps running with zero extra dependencies.

Coordinates: our `Scene` is shifted so its floor bbox starts at the origin, while
MuJoCo keeps the export's raw world frame. `mjcf_to_scene(..., return_meta=True)`
hands back `world_offset`; here `world = scene - offset`.
"""
from __future__ import annotations
import math
import threading
import xml.etree.ElementTree as ET

import numpy as np

from .scene import Scene
from .geometry import Pose
from .renderer import Renderer

BALL_RGBA = (0.90, 0.05, 0.05, 1.0)   # vivid red, easy to segment from a gray room
BALL_RADIUS = 0.30


def inject_red_ball(xml: str, world_xy: tuple[float, float], z: float,
                    radius: float = BALL_RADIUS, rgba=BALL_RGBA) -> str:
    """Return MJCF XML with a free-standing red sphere added at `world_xy`,`z`.

    Gizmo exports don't contain our goal object, so we add it ourselves at the
    same spot `mjcf_to_scene` chose, in the MJCF's raw world frame. The geom has
    no collision (contype/conaffinity 0) — it's a visual target, not an obstacle.
    The ball is **emissive** so it reads as a bright, saturated red beacon
    regardless of a scene's lighting/shadows (some exports light it to near-black),
    which keeps colour segmentation reliable across scenes.
    """
    root = ET.fromstring(xml)
    world = root.find(".//worldbody")
    if world is None:
        raise ValueError("MJCF has no <worldbody> to add the ball to")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if asset.find("./material[@name='wander_ball_mat']") is None:
        ET.SubElement(asset, "material", {
            "name": "wander_ball_mat", "rgba": " ".join(f"{c}" for c in rgba),
            "emission": "0.8", "specular": "0.1", "shininess": "0.1",
        })
    wx, wy = world_xy
    body = ET.SubElement(world, "body",
                         {"name": "wander_red_ball", "pos": f"{wx} {wy} {z}"})
    ET.SubElement(body, "geom", {
        "name": "wander_red_ball_geom", "type": "sphere", "size": f"{radius}",
        "material": "wander_ball_mat", "rgba": " ".join(f"{c}" for c in rgba),
        "contype": "0", "conaffinity": "0", "group": "0",
    })
    return ET.tostring(root, encoding="unicode")


class MuJoCoRenderer(Renderer):
    """First-person RGB+depth renderer driven by a Scene-frame `Pose`.

    The agent is a free camera floating at `eye_height` over the floor, looking
    along its heading with a small downward `pitch` (so the floor-resting ball is
    in frame). Horizontal FOV is pinned to 90° to match the symbolic observation's
    field of view, so the two perception modes stay comparable.
    """

    def __init__(self, model, world_offset: tuple[float, float], floor_z: float,
                 width: int = 192, height: int = 144, eye_height: float = 1.4,
                 pitch_deg: float = 8.0, fov_x_deg: float = 90.0,
                 max_depth: float = 10.0):
        import mujoco  # lazy: only needed for 3D vision
        self._mj = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.dx, self.dy = world_offset
        self.floor_z = floor_z
        self.width, self.height = width, height
        self.eye_height = eye_height
        self.pitch_deg = pitch_deg
        self.max_depth = max_depth

        # Free camera FOV comes from the model's global fovy. Set fovy so the
        # horizontal FOV equals fov_x_deg for our aspect ratio.
        aspect = width / height
        self.fov_x = math.radians(fov_x_deg)
        fovy = 2.0 * math.atan(math.tan(self.fov_x / 2.0) / aspect)
        model.vis.global_.fovy = math.degrees(fovy)
        self.fov_y = fovy

        mujoco.mj_forward(model, self.data)
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._depth_on = False
        self._lock = threading.Lock()   # GL context / renderer is not re-entrant

    # --- camera ---
    def _camera(self, pose: Pose):
        mj = self._mj
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        azi = math.degrees(pose.heading)
        elev = -self.pitch_deg
        fa, fe = math.radians(azi), math.radians(elev)
        fwd = np.array([math.cos(fe) * math.cos(fa),
                        math.cos(fe) * math.sin(fa),
                        math.sin(fe)])
        eye = np.array([pose.x - self.dx, pose.y - self.dy,
                        self.floor_z + self.eye_height])
        d = 1.0
        cam.lookat = eye + d * fwd
        cam.distance = d
        cam.azimuth = azi
        cam.elevation = elev
        return cam

    def set_ball(self, world_xy: tuple[float, float], z: float) -> None:
        """Move the red ball to a new world position (no recompile)."""
        mj = self._mj
        bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "wander_red_ball")
        if bid < 0:
            return
        self.model.body_pos[bid] = [world_xy[0], world_xy[1], z]
        mj.mj_forward(self.model, self.data)

    def _render(self, pose: Pose, depth: bool) -> np.ndarray:
        cam = self._camera(pose)
        if depth != self._depth_on:
            (self._renderer.enable_depth_rendering if depth
             else self._renderer.disable_depth_rendering)()
            self._depth_on = depth
        self._renderer.update_scene(self.data, camera=cam)
        return self._renderer.render().copy()

    # --- Renderer interface ---
    def render(self, scene: Scene, pose: Pose) -> np.ndarray:
        """Egocentric RGB (HxWx3 uint8) — satisfies the `Renderer` ABC."""
        with self._lock:
            return self._render(pose, depth=False)

    def render_rgb_depth(self, scene: Scene, pose: Pose):
        """(rgb uint8 HxWx3, depth float32 HxW in metres, capped at max_depth)."""
        with self._lock:
            rgb = self._render(pose, depth=False)
            depth = self._render(pose, depth=True)
        depth = np.where(np.isfinite(depth), depth, self.max_depth)
        return rgb, np.minimum(depth, self.max_depth)


def _ball_visible_from_ring(scene, renderer, ball_xy, ring_dist=2.4, n=8) -> int:
    """How many free positions on a ring around `ball_xy`, facing it, actually
    SEE the ball when rendered — a direct 3D visibility test (color segmentation)."""
    from .perception import detect_ball
    from dataclasses import replace
    probe = replace(scene, ball=tuple(ball_xy))
    seen = 0
    for k in range(n):
        a = 2 * math.pi * k / n
        px, py = ball_xy[0] - ring_dist * math.cos(a), ball_xy[1] - ring_dist * math.sin(a)
        if not probe.is_free(px, py):
            continue
        heading = math.atan2(ball_xy[1] - py, ball_xy[0] - px)
        rgb, _ = renderer.render_rgb_depth(probe, Pose(px, py, heading))
        if detect_ball(rgb) is not None:
            seen += 1
    return seen


def place_visible_ball(scene, renderer, floor_z, ball_radius=BALL_RADIUS,
                       min_views=2):
    """Return a Scene whose ball sits in an open, reachable, *3D-visible* spot.

    Gizmo picks the farthest free corner for our ball, which is often jammed
    against real walls the coarse 2D occupancy doesn't see, so it renders occluded.
    We keep the original spot if it's already visible; otherwise we search free,
    reachable cells, preferring open ones far from the start, and verify each by
    rendering. We place our own goal object, so choosing a fair, findable spot is
    legitimate scene design — the agent still has to search for it.
    """
    from dataclasses import replace
    from .occupancy import OccupancyGrid
    from .distance_field import DistanceField
    from .observation import cast_ray

    dx, dy = renderer.dx, renderer.dy
    if _ball_visible_from_ring(scene, renderer, scene.ball) >= min_views:
        return scene

    grid = OccupancyGrid.from_scene(scene, 0.1)
    sx, sy = scene.agent_start.x, scene.agent_start.y

    def openness(x, y):
        return min(cast_ray(scene, x, y, k * math.pi / 4, 6.0) for k in range(8))

    b = scene.bounds
    cands = []
    y = b.min_y + 0.5
    while y < b.max_y - 0.4:
        x = b.min_x + 0.5
        while x < b.max_x - 0.4:
            if scene.is_free(x, y):
                op = openness(x, y)
                dist = math.hypot(x - sx, y - sy)
                if op >= 1.0 and dist >= 2.0:
                    cands.append((op + 0.15 * dist, x, y))
            x += 0.5
        y += 0.5
    cands.sort(reverse=True)

    for _, x, y in cands[:16]:
        field = DistanceField.from_grid(grid, (x, y))
        if not math.isfinite(field.query(sx, sy)):
            continue
        renderer.set_ball((x - dx, y - dy), floor_z + ball_radius)
        if _ball_visible_from_ring(scene, renderer, (x, y)) >= min_views:
            return replace(scene, ball=(x, y))

    renderer.set_ball((scene.ball[0] - dx, scene.ball[1] - dy), floor_z + ball_radius)
    return scene


def load_mjcf_3d(source: str, *, agent_radius: float = 0.2,
                 ball_radius: float = BALL_RADIUS, relocate_ball: bool = True,
                 return_meta: bool = False, **renderer_kwargs):
    """Load a MuJoCo MJCF into (Scene, MuJoCoRenderer).

    Parses the export into our 2D `Scene` (obstacles + geodesic substrate), injects
    the red ball into the MJCF at the same spot, and builds a renderer that draws
    the agent's first-person view of that exact scene. The 2D `Scene` still drives
    the geodesic reward and occupancy; the renderer only supplies pixels.
    """
    import os
    import mujoco
    from .antim_import import mjcf_to_scene

    xml = source
    if "\n" not in source and os.path.exists(source) and source.endswith(".xml"):
        with open(source) as fh:
            xml = fh.read()

    scene, meta = mjcf_to_scene(xml, agent_radius=agent_radius, return_meta=True)
    dx, dy = meta["world_offset"]
    floor_z = meta["floor_z"]
    ball_world = (scene.ball[0] - dx, scene.ball[1] - dy)

    if not meta["ball_from_mjcf"]:
        xml = inject_red_ball(xml, ball_world, floor_z + ball_radius,
                              radius=ball_radius)
    model = mujoco.MjModel.from_xml_string(xml)
    # Cap depth at the room diagonal so far walls report a real distance, not a
    # tiny default, while the skybox/far-plane still clamps to a finite value.
    if "max_depth" not in renderer_kwargs:
        diag = math.hypot(scene.bounds.max_x - scene.bounds.min_x,
                          scene.bounds.max_y - scene.bounds.min_y)
        renderer_kwargs["max_depth"] = float(max(8.0, round(diag) + 1))
    renderer = MuJoCoRenderer(model, world_offset=(dx, dy), floor_z=floor_z,
                              **renderer_kwargs)
    if relocate_ball and not meta["ball_from_mjcf"]:
        scene = place_visible_ball(scene, renderer, floor_z, ball_radius)
    if return_meta:
        return scene, renderer, meta
    return scene, renderer
