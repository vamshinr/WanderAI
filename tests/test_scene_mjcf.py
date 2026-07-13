import math
import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "osmesa")

from wanderai.scene import default_scene
from wanderai.scene_mjcf import scene_to_mjcf, _obstacle_height


def test_mjcf_contains_every_obstacle_and_the_ball():
    scene = default_scene()
    xml = scene_to_mjcf(scene)
    assert xml.count('name="obstacle_') == len(scene.obstacles)
    assert "wander_red_ball" in xml
    for side in ("wall_s", "wall_n", "wall_w", "wall_e"):
        assert side in xml


def test_obstacle_heights_deterministic_and_bounded():
    scene = default_scene()
    for ob in scene.obstacles:
        h1, h2 = _obstacle_height(ob), _obstacle_height(ob)
        assert h1 == h2
        assert 0.45 <= h1 <= 1.9


def test_rendered_geometry_matches_collision_world():
    """The 3D room's depth must agree with the 2D ray-cast ground truth — the
    property the vision benchmark rests on."""
    mujoco = pytest.importorskip("mujoco")
    from wanderai.scene_mjcf import scene_renderer_3d
    from wanderai.mapping import depth_strip, obstacle_strip_from_depth

    scene, renderer = scene_renderer_3d(default_scene())
    with renderer:
        rgb, depth = renderer.render_rgb_depth(scene, scene.agent_start)
        geo = depth_strip(scene, scene.agent_start, max_range=6.0)
        vis = obstacle_strip_from_depth(
            depth, fov_x=renderer.fov_x, fov_y=renderer.fov_y,
            eye_height=renderer.eye_height,
            pitch_rad=math.radians(renderer.pitch_deg), max_range=6.0)
    mae = np.mean([abs(dg - dv) for (_, dg), (_, dv) in zip(geo, vis)])
    assert mae < 0.4, f"vision strip disagrees with geometry: MAE {mae:.2f} m"


def test_ball_is_detectable_in_render():
    mujoco = pytest.importorskip("mujoco")
    from wanderai.scene_mjcf import scene_renderer_3d
    from wanderai.perception import detect_ball
    from wanderai.geometry import Pose

    scene, renderer = scene_renderer_3d(default_scene())
    with renderer:
        bx, by = scene.ball
        # Stand 2.5m from the ball, facing it, from a free spot.
        for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            px, py = bx - 2.5 * math.cos(ang), by - 2.5 * math.sin(ang)
            if not scene.is_free(px, py):
                continue
            pose = Pose(px, py, math.atan2(by - py, bx - px))
            rgb, _ = renderer.render_rgb_depth(scene, pose)
            if detect_ball(rgb) is not None:
                return
    pytest.fail("red ball never detected from any free ring position")
