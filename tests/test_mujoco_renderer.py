"""MuJoCo 3D renderer smoke tests against the committed Gizmo train scene.
Skipped automatically where MuJoCo isn't installed (e.g. the 2D-only CI)."""
import math
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from wanderai.mujoco_renderer import load_mjcf_3d, inject_red_ball
from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.policies import OraclePolicy
from wanderai.perception import detect_ball
from wanderai.geometry import Pose

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(HERE, "examples", "js755rrf6gmkwj444nzqh6ermx89394v.xml")
pytestmark = pytest.mark.skipif(not os.path.exists(TRAIN), reason="train scene xml absent")


def test_inject_red_ball_adds_geom_and_material():
    xml = "<mujoco><asset/><worldbody/></mujoco>"
    out = inject_red_ball(xml, (1.0, 2.0), 0.3)
    assert "wander_red_ball_geom" in out and "wander_ball_mat" in out


def test_load_renders_rgb_and_depth():
    scene, renderer = load_mjcf_3d(TRAIN)
    rgb, depth = renderer.render_rgb_depth(scene, scene.agent_start)
    assert rgb.shape == (renderer.height, renderer.width, 3)
    assert rgb.dtype == np.uint8
    assert depth.shape == (renderer.height, renderer.width)
    assert np.all(depth <= renderer.max_depth + 1e-3)
    assert renderer.max_depth >= 8.0


def test_relocated_ball_is_actually_visible():
    scene, renderer = load_mjcf_3d(TRAIN)
    bx, by = scene.ball
    # stand 2.5 m from the ball, facing it; the renderer should show it
    for k in range(8):
        a = k * math.pi / 4
        px, py = bx - 2.5 * math.cos(a), by - 2.5 * math.sin(a)
        if not scene.is_free(px, py):
            continue
        rgb, _ = renderer.render_rgb_depth(scene, Pose(px, py, math.atan2(by - py, bx - px)))
        if detect_ball(rgb) is not None:
            return
    pytest.fail("relocated ball never visible from its surrounding ring")


def test_oracle_converges_with_vision_observation():
    scene, renderer = load_mjcf_3d(TRAIN)
    env = SceneSearchEnv(scene, renderer=renderer,
                         config=EnvConfig(max_steps=400, perception="vision"))
    env.reset()
    oracle = OraclePolicy()
    done = False
    while not done:
        _, _, done, info = env.step(oracle.act(None, env))
    assert info["success"]
    # near-optimal: path length within 1.5x the geodesic optimum
    assert info["path_length"] <= 1.5 * info["optimal"]


def test_vision_obs_text_has_same_shape_as_geometry():
    scene, renderer = load_mjcf_3d(TRAIN)
    env = SceneSearchEnv(scene, renderer=renderer,
                         config=EnvConfig(perception="vision"))
    _, info = env.reset()
    txt = info["obs_text"]
    assert "Red ball:" in txt and "Clearance" in txt and "Explored" in txt
