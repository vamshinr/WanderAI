import math
from wanderai.scene import Scene, default_scene
from wanderai.geometry import AABB, Pose
from wanderai.environment import SceneSearchEnv, EnvConfig, Action


def test_reset_includes_obs_text():
    env = SceneSearchEnv(default_scene())
    obs, info = env.reset()
    assert "obs_text" in info
    assert isinstance(info["obs_text"], str) and len(info["obs_text"]) > 0
    # No moves yet at reset.
    assert "Recent moves: none" in info["obs_text"]


def test_obs_text_reports_visible_ball():
    # Ball straight ahead, clear line of sight.
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 3.0), Pose(1.0, 3.0, 0.0), 0.2)
    env = SceneSearchEnv(s)
    _, info = env.reset()
    assert "VISIBLE" in info["obs_text"]


def test_history_tracks_recent_actions():
    env = SceneSearchEnv(default_scene())
    env.reset()
    env.step(Action.TURN_LEFT)
    _, _, _, info = env.step(Action.TURN_RIGHT)
    assert "TURN_RIGHT" in info["obs_text"]
    assert env.history[-2:] == [Action.TURN_LEFT, Action.TURN_RIGHT]


def test_reset_clears_history():
    env = SceneSearchEnv(default_scene())
    env.reset()
    env.step(Action.TURN_LEFT)
    env.reset()
    assert env.history == []


def test_text_observation_method_matches_info():
    env = SceneSearchEnv(default_scene())
    _, info = env.reset()
    assert env.text_observation() == info["obs_text"]
