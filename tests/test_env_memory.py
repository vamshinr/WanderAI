from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig, Action


def test_env_tracks_visited_cells_growing():
    env = SceneSearchEnv(default_scene())
    env.reset()
    start = len(env.visited)
    for _ in range(8):
        env.step(Action.MOVE_FORWARD)
    assert len(env.visited) > start         # exploring adds cells


def test_reset_clears_visited():
    env = SceneSearchEnv(default_scene())
    env.reset()
    for _ in range(5):
        env.step(Action.MOVE_FORWARD)
    env.reset()
    assert len(env.visited) == 1            # just the start cell


def test_obs_text_reports_explored_after_moving():
    env = SceneSearchEnv(default_scene())
    env.reset()
    for _ in range(6):
        env.step(Action.MOVE_FORWARD)
    _, _, _, info = env.step(Action.MOVE_FORWARD)
    assert "Explored" in info["obs_text"]
    # After walking forward, the cell behind (and likely center-ahead path) is explored.
    assert "cells seen" in info["obs_text"]