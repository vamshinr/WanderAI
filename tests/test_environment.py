import math
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
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 1.0), Pose(1.0, 1.0, 0.0), 0.2)
    env = SceneSearchEnv(s, config=EnvConfig(cell_size=0.1))
    env.reset()
    _, reward, _, info = env.step(Action.MOVE_FORWARD)
    assert reward > 0           # progress term beats time penalty
    assert not info["collision"]


def test_collision_blocks_and_penalizes():
    # Wall directly ahead; forward move is rejected, pose unchanged, collision flagged.
    s = Scene(AABB(0, 0, 6, 6), [AABB(1.3, 0.0, 1.6, 6.0)], (5, 1), Pose(1.0, 1.0, 0.0), 0.2)
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
    s = Scene(AABB(0, 0, 6, 6), [], (1.3, 1.0), Pose(1.0, 1.0, 0.0), 0.2)
    env = SceneSearchEnv(s, config=EnvConfig(step_size=0.25, success_radius=0.3, goal_reward=10.0))
    env.reset()
    done = False
    info = {}
    for _ in range(5):
        _, reward, done, info = env.step(Action.MOVE_FORWARD)
        if done:
            break
    assert done and info["success"] and reward > 5
