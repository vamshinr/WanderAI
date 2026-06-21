import math
import pytest
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
    s = Scene(AABB(0, 0, 6, 6), [AABB(1.3, 0.0, 1.6, 6.0)], (0.5, 1), Pose(1.0, 1.0, 0.0), 0.2)
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


def test_forward_move_sweeps_collision_between_cells():
    s = Scene(
        AABB(0, 0, 3, 2),
        [AABB(0.95, 0.8, 1.05, 1.2)],
        (2.5, 1.0),
        Pose(0.5, 1.0, 0.0),
        0.0,
    )
    env = SceneSearchEnv(s, config=EnvConfig(cell_size=0.05, step_size=1.0, success_radius=0.1))
    env.reset()
    before = (env.pose.x, env.pose.y)

    _, _, _, info = env.step(Action.MOVE_FORWARD)

    assert info["collision"]
    assert (env.pose.x, env.pose.y) == before
    assert info["path_length"] == 0.0


def test_reset_rejects_blocked_start():
    s = Scene(
        AABB(0, 0, 2, 2),
        [AABB(0.4, 0.4, 0.6, 0.6)],
        (1.5, 1.5),
        Pose(0.5, 0.5, 0.0),
        0.0,
    )

    with pytest.raises(ValueError, match="start"):
        SceneSearchEnv(s, config=EnvConfig(cell_size=0.1)).reset()


def test_reset_rejects_blocked_goal():
    s = Scene(
        AABB(0, 0, 2, 2),
        [AABB(1.4, 1.4, 1.6, 1.6)],
        (1.5, 1.5),
        Pose(0.5, 0.5, 0.0),
        0.0,
    )

    with pytest.raises(ValueError, match="goal"):
        SceneSearchEnv(s, config=EnvConfig(cell_size=0.1)).reset()


def test_reset_rejects_out_of_bounds_start_or_goal():
    out_of_bounds_start = Scene(AABB(0, 0, 2, 2), [], (1.5, 1.5), Pose(-0.1, 0.5, 0.0), 0.0)
    out_of_bounds_goal = Scene(AABB(0, 0, 2, 2), [], (2.5, 1.5), Pose(0.5, 0.5, 0.0), 0.0)

    with pytest.raises(ValueError, match="start"):
        SceneSearchEnv(out_of_bounds_start, config=EnvConfig(cell_size=0.1)).reset()
    with pytest.raises(ValueError, match="goal"):
        SceneSearchEnv(out_of_bounds_goal, config=EnvConfig(cell_size=0.1)).reset()


def test_reset_rejects_unreachable_start_goal_pair():
    s = Scene(
        AABB(0, 0, 4, 4),
        [AABB(1.9, 0.0, 2.1, 4.0)],
        (3.0, 1.0),
        Pose(1.0, 1.0, 0.0),
        0.0,
    )

    with pytest.raises(ValueError, match="unreachable"):
        SceneSearchEnv(s, config=EnvConfig(cell_size=0.1)).reset()


def test_success_requires_geodesic_distance_within_radius():
    s = Scene(
        AABB(0, 0, 4, 4),
        [AABB(2.0, 0.0, 2.1, 3.0)],
        (2.25, 1.0),
        Pose(1.75, 1.0, 0.0),
        0.0,
    )
    env = SceneSearchEnv(s, config=EnvConfig(cell_size=0.05, success_radius=0.6))
    env.reset()

    _, _, done, info = env.step(Action.TURN_LEFT)

    assert info["geodesic"] > env.config.success_radius
    assert not info["success"]
    assert not done


def test_step_before_reset_raises():
    env = SceneSearchEnv(default_scene())

    with pytest.raises(RuntimeError, match="reset"):
        env.step(Action.TURN_LEFT)


def test_step_after_done_raises():
    s = Scene(AABB(0, 0, 3, 3), [], (1.25, 1.0), Pose(1.0, 1.0, 0.0), 0.0)
    env = SceneSearchEnv(s, config=EnvConfig(step_size=0.25, success_radius=0.1))
    env.reset()
    _, _, done, info = env.step(Action.MOVE_FORWARD)
    assert done and info["success"]

    with pytest.raises(RuntimeError, match="done"):
        env.step(Action.TURN_LEFT)


def test_invalid_action_raises():
    env = SceneSearchEnv(default_scene())
    env.reset()

    with pytest.raises(ValueError, match="invalid action"):
        env.step(99)
