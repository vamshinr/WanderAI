import math
from wanderai.rft import (episode_reward, run_scored, group_advantages, grpo_preview,
                          single_step_reward, scene_to_dict, scene_from_dict, build_dataset)
from wanderai.environment import EnvConfig, Action
from wanderai.geometry import AABB, Pose
from wanderai.scene import Scene, default_scene
from wanderai.scene_gen import make_split
from wanderai.policies import OraclePolicy, RandomPolicy


def test_reward_is_bounded():
    r = episode_reward(optimal=5.0, final_geodesic=0.0, path_length=5.0, success=True, steps=20)
    assert 0.0 <= r.reward <= 1.0


def test_success_beats_no_progress():
    reached = episode_reward(5.0, 0.0, 5.2, success=True, steps=22)
    stuck = episode_reward(5.0, 5.0, 0.0, success=False, steps=400)   # never moved closer
    assert reached.reward > stuck.reward
    assert stuck.reward == 0.0


def test_partial_progress_gives_partial_reward():
    half = episode_reward(4.0, 2.0, 2.0, success=False, steps=50)     # closed half the distance
    assert 0.0 < half.reward < 1.0


def test_oracle_scores_high_random_low_on_default():
    cfg = EnvConfig(max_steps=400)
    assert run_scored(default_scene(), OraclePolicy(), cfg).reward > 0.8
    # random often makes little net progress
    assert run_scored(default_scene(), RandomPolicy(seed=0), cfg).reward < \
           run_scored(default_scene(), OraclePolicy(), cfg).reward


def test_group_advantages_standardize():
    adv = group_advantages([0.0, 0.5, 1.0])
    assert abs(sum(adv)) < 1e-9                 # mean zero
    assert adv[0] < 0 < adv[2]                  # worst suppressed, best reinforced


def test_group_advantages_zero_variance():
    assert group_advantages([0.4, 0.4, 0.4]) == [0.0, 0.0, 0.0]


def test_scene_serialization_roundtrip():
    s = default_scene()
    s2 = scene_from_dict(scene_to_dict(s))
    assert s2.ball == s.ball and len(s2.obstacles) == len(s.obstacles)
    assert s2.bounds.max_x == s.bounds.max_x


def test_single_step_reward_forward_progress():
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 1.0), Pose(1.0, 1.0, 0.0), 0.2)  # facing ball (+x)
    fwd = single_step_reward(s, (1.0, 1.0, 0.0), Action.MOVE_FORWARD)
    assert fwd > 0.5                        # moving toward the ball is rewarded


def test_single_step_reward_distinguishes_turn_direction():
    # Ball is to the +x; agent faces +y. Turning RIGHT (toward the ball) must beat
    # turning LEFT (away) — the fix that makes turn direction learnable.
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 1.0), Pose(1.0, 1.0, math.pi / 2), 0.2)
    pose = (1.0, 1.0, math.pi / 2)
    right = single_step_reward(s, pose, Action.TURN_RIGHT)
    left = single_step_reward(s, pose, Action.TURN_LEFT)
    assert right > 0.5 > left


def test_single_step_reward_success_and_collision():
    reach = Scene(AABB(0, 0, 6, 6), [], (1.3, 1.0), Pose(1.0, 1.0, 0.0), 0.2)
    assert single_step_reward(reach, (1.0, 1.0, 0.0), Action.MOVE_FORWARD) == 1.0
    wall = Scene(AABB(0, 0, 6, 6), [AABB(1.3, 0, 1.6, 6)], (5, 1), Pose(1.0, 1.0, 0.0), 0.2)
    assert single_step_reward(wall, (1.0, 1.0, 0.0), Action.MOVE_FORWARD) == 0.0


def test_build_dataset_rows_have_prompt_and_state():
    _, scenes = make_split(0, 3, seed=9)
    rows = build_dataset(scenes, [OraclePolicy(), RandomPolicy(seed=0)], max_steps=10)
    assert len(rows) > 0
    r = rows[0]
    assert "Red ball" in r["obs"] and "scene" in r and len(r["pose"]) == 3
    # the verifier can score the captured state
    assert 0.0 <= single_step_reward(scene_from_dict(r["scene"]), r["pose"], Action.MOVE_FORWARD) <= 1.0


def test_grpo_preview_produces_signal_with_stochastic_policy():
    _, test = make_split(0, 1, seed=3)
    out = grpo_preview(test[0], lambda: RandomPolicy(seed=None), group_size=5,
                       config=EnvConfig(max_steps=80))
    assert len(out["rewards"]) == 5
    assert len(out["advantages"]) == 5
    assert abs(sum(out["advantages"])) < 1e-6   # advantages are mean-centered
