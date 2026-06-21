import math
from wanderai.rft import (episode_reward, run_scored, group_advantages, grpo_preview)
from wanderai.environment import EnvConfig
from wanderai.scene import default_scene
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


def test_grpo_preview_produces_signal_with_stochastic_policy():
    _, test = make_split(0, 1, seed=3)
    out = grpo_preview(test[0], lambda: RandomPolicy(seed=None), group_size=5,
                       config=EnvConfig(max_steps=80))
    assert len(out["rewards"]) == 5
    assert len(out["advantages"]) == 5
    assert abs(sum(out["advantages"])) < 1e-6   # advantages are mean-centered
