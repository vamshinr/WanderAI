import math

import numpy as np

from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.geometry import AABB, Pose
from wanderai.observation import Observation
from wanderai.policies import run_episode
from wanderai.rl_local import (FEATURE_DIM, PolicyNet, ReinforceTrainer,
                               RLConfig, TrainedLocalPolicy, _EpisodeTrace,
                               featurize)
from wanderai.scene import Scene, default_scene


def _obs(visible=False, bearing=None, distance=None):
    return Observation(
        ball_visible=visible, ball_bearing=bearing, ball_distance=distance,
        clearance={"left": 3.0, "center": 6.0, "right": 1.5},
        recent_actions=[], explored={"left": True, "center": False, "right": False},
        n_visited=4)


def test_featurize_dim_and_zeros():
    x = featurize(_obs(), last_action=None, hint=None)
    assert x.shape == (FEATURE_DIM,) and x.dtype == np.float32
    # Ball invisible -> indicator and bearing/distance block all zero.
    assert np.all(x[0:4] == 0.0)
    # No last action, no hint -> those blocks all zero; bias is 1.
    assert np.all(x[10:18] == 0.0)
    assert x[18] == 1.0
    # Clearance normalized by 6 and clipped.
    assert math.isclose(x[4], 0.5) and math.isclose(x[5], 1.0)
    assert math.isclose(x[6], 1.5 / 6.0)
    assert x[7] == 1.0 and x[8] == 0.0


def test_featurize_visible_hint_and_determinism():
    obs = _obs(visible=True, bearing=math.pi / 4, distance=12.0)
    hint = {"direction": "behind", "distance": 20.0, "bearing": 3.0, "size": 5}
    x = featurize(obs, last_action=2, hint=hint)
    assert x[0] == 1.0
    assert math.isclose(x[1], math.sin(math.pi / 4), rel_tol=1e-6)
    assert math.isclose(x[2], math.cos(math.pi / 4), rel_tol=1e-6)
    assert x[3] == 1.0                       # distance capped at 8m
    assert list(x[10:13]) == [0.0, 0.0, 1.0]  # last action one-hot
    assert list(x[13:17]) == [0.0, 0.0, 0.0, 1.0]  # "behind"
    assert x[17] == 1.0                      # hint distance capped
    assert np.array_equal(x, featurize(obs, last_action=2, hint=hint))


def test_policynet_roundtrip_preserves_argmax():
    net = PolicyNet(hidden=16, seed=5)
    # The output layer starts at zero (uniform policy); give it real weights so
    # the round-trip check exercises non-trivial argmax decisions.
    wrng = np.random.default_rng(9)
    net.W2 = wrng.normal(0, 0.5, net.W2.shape)
    net.b2 = wrng.normal(0, 0.5, net.b2.shape)
    clone = PolicyNet.from_dict(net.to_dict())
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-1, 1, FEATURE_DIM).astype(np.float32)
        assert net.argmax(x) == clone.argmax(x)
        assert np.allclose(net.logits(x), clone.logits(x))


def test_identical_returns_give_zero_update():
    """GRPO advantage: a group with identical returns carries no learning signal.
    With entropy off, the gradient is exactly zero and the Adam step is a no-op."""
    scene = default_scene()
    cfg = RLConfig(iters=1, group_size=4, entropy_coef=0.0, seed=1)
    trainer = ReinforceTrainer([scene], cfg)
    before = [p.copy() for p in trainer.net.params()]

    rng = np.random.default_rng(2)
    episodes = []
    for _ in range(4):
        e = _EpisodeTrace(ret=3.7)          # identical returns across the group
        for _ in range(5):
            e.features.append(rng.uniform(-1, 1, FEATURE_DIM).astype(np.float32))
            e.actions.append(int(rng.integers(0, 3)))
            e.logps.append(0.0)
        episodes.append(e)
    stats = trainer.update_from_group(episodes)

    assert math.isfinite(stats["grad_norm"])
    assert stats["grad_norm"] < 1e-12
    for p, q in zip(trainer.net.params(), before):
        assert np.all(np.abs(p - q) < 1e-9)


def test_learns_trivial_scene():
    """Empty room, ball 2.5m directly ahead: training must (a) make the argmax
    policy succeed and (b) raise the mean return late vs early in training."""
    scene = Scene(bounds=AABB(0, 0, 6, 6), obstacles=[], ball=(3.5, 3.0),
                  agent_start=Pose(1.0, 3.0, 0.0), agent_radius=0.2)
    cfg = RLConfig(iters=60, group_size=6, max_steps=30, seed=3)
    trainer = ReinforceTrainer([scene], cfg)
    history = trainer.train()

    first = np.mean([h["mean_return"] for h in history[:20]])
    last = np.mean([h["mean_return"] for h in history[-20:]])
    assert last > first, f"no improvement: first20={first:.3f} last20={last:.3f}"

    policy = TrainedLocalPolicy(trainer.net)
    result = run_episode(SceneSearchEnv(scene, config=EnvConfig(max_steps=30)),
                         policy)
    assert result.success


def test_trained_policy_interface_runs():
    net = PolicyNet(hidden=8, seed=0)
    for use_hint in (False, True):
        policy = TrainedLocalPolicy(net.to_dict(), use_hint=use_hint)
        env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=40))
        result = run_episode(env, policy)
        assert result.steps > 0
        # Self-reset: a fresh episode rebuilds internal state from scratch.
        env2 = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=5))
        env2.reset()
        policy.act(None, env2)
        if use_hint:
            assert policy.map.updates == 1
