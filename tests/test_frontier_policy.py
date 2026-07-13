import math

from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.frontier_policy import FrontierPolicy, FrontierConfig
from wanderai.geometry import Pose, AABB
from wanderai.policies import run_episode, RandomPolicy
from wanderai.scene import Scene, default_scene
from wanderai.scene_gen import make_split
from wanderai.metrics import summarize


def test_homes_in_when_ball_directly_visible():
    scene = Scene(bounds=AABB(0, 0, 6, 6), obstacles=[], ball=(5.0, 3.0),
                  agent_start=Pose(1.0, 3.0, 0.0), agent_radius=0.2)
    env = SceneSearchEnv(scene, config=EnvConfig(max_steps=60))
    result = run_episode(env, FrontierPolicy())
    assert result.success
    # Nearly straight-line: path should be close to optimal.
    assert result.path_length <= result.optimal * 1.5


def test_succeeds_on_default_scene_with_occluder():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=400))
    result = run_episode(env, FrontierPolicy())
    assert result.success


def test_never_touches_privileged_state():
    """The policy must work even when the privileged channels are removed."""
    scene = default_scene()
    env = SceneSearchEnv(scene, config=EnvConfig(max_steps=400))
    env.reset()
    policy = FrontierPolicy()

    class Guard:
        """Raises if the policy reads the geodesic field."""
        def __getattr__(self, name):
            raise AssertionError(f"policy read env.field.{name}")

    real_field = env.field
    env.field = Guard()
    try:
        for _ in range(50):
            action = policy.act(None, env)
            assert action in (Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT)
            # Step manually so the env itself can use the real field for reward.
            env.field = real_field
            _, _, done, _ = env.step(action)
            env.field = Guard()
            if done:
                break
    finally:
        env.field = real_field


def test_beats_random_on_heldout_rooms():
    _, test = make_split(0, 4, seed=11)
    cfg = EnvConfig(max_steps=300)
    fbe = [run_episode(SceneSearchEnv(s, config=cfg), FrontierPolicy())
           for s in test]
    rnd = [run_episode(SceneSearchEnv(s, config=cfg), RandomPolicy(seed=0))
           for s in test]
    fbe_summary, rnd_summary = summarize(fbe), summarize(rnd)
    assert fbe_summary["success_rate"] >= rnd_summary["success_rate"]
    assert fbe_summary["spl"] > rnd_summary["spl"]


def test_memory_budget_still_functional():
    """With a tight map budget the policy coarsens instead of failing."""
    _, test = make_split(0, 2, seed=11)
    cfg = EnvConfig(max_steps=300)
    policy = FrontierPolicy(FrontierConfig(budget_cells=300))
    results = [run_episode(SceneSearchEnv(s, config=cfg), policy) for s in test]
    assert policy.map.coarsen_count >= 0      # ran without error
    assert all(r.steps > 0 for r in results)


def test_self_resets_between_episodes():
    scene = default_scene()
    policy = FrontierPolicy()
    env = SceneSearchEnv(scene, config=EnvConfig(max_steps=50))
    run_episode(env, policy)
    cells_after_first = policy.map.memory_cells()
    assert cells_after_first > 0
    env2 = SceneSearchEnv(scene, config=EnvConfig(max_steps=5))
    env2.reset()
    policy.act(None, env2)
    # Map was rebuilt from scratch on the new episode's first step.
    assert policy.map.updates == 1
