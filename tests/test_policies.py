from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode


def test_oracle_solves_default_scene():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=400))
    res = run_episode(env, OraclePolicy())
    assert res.success
    assert res.path_length <= res.optimal * 1.6   # near-optimal route


def test_random_policy_runs_without_error():
    env = SceneSearchEnv(default_scene(), config=EnvConfig(max_steps=50))
    res = run_episode(env, RandomPolicy(seed=0))
    assert res.steps <= 50
