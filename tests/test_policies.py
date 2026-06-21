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


_IMPORTED = """<mujoco><worldbody>
  <geom type="plane" size="3 3 0.1" pos="0 0 0"/>
  <body pos="1 0 0.25"><geom type="box" size="0.3 0.5 0.25"/></body>
  <body pos="-1.5 1 0.25"><geom type="box" size="0.4 0.4 0.25"/></body>
  <body name="red_ball" pos="2 -2 0.1"><geom type="sphere" size="0.1" rgba="1 0 0 1"/></body>
</worldbody></mujoco>"""


def test_oracle_solves_imported_scene():
    # Regression: the old oracle wedged against an inflated obstacle here and
    # spammed MOVE_FORWARD forever. The lookahead oracle must solve it.
    from wanderai.antim_import import mjcf_to_scene
    env = SceneSearchEnv(mjcf_to_scene(_IMPORTED), config=EnvConfig(max_steps=400))
    res = run_episode(env, OraclePolicy())
    assert res.success


def test_oracle_solves_most_generated_scenes():
    from wanderai.scene_gen import make_split
    train, _ = make_split(8, 0, seed=5)
    solved = sum(run_episode(SceneSearchEnv(s, config=EnvConfig(max_steps=400)),
                             OraclePolicy()).success for s in train)
    assert solved >= 7   # robust greedy descent on procedural rooms
