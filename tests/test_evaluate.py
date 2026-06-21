from wanderai.evaluate import evaluate
from wanderai.environment import EnvConfig
from wanderai.scene_gen import make_split
from wanderai.policies import OraclePolicy, RandomPolicy


def test_evaluate_returns_summary_and_results():
    _, test = make_split(0, 4, seed=11)
    out = evaluate(RandomPolicy(seed=0), test, EnvConfig(max_steps=40))
    assert out["summary"]["n"] == 4
    assert len(out["results"]) == 4
    assert 0.0 <= out["summary"]["spl"] <= 1.0


def test_oracle_beats_random_on_heldout():
    _, test = make_split(0, 6, seed=12)
    cfg = EnvConfig(max_steps=400)
    oracle = evaluate(OraclePolicy(), test, cfg)["summary"]
    random = evaluate(RandomPolicy(seed=0), test, cfg)["summary"]
    assert oracle["spl"] >= random["spl"]            # ceiling >= floor
    assert oracle["success_rate"] >= random["success_rate"]
