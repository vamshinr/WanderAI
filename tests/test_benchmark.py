import json
import math

from wanderai.benchmark import (aggregate, bootstrap_ci, run_policy,
                                save_report, to_markdown)
from wanderai.environment import EnvConfig
from wanderai.metrics import EpisodeResult
from wanderai.policies import OraclePolicy
from wanderai.scene_gen import make_split


def test_bootstrap_ci_deterministic_and_covers_mean():
    vals = [0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    lo1, hi1 = bootstrap_ci(vals, seed=0)
    lo2, hi2 = bootstrap_ci(vals, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= sum(vals) / len(vals) <= hi1
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_aggregate_and_markdown_roundtrip():
    results = [
        EpisodeResult(True, 5.0, 6.0, 24, final_geodesic=0.1, collisions=1,
                      visited_cells=30),
        EpisodeResult(False, 4.0, 10.0, 400, final_geodesic=2.0, collisions=5,
                      visited_cells=50),
    ]
    row = aggregate("test", results)
    assert row["episodes"] == 2
    assert row["success_rate"] == 0.5
    assert 0 < row["spl"] < 1
    md = to_markdown([row], caption="caption")
    assert "| test |" in md and "caption" in md


def test_run_policy_on_oracle_succeeds(tmp_path):
    _, test = make_split(0, 2, seed=5)
    results = run_policy(lambda seed: OraclePolicy(), test,
                         EnvConfig(max_steps=400))
    assert all(r.success for r in results)
    row = aggregate("oracle", results)
    out = tmp_path / "report.json"
    save_report(str(out), [row], {"benchmark": "unit"})
    payload = json.loads(out.read_text())
    assert payload["meta"]["benchmark"] == "unit"
    assert payload["results"][0]["success_rate"] == 1.0
