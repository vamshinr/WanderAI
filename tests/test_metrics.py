import math
from wanderai.metrics import EpisodeResult, spl, summarize


def test_spl_perfect_path():
    r = EpisodeResult(success=True, optimal=4.0, path_length=4.0, steps=16)
    assert math.isclose(spl([r]), 1.0)


def test_spl_penalizes_detour():
    r = EpisodeResult(success=True, optimal=4.0, path_length=8.0, steps=32)
    assert math.isclose(spl([r]), 0.5)


def test_failed_episode_zero():
    r = EpisodeResult(success=False, optimal=4.0, path_length=2.0, steps=200)
    assert spl([r]) == 0.0


def test_summarize():
    rs = [EpisodeResult(True, 4, 4, 16), EpisodeResult(False, 4, 3, 200)]
    out = summarize(rs)
    assert math.isclose(out["success_rate"], 0.5)
    assert math.isclose(out["spl"], 0.5)
    assert out["mean_steps"] == 16
