from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EpisodeResult:
    success: bool
    optimal: float       # optimal geodesic distance start -> ball
    path_length: float   # distance the agent actually walked
    steps: int


def spl(results: list[EpisodeResult]) -> float:
    """Success weighted by Path Length: (1/N) sum_i S_i * l_i / max(p_i, l_i).
    Failed episodes (and degenerate optimal<=0) contribute 0."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.success and r.optimal > 0:
            total += r.optimal / max(r.path_length, r.optimal)
    return total / len(results)


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"success_rate": 0.0, "spl": 0.0, "mean_steps": 0.0}
    succ = [r for r in results if r.success]
    return {
        "success_rate": len(succ) / n,
        "spl": spl(results),
        "mean_steps": sum(r.steps for r in succ) / len(succ) if succ else 0.0,
    }
