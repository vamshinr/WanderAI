from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class EpisodeResult:
    success: bool
    optimal: float       # optimal geodesic distance start -> ball
    path_length: float   # distance the agent actually walked
    steps: int
    # Extended fields (default-compatible with older call sites):
    final_geodesic: float = math.inf   # geodesic distance to goal at episode end
    collisions: int = 0                # blocked MOVE_FORWARD attempts
    visited_cells: int = 0             # distinct coarse cells visited (coverage)


def spl(results: list[EpisodeResult]) -> float:
    """Success weighted by Path Length: (1/N) sum_i S_i * l_i / max(p_i, l_i).
    Failed episodes (and degenerate optimal<=0) contribute 0.
    (Anderson et al. 2018, "On Evaluation of Embodied Navigation Agents".)"""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.success and r.optimal > 0:
            total += r.optimal / max(r.path_length, r.optimal)
    return total / len(results)


def soft_spl(results: list[EpisodeResult]) -> float:
    """SoftSPL (Datta et al. 2020): replaces the binary success indicator with
    geodesic *progress* toward the goal, so partial progress is credited:
      (1/N) sum_i max(0, 1 - d_T/l_i) * l_i / max(p_i, l_i).
    Episodes with unknown final distance or degenerate optimal contribute 0."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.optimal > 0 and math.isfinite(r.final_geodesic):
            progress = max(0.0, 1.0 - r.final_geodesic / r.optimal)
            total += progress * (r.optimal / max(r.path_length, r.optimal))
    return total / len(results)


def distance_to_success(results: list[EpisodeResult]) -> float:
    """Mean geodesic distance remaining at episode end (0 for successes).
    Lower is better; complements SPL for all-fail comparisons. A failure with
    no usable final distance counts as its full start distance (no progress)
    rather than being dropped — silently excluding such episodes would bias
    the metric toward 0 for exactly the worst outcomes."""
    vals = []
    for r in results:
        if r.success:
            vals.append(0.0)
        elif math.isfinite(r.final_geodesic):
            vals.append(r.final_geodesic)
        elif math.isfinite(r.optimal):
            vals.append(r.optimal)
    return sum(vals) / len(vals) if vals else math.inf


def summarize(results: list[EpisodeResult]) -> dict:
    n = len(results)
    if n == 0:
        return {"success_rate": 0.0, "spl": 0.0, "soft_spl": 0.0,
                "dts": math.inf, "mean_steps": 0.0, "mean_collisions": 0.0,
                "mean_coverage": 0.0}
    succ = [r for r in results if r.success]
    return {
        "success_rate": len(succ) / n,
        "spl": spl(results),
        "soft_spl": soft_spl(results),
        "dts": distance_to_success(results),
        "mean_steps": sum(r.steps for r in succ) / len(succ) if succ else 0.0,
        "mean_collisions": sum(r.collisions for r in results) / n,
        "mean_coverage": sum(r.visited_cells for r in results) / n,
    }
