"""Benchmark harness — seeded, reproducible policy comparisons with CIs.

Everything a claim needs: fixed scene splits (`make_split`), per-episode seeds,
success rate / SPL / SoftSPL / DTS / collisions / coverage per policy, and
bootstrap 95% confidence intervals over episodes (percentile method — the
practice recommended for small-sample RL evaluation by Agarwal et al. 2021).
Results are written as both JSON (machine-readable, with full metadata: git
revision, config, seeds) and a markdown table for the report.
"""
from __future__ import annotations

import json
import math
import subprocess
import time

import numpy as np

from .environment import SceneSearchEnv, EnvConfig
from .metrics import EpisodeResult, spl, soft_spl, distance_to_success
from .policies import run_episode


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile-bootstrap CI for the mean of `values`."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return (float(lo), float(hi))


def _spl_terms(results):
    return [r.optimal / max(r.path_length, r.optimal)
            if (r.success and r.optimal > 0) else 0.0 for r in results]


def run_policy(policy_factory, scenes, config: EnvConfig,
               episodes_per_scene: int = 1, base_seed: int = 0):
    """Run `policy_factory(seed)` over every scene x episode; fresh policy per
    episode so no state leaks across rooms."""
    results: list[EpisodeResult] = []
    for si, scene in enumerate(scenes):
        env = SceneSearchEnv(scene, config=config)
        for ep in range(episodes_per_scene):
            policy = policy_factory(base_seed + 1000 * si + ep)
            results.append(run_episode(env, policy))
    return results


def aggregate(name: str, results: list[EpisodeResult]) -> dict:
    succ = [r for r in results if r.success]
    sr = [1.0 if r.success else 0.0 for r in results]
    terms = _spl_terms(results)
    return {
        "policy": name,
        "episodes": len(results),
        "success_rate": float(np.mean(sr)),
        "success_ci": bootstrap_ci(sr),
        # Point estimate from the SAME per-episode terms the CI resamples, so
        # the published number can never sit outside its own interval.
        "spl": float(np.mean(terms)) if terms else 0.0,
        "spl_ci": bootstrap_ci(terms),
        "soft_spl": float(soft_spl(results)),
        "dts": float(distance_to_success(results)),
        "mean_steps_success": (float(np.mean([r.steps for r in succ]))
                               if succ else float("nan")),
        "mean_collisions": float(np.mean([r.collisions for r in results])),
        "mean_coverage": float(np.mean([r.visited_cells for r in results])),
    }


def to_markdown(rows: list[dict], caption: str = "") -> str:
    def ci(row, key, ci_key):
        lo, hi = row[ci_key]
        return f"{row[key]:.3f} [{lo:.3f}, {hi:.3f}]"

    lines = []
    if caption:
        lines.append(f"**{caption}**\n")
    lines.append("| Policy | Episodes | Success rate [95% CI] | SPL [95% CI] | "
                 "SoftSPL | DTS (m) | Steps (succ.) | Collisions | Coverage |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        steps = ("-" if math.isnan(r["mean_steps_success"])
                 else f"{r['mean_steps_success']:.0f}")
        lines.append(
            f"| {r['policy']} | {r['episodes']} | "
            f"{ci(r, 'success_rate', 'success_ci')} | "
            f"{ci(r, 'spl', 'spl_ci')} | {r['soft_spl']:.3f} | {r['dts']:.2f} | {steps} | "
            f"{r['mean_collisions']:.1f} | {r['mean_coverage']:.1f} |")
    return "\n".join(lines) + "\n"


def git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _json_safe(value):
    """NaN/inf are invalid JSON (RFC 8259) — map them to None so strict
    consumers (browsers, jq) can parse the report."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_report(path_json: str, rows: list[dict], meta: dict):
    payload = {"meta": {**meta, "git": git_revision(),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                                   time.gmtime())},
               "results": rows}
    with open(path_json, "w") as fh:
        json.dump(_json_safe(payload), fh, indent=1, allow_nan=False)
