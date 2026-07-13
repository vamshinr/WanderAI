"""3D vision benchmark: matched procedural MuJoCo rooms, pixels-only sensing.

    MUJOCO_GL=osmesa python3 scripts/run_benchmark_3d.py --rooms 10 \
        --out-json docs/research/data/benchmark_3d.json

For each held-out room, the SAME scene runs as:
  * geometry-FBE — symbolic ray sensing (perception upper bound),
  * vision-FBE  — RGB ball detection + height-aware depth mapping (pixels only),
  * random      — floor.
The room's 3D geometry matches the 2D collision substrate exactly
(`scene_mjcf`), so the geometry-vs-vision gap measures the PERCEPTION pipeline
and nothing else. Also reports depth-strip fidelity (MAE vs ray-cast ground
truth at the start pose) and, if available, planar-surface extraction metrics.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from wanderai.benchmark import aggregate, save_report, to_markdown
from wanderai.environment import EnvConfig, SceneSearchEnv
from wanderai.frontier_policy import FrontierPolicy, FrontierConfig
from wanderai.mapping import depth_strip, obstacle_strip_from_depth
from wanderai.metrics import EpisodeResult
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode
from wanderai.scene_gen import make_split
from wanderai.scene_mjcf import scene_renderer_3d


def strip_mae(scene, renderer) -> float:
    """Sensor fidelity: vision strip vs ray-cast ground truth measured at the
    SAME bearings against RAW (un-inflated) surfaces. Comparing against the
    default depth_strip would charge two harness artifacts to perception:
    its rays sit on a different bearing grid than the vision bins, and
    cast_ray measures configuration-space ranges (obstacles inflated by
    agent_radius) while the camera sees the raw surface."""
    from dataclasses import replace
    from wanderai.observation import cast_ray
    rgb, depth = renderer.render_rgb_depth(scene, scene.agent_start)
    vis = obstacle_strip_from_depth(
        depth, fov_x=renderer.fov_x, fov_y=renderer.fov_y,
        eye_height=renderer.eye_height,
        pitch_rad=math.radians(renderer.pitch_deg), max_range=6.0)
    raw = replace(scene, agent_radius=0.0)
    pose = scene.agent_start
    errs = [abs(cast_ray(raw, pose.x, pose.y, pose.heading + rel, 6.0) - rng)
            for rel, rng in vis]
    return float(np.mean(errs))


def plane_metrics(scene, renderer):
    """Floor-height error, wall-verticality error, navigable-point precision —
    None if the planes module is unavailable."""
    try:
        from wanderai.planes import (world_point_cloud, estimate_normals,
                                     extract_planes, classify_plane,
                                     navigable_points)
    except ImportError:
        return None
    rgb, depth = renderer.render_rgb_depth(scene, scene.agent_start)
    # Depth-cap pixels are sky (no ceiling), not surfaces — mask before fitting.
    depth = np.where(depth >= 0.99 * renderer.max_depth, np.nan, depth)
    pts = world_point_cloud(depth, scene.agent_start, fov_x=renderer.fov_x,
                            fov_y=renderer.fov_y, eye_height=renderer.eye_height,
                            pitch_rad=math.radians(renderer.pitch_deg))
    normals = estimate_normals(pts)
    planes = extract_planes(pts, normals)
    floor_err = wall_err = None
    kinds = []
    for p in planes:
        kind = classify_plane(p, agent_height=renderer.eye_height)
        kinds.append(kind)
        if kind == "floor" and floor_err is None:
            floor_err = abs(float(p.centroid[2]))
        if kind == "wall" and wall_err is None:
            wall_err = math.degrees(math.asin(min(1.0, abs(float(p.normal[2])))))
    nav = navigable_points(planes, pts)
    if len(nav):
        # Validate against RAW footprints: scene.is_free is a configuration-
        # space test (inflated by agent_radius), which would score genuine
        # floor within 0.2 m of furniture as a false positive.
        from dataclasses import replace
        raw = replace(scene, agent_radius=0.0)
        free = np.array([raw.is_free(float(x), float(y)) for x, y in nav])
        precision = float(free.mean())
    else:
        precision = None
    return {"n_planes": len(planes), "kinds": kinds, "floor_height_err_m": floor_err,
            "wall_vertical_err_deg": wall_err, "n_navigable": int(len(nav)),
            "navigable_precision": precision}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="scene-split seed")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--skip-planes", action="store_true")
    ap.add_argument("--out-json", default="docs/research/data/benchmark_3d.json")
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()

    _, test = make_split(0, args.rooms, seed=args.seed)
    by_policy: dict[str, list[EpisodeResult]] = {
        "fbe-geometry": [], "fbe-vision (pixels only)": [], "random": [],
        "oracle (privileged)": []}
    maes, planes_rows = [], []

    for i, s in enumerate(test):
        scene, renderer = scene_renderer_3d(s)
        with renderer:
            maes.append(strip_mae(scene, renderer))
            if not args.skip_planes:
                pm = plane_metrics(scene, renderer)
                if pm:
                    planes_rows.append(pm)
            cfg_v = EnvConfig(max_steps=args.max_steps, perception="vision")
            by_policy["fbe-vision (pixels only)"].append(run_episode(
                SceneSearchEnv(scene, renderer=renderer, config=cfg_v),
                FrontierPolicy(FrontierConfig(seed=i))))
        cfg_g = EnvConfig(max_steps=args.max_steps)
        by_policy["fbe-geometry"].append(
            run_episode(SceneSearchEnv(scene, config=cfg_g),
                        FrontierPolicy(FrontierConfig(seed=i))))
        by_policy["random"].append(
            run_episode(SceneSearchEnv(scene, config=cfg_g), RandomPolicy(seed=i)))
        by_policy["oracle (privileged)"].append(
            run_episode(SceneSearchEnv(scene, config=cfg_g), OraclePolicy()))
        print(f"room {i}: vision success="
              f"{by_policy['fbe-vision (pixels only)'][-1].success} "
              f"strip_mae={maes[-1]:.2f}m")

    rows = [aggregate(name, results) for name, results in by_policy.items()]
    meta = {"benchmark": "3d-matched-procedural", "rooms": args.rooms,
            "split_seed": args.seed, "max_steps": args.max_steps,
            "strip_mae_mean_m": float(np.mean(maes)),
            "strip_mae_per_room": [round(m, 3) for m in maes]}
    if planes_rows:
        fe = [p["floor_height_err_m"] for p in planes_rows
              if p["floor_height_err_m"] is not None]
        we = [p["wall_vertical_err_deg"] for p in planes_rows
              if p["wall_vertical_err_deg"] is not None]
        pr = [p["navigable_precision"] for p in planes_rows
              if p["navigable_precision"] is not None]
        meta["planes"] = {
            "rooms_evaluated": len(planes_rows),
            "floor_detected_in": len(fe),
            "floor_height_mae_m": float(np.mean(fe)) if fe else None,
            "wall_vertical_err_deg_mean": float(np.mean(we)) if we else None,
            "navigable_precision_mean": float(np.mean(pr)) if pr else None,
        }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    save_report(args.out_json, rows, meta)
    caption = (f"3D matched procedural rooms ({args.rooms} held-out, split seed "
               f"{args.seed}): geometry vs pixels-only sensing. "
               f"Depth-strip MAE {np.mean(maes):.2f} m")
    md = to_markdown(rows, caption)
    if args.out_md:
        with open(args.out_md, "w") as fh:
            fh.write(md)
    print("\n" + md)
    if "planes" in meta:
        print("planes:", meta["planes"])


if __name__ == "__main__":
    main()
