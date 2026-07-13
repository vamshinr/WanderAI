"""Transfer experiment: the real Gizmo-exported room (20x15 m).

    MUJOCO_GL=osmesa python3 scripts/run_gizmo_transfer.py

Gizmo exports carry furniture as position-only bodies (meshes external), so
the VISUAL world contains real clutter the imported 2D collision world lacks.
Pixels-only mapping therefore perceives obstacles that do not exist in the
collision substrate — a known, documented sim artifact of this export format.
The matched procedural rooms (scripts/run_benchmark_3d.py) are the controlled
vision benchmark; this script measures transfer of the geometry-mode policy to
a real export and records the visual/collision mismatch quantitatively.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np

from wanderai.environment import EnvConfig, SceneSearchEnv
from wanderai.frontier_policy import FrontierPolicy, FrontierConfig
from wanderai.mapping import depth_strip, obstacle_strip_from_depth
from wanderai.mujoco_renderer import load_mjcf_3d
from wanderai.policies import OraclePolicy, RandomPolicy, run_episode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", default="wander_lake/scene_3d_train.xml")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-json", default="docs/research/data/gizmo_transfer.json")
    args = ap.parse_args()

    scene, renderer, meta = load_mjcf_3d(args.scene, return_meta=True)
    out = {
        "scene": f"{args.scene} (Gizmo export)",
        "note": ("Gizmo furniture is position-only (meshes external): the "
                 "visual world contains real clutter the 2D collision world "
                 "lacks, so pixels-only mapping perceives obstacles that do "
                 "not exist in the collision substrate. The strip disagreement "
                 "below quantifies that mismatch — it is a property of the "
                 "export format, not of the perception pipeline (see the "
                 "matched-room benchmark for the controlled comparison)."),
        "bounds": [scene.bounds.max_x, scene.bounds.max_y],
        "obstacles_2d": len(scene.obstacles),
    }
    rgb, depth = renderer.render_rgb_depth(scene, scene.agent_start)
    geo = depth_strip(scene, scene.agent_start, max_range=6.0)
    vis = obstacle_strip_from_depth(
        depth, fov_x=renderer.fov_x, fov_y=renderer.fov_y,
        eye_height=renderer.eye_height,
        pitch_rad=math.radians(renderer.pitch_deg), max_range=6.0)
    out["visual_vs_collision_strip_mad_m"] = float(
        np.mean([abs(a - b) for (_, a), (_, b) in zip(geo, vis)]))

    cfg_g = EnvConfig(max_steps=args.max_steps)
    cfg_v = EnvConfig(max_steps=args.max_steps, perception="vision")

    def ep(policy, cfg):
        r = run_episode(SceneSearchEnv(scene, renderer=renderer, config=cfg),
                        policy)
        return {"success": bool(r.success), "steps": r.steps,
                "spl_term": (float(r.optimal / max(r.path_length, r.optimal))
                             if r.success else 0.0),
                "collisions": r.collisions, "coverage": r.visited_cells}

    out["episodes"] = {
        "oracle": ep(OraclePolicy(), cfg_g),
        "random": ep(RandomPolicy(seed=0), cfg_g),
        "fbe_geometry": [ep(FrontierPolicy(FrontierConfig(seed=s)), cfg_g)
                         for s in range(args.seeds)],
        "fbe_vision": [ep(FrontierPolicy(FrontierConfig(seed=s)), cfg_v)
                       for s in range(args.seeds)],
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
