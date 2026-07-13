"""Planar-surface extraction tests.

The synthetic cases build analytic depth images (exact ray–plane intersection,
no renderer needed) so the geometry — perpendicular-z back-projection, the
camera extrinsics, RANSAC and classification — is verified without MuJoCo.
The integration test renders a real MJCF scene and is skipped where MuJoCo
is unavailable."""
import math
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")   # before any mujoco import

import numpy as np
import pytest

from wanderai.geometry import Pose
from wanderai.planes import (camera_point_cloud, world_point_cloud,
                             estimate_normals, extract_planes, classify_plane,
                             navigable_points, plane_report)

W, H = 160, 120
FOV_X = math.pi / 2
FOV_Y = 2.0 * math.atan(math.tan(FOV_X / 2.0) * H / W)
EYE = 1.4
PITCH = math.radians(8.0)


def floor_depth(eye_z: float = EYE, pitch: float = PITCH,
                max_range: float = 8.0) -> np.ndarray:
    """Analytic perpendicular-z depth of an infinite floor at z=0.

    A pixel's ray direction is u_tan*right + v_tan*down + forward; its world-z
    slope is -(v_tan*cos(pitch) + sin(pitch)), so the ray meets the floor at
    forward-distance z = eye_z / (v_tan*cos p + sin p). Pixels whose ray never
    hits the floor (or hits it beyond max_range) are NaN."""
    ndc_x = 2.0 * (np.arange(W) + 0.5) / W - 1.0
    ndc_y = 2.0 * (np.arange(H) + 0.5) / H - 1.0
    v_tan = (ndc_y * math.tan(FOV_Y / 2.0))[:, None] * np.ones((1, W))
    denom = v_tan * math.cos(pitch) + math.sin(pitch)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = eye_z / denom
    z[(denom <= 0) | (z > max_range)] = np.nan
    return z


def test_camera_point_cloud_center_and_edges():
    depth = np.full((H, W), 2.0)
    pts = camera_point_cloud(depth, FOV_X, FOV_Y)
    assert pts.shape == (H, W, 3)
    assert np.allclose(pts[..., 2], 2.0)
    # centre pixel looks (almost) straight ahead
    c = pts[H // 2, W // 2]
    assert abs(c[0]) < 2.0 * math.tan(FOV_X / 2) / W * 2.01
    # rightmost column sits at nearly x = z*tan(fov_x/2); bottom rows have y > 0
    assert pts[H // 2, -1, 0] == pytest.approx(
        2.0 * math.tan(FOV_X / 2) * (1 - 1.0 / W), abs=1e-9)
    assert pts[-1, W // 2, 1] > 0


def test_world_point_cloud_recovers_flat_floor():
    pose = Pose(1.0, 2.0, 0.5)
    depth = floor_depth()
    pts = world_point_cloud(depth, pose, fov_x=FOV_X, fov_y=FOV_Y,
                            eye_height=EYE, pitch_rad=PITCH, floor_z=0.0)
    zs = pts[..., 2][np.isfinite(depth)]
    assert zs.size > 2000
    assert np.max(np.abs(zs)) < 0.02


def test_extract_and_classify_floor():
    pose = Pose(0.0, 0.0, -1.2)
    depth = floor_depth()
    pts = world_point_cloud(depth, pose, fov_x=FOV_X, fov_y=FOV_Y,
                            eye_height=EYE, pitch_rad=PITCH, floor_z=0.0)
    nrm = estimate_normals(pts, view_point=np.array([pose.x, pose.y, EYE]))
    planes = extract_planes(pts, nrm)
    assert planes, "no plane found on an analytic floor"
    floors = [p for p in planes
              if classify_plane(p, floor_z=0.0, agent_height=EYE) == "floor"]
    assert floors, "floor plane not classified as floor"
    best = max(floors, key=lambda p: p.inliers)
    assert abs(best.normal[2]) > 0.98
    assert abs(best.centroid[2]) < 0.05
    # navigable evidence exists and lies at floor height
    nav = navigable_points(planes, pts, nrm)
    assert nav.shape[0] > 500 and nav.shape[1] == 2


def test_extract_and_classify_wall():
    # Vertical wall 3 m ahead, camera unpitched: perpendicular depth is a
    # constant 3.0 for every pixel (the wall fills the whole view).
    pose = Pose(0.0, 0.0, 0.3)
    depth = np.full((H, W), 3.0)
    pts = world_point_cloud(depth, pose, fov_x=FOV_X, fov_y=FOV_Y,
                            eye_height=EYE, pitch_rad=0.0, floor_z=0.0)
    nrm = estimate_normals(pts, view_point=np.array([pose.x, pose.y, EYE]))
    planes = extract_planes(pts, nrm)
    assert planes
    kinds = [classify_plane(p, floor_z=0.0, agent_height=EYE) for p in planes]
    assert "wall" in kinds
    wall = planes[kinds.index("wall")]
    assert abs(wall.normal[2]) < 0.1
    # the wall plane sits 3 m ahead of the camera along its heading
    ahead = wall.normal @ np.array([math.cos(pose.heading),
                                    math.sin(pose.heading), 0.0])
    assert abs(abs(wall.offset - wall.normal[:2] @ [pose.x, pose.y]) / abs(ahead)
               - 3.0) < 0.05
    # no navigable ground in a wall-only view
    assert navigable_points(planes, pts, nrm).shape[0] == 0


BOX = (2.5, 3.5, -0.5, 0.5)     # world footprint of the MJCF box below

MJCF = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.5 0.55 0.5 1"/>
    <geom name="box" type="box" size="0.5 0.5 0.4" pos="3 0 0.4"
          rgba="0.6 0.4 0.3 1"/>
    <geom name="ball" type="sphere" size="0.15" pos="2 1.5 0.15"
          rgba="0.9 0.05 0.05 1"/>
  </worldbody>
</mujoco>
"""


def _mujoco_report():
    mujoco = pytest.importorskip("mujoco")
    from wanderai.mujoco_renderer import MuJoCoRenderer
    model = mujoco.MjModel.from_xml_string(MJCF)
    # Close the GL context when done — a leaked OSMesa context corrupts every
    # renderer created later in the same pytest process.
    with MuJoCoRenderer(model, world_offset=(0.0, 0.0), floor_z=0.0) as renderer:
        pose = Pose(0.0, 0.0, 0.0)          # facing the box 3 m ahead
        _, depth = renderer.render_rgb_depth(None, pose)
        meta = {"fov_x": renderer.fov_x, "fov_y": renderer.fov_y,
                "eye_height": renderer.eye_height,
                "pitch_rad": math.radians(renderer.pitch_deg),
                "floor_z": renderer.floor_z, "max_depth": renderer.max_depth}
    return depth, pose, meta


def test_mujoco_pipeline_floor_wall_and_navigable_precision():
    depth, pose, meta = _mujoco_report()
    report = plane_report(depth, pose, meta)
    kinds = [p["type"] for p in report["planes"]]
    floors = [p for p in report["planes"] if p["type"] == "floor"]
    assert floors, f"no floor plane in {report['planes']}"
    assert abs(max(floors, key=lambda p: p["inliers"])["height"]) < 0.1
    assert "wall" in kinds, f"no vertical plane in {report['planes']}"

    # Recompute navigable coordinates and check precision against the box
    # footprint (privileged geometry used ONLY as test ground truth).
    z = np.where(depth >= 0.999 * meta["max_depth"], np.nan, depth)
    pts = world_point_cloud(z, pose, fov_x=meta["fov_x"], fov_y=meta["fov_y"],
                            eye_height=meta["eye_height"],
                            pitch_rad=meta["pitch_rad"], floor_z=0.0)
    nrm = estimate_normals(pts, view_point=np.array([pose.x, pose.y,
                                                     meta["eye_height"]]))
    nav = navigable_points(extract_planes(pts, nrm), pts, nrm)
    assert nav.shape[0] == report["n_navigable"] > 1000
    x0, x1, y0, y1 = BOX
    inside = ((nav[:, 0] > x0) & (nav[:, 0] < x1) &
              (nav[:, 1] > y0) & (nav[:, 1] < y1))
    assert inside.mean() < 0.05, f"{inside.mean():.1%} of navigable pts in box"
