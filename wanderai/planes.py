"""Planar-surface extraction from egocentric depth — navigable-space estimation.

Indoor scenes are dominated by planes: the floor the agent can drive on, walls
that bound it, and elevated horizontal surfaces (tabletops, shelves) that are
*actionable* but not traversable. This module lifts a rendered depth image into
a world-frame point cloud, segments it into planes (sequential RANSAC with
normal agreement), and classifies each plane so a pixels-only pipeline can say
"this ground is navigable" without ever touching privileged geometry — the
`navigable_points` output is exactly the free-space evidence a vision-driven
`EgoMap` would consume instead of ray-cast strips.

Honesty contract — everything here is computed from:
  * the rendered depth buffer (a sensor the agent legitimately owns);
  * the agent's own pose (GPS+Compass, standard in ObjectNav) plus the fixed,
    known camera intrinsics/extrinsics (FOV, eye height, pitch).
It never reads `scene.obstacles`, `scene.ball`, or the env's occupancy grid.

Depth convention: MuJoCo depth is PERPENDICULAR z-distance — the component of
the point along the camera's forward axis, not the length of the ray (the same
convention `mapping.depth_strip_from_image` corrects for with a cos division).
Camera frame follows the image convention: +z forward, +x right, +y down.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Pose

# Matches MuJoCoRenderer defaults: horizontal planes above this height are
# overhead structure (ceiling, soffit), not surfaces the agent can act on.
DEFAULT_AGENT_HEIGHT = 1.4
FLOOR_TOLERANCE = 0.15      # |plane height - floor_z| below this counts as floor
RANSAC_HYPOTHESES = 40      # candidate (point, normal) seeds tried per plane


def camera_point_cloud(depth: np.ndarray, fov_x: float, fov_y: float) -> np.ndarray:
    """Back-project a depth image to camera-frame points, shape (H, W, 3).

    Depth is perpendicular z, so each pixel's point is p = (u_tan*z, v_tan*z, z)
    where u_tan = ndc_x * tan(fov_x/2) with ndc_x = 2*(col+0.5)/W - 1 (pixel
    centers), and v_tan likewise from the row and fov_y. Non-finite depth
    yields non-finite points (callers mask them)."""
    z = np.asarray(depth, dtype=np.float64)
    h, w = z.shape
    ndc_x = 2.0 * (np.arange(w) + 0.5) / w - 1.0
    ndc_y = 2.0 * (np.arange(h) + 0.5) / h - 1.0
    u_tan = ndc_x * math.tan(fov_x / 2.0)          # (W,) — + is image right
    v_tan = ndc_y * math.tan(fov_y / 2.0)          # (H,) — + is image down
    return np.stack([u_tan[None, :] * z, v_tan[:, None] * z, z], axis=-1)


def world_point_cloud(depth: np.ndarray, pose: Pose, *, fov_x: float, fov_y: float,
                      eye_height: float, pitch_rad: float,
                      floor_z: float = 0.0) -> np.ndarray:
    """Back-project depth into the scene frame, shape (H, W, 3).

    The camera sits at (pose.x, pose.y, floor_z + eye_height), yawed to
    pose.heading and pitched DOWN by `pitch_rad` (matching `MuJoCoRenderer`'s
    free camera with elevation = -pitch). Camera axes in world coordinates:
      forward = (cos p * cos yaw, cos p * sin yaw, -sin p)
      right   = (sin yaw, -cos yaw, 0)          # +x world, yaw=0 -> right is -y
      down    = forward x right = (-sin p * cos yaw, -sin p * sin yaw, -cos p)
    so world = eye + x_cam*right + y_cam*down + z_cam*forward."""
    cam = camera_point_cloud(depth, fov_x, fov_y)
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cy, sy = math.cos(pose.heading), math.sin(pose.heading)
    right = np.array([sy, -cy, 0.0])
    down = np.array([-sp * cy, -sp * sy, -cp])
    forward = np.array([cp * cy, cp * sy, -sp])
    eye = np.array([pose.x, pose.y, floor_z + eye_height])
    # rows of R are the camera basis vectors: cam @ R = x*right + y*down + z*fwd
    R = np.stack([right, down, forward], axis=0)
    return cam @ R + eye


def estimate_normals(points: np.ndarray,
                     view_point: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel unit normals of a point image, shape (H, W, 3).

    Cross product of the horizontal and vertical finite differences of the
    point image (np.gradient), normalized, then flipped so every normal faces
    the camera at `view_point` (origin by default — the camera-frame case).
    Consistent orientation is what lets RANSAC use a signed normal-agreement
    test, so the two faces of a thin wall can never pool into one plane.
    Pixels with non-finite neighbours get non-finite normals."""
    pts = np.asarray(points, dtype=np.float64)
    d_row = np.gradient(pts, axis=0)               # down the image
    d_col = np.gradient(pts, axis=1)               # across the image
    n = np.cross(d_col, d_row)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    vp = np.zeros(3) if view_point is None else np.asarray(view_point, dtype=np.float64)
    flip = np.sum(n * (vp - pts), axis=-1, keepdims=True) < 0
    return np.where(flip, -n, n)


@dataclass
class Plane:
    normal: np.ndarray      # unit (3,), oriented toward the camera
    offset: float           # plane equation: normal . p = offset
    inliers: int
    centroid: np.ndarray    # (3,) mean of the inlier points


def extract_planes(points: np.ndarray, normals: np.ndarray, *,
                   dist_thresh: float = 0.04, normal_thresh_deg: float = 15.0,
                   min_inliers: int = 150, max_planes: int = 6, seed: int = 0,
                   sample_stride: int = 2) -> list[Plane]:
    """Sequential RANSAC plane segmentation of a (H, W, 3) point image.

    Each round: seed candidate planes from randomly sampled unassigned pixels
    (a point plus its estimated normal fully determines a plane hypothesis),
    count unassigned inliers — within `dist_thresh` of the plane AND normal
    within `normal_thresh_deg` of the plane normal — keep the best of
    `RANSAC_HYPOTHESES` seeds, refine it by total least squares (SVD) over its
    inliers, re-count, and remove the inliers from the pool. Stops when the
    best plane falls below `min_inliers` or `max_planes` is reached. The pixel
    grid is subsampled by `sample_stride` (inlier counts are in subsampled
    pixels). All inlier tests are vectorized."""
    P = np.asarray(points, dtype=np.float64)[::sample_stride, ::sample_stride]
    N = np.asarray(normals, dtype=np.float64)[::sample_stride, ::sample_stride]
    P, N = P.reshape(-1, 3), N.reshape(-1, 3)
    valid = np.isfinite(P).all(axis=1) & np.isfinite(N).all(axis=1)
    cos_thr = math.cos(math.radians(normal_thresh_deg))
    rng = np.random.default_rng(seed)

    unassigned = valid.copy()
    planes: list[Plane] = []
    while len(planes) < max_planes:
        idx = np.flatnonzero(unassigned)
        if idx.size < min_inliers:
            break
        Pu, Nu = P[idx], N[idx]
        seeds = rng.choice(idx, size=min(RANSAC_HYPOTHESES, idx.size), replace=False)
        best_count, best_mask, best_n = 0, None, None
        for i in seeds:
            n0, d0 = N[i], float(N[i] @ P[i])
            mask = (np.abs(Pu @ n0 - d0) < dist_thresh) & (Nu @ n0 > cos_thr)
            count = int(mask.sum())
            if count > best_count:
                best_count, best_mask, best_n = count, mask, n0
        if best_count < min_inliers:
            break
        # Refine: total-least-squares plane through the inliers (smallest
        # singular vector of the centred cloud), oriented like the hypothesis.
        pin = Pu[best_mask]
        c = pin.mean(axis=0)
        _, _, vt = np.linalg.svd(pin - c, full_matrices=False)
        n = vt[2] if vt[2] @ best_n >= 0 else -vt[2]
        d = float(n @ c)
        mask = (np.abs(Pu @ n - d) < dist_thresh) & (Nu @ n > cos_thr)
        if int(mask.sum()) < min_inliers:
            break
        pin = Pu[mask]
        planes.append(Plane(normal=n, offset=d, inliers=int(mask.sum()),
                            centroid=pin.mean(axis=0)))
        unassigned[idx[mask]] = False
    planes.sort(key=lambda p: -p.inliers)
    return planes


def classify_plane(plane: Plane, *, floor_z: float = 0.0, agent_height: float,
                   horiz_cos: float = math.cos(math.radians(20.0))) -> str:
    """'floor' | 'wall' | 'surface' | 'other'.

    floor    — horizontal normal (|nz| > horiz_cos) at floor height;
    surface  — horizontal, elevated above the floor but within the agent's
               reach (an actionable surface: tabletop, shelf, seat);
    wall     — vertical normal (|nz| < sin of the same tolerance angle);
    other    — slanted planes, ceilings/overhead structure."""
    nz = abs(float(plane.normal[2]))
    height = float(plane.centroid[2]) - floor_z
    vert_sin = math.sqrt(max(0.0, 1.0 - horiz_cos * horiz_cos))
    if nz > horiz_cos:
        if abs(height) < FLOOR_TOLERANCE:
            return "floor"
        if FLOOR_TOLERANCE < height <= agent_height:
            return "surface"
        return "other"
    if nz < vert_sin:
        return "wall"
    return "other"


def navigable_points(planes: list[Plane], points: np.ndarray,
                     normals: np.ndarray | None = None, *,
                     floor_z: float = 0.0,
                     agent_height: float = DEFAULT_AGENT_HEIGHT,
                     dist_thresh: float = 0.04,
                     normal_thresh_deg: float = 15.0) -> np.ndarray:
    """World (x, y) of pixels lying on floor-classified planes, shape (N, 2).

    This is vision-only free-space evidence: every returned coordinate is a
    piece of ground the agent has *seen* to be flat and at floor height — the
    natural free-cell input for `EgoMap` in a pixels-only pipeline. Pass the
    normal image to also require per-pixel normal agreement, which trims the
    thin ring of near-floor pixels at the base of obstacles."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    member = np.zeros(P.shape[0], dtype=bool)
    valid = np.isfinite(P).all(axis=1)
    Nn = None
    if normals is not None:
        Nn = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        valid &= np.isfinite(Nn).all(axis=1)
    cos_thr = math.cos(math.radians(normal_thresh_deg))
    for plane in planes:
        if classify_plane(plane, floor_z=floor_z, agent_height=agent_height) != "floor":
            continue
        with np.errstate(invalid="ignore"):
            m = np.abs(P @ plane.normal - plane.offset) < dist_thresh
            if Nn is not None:
                m &= Nn @ plane.normal > cos_thr
        member |= m & valid
    return P[member][:, :2]


def plane_report(depth: np.ndarray, pose: Pose, renderer_meta: dict) -> dict:
    """Run the full pipeline on one depth frame and summarize it.

    `renderer_meta` carries the camera model: fov_x, fov_y, eye_height,
    pitch_rad, floor_z (all straight off a `MuJoCoRenderer`), and optionally
    max_depth — pixels at the depth cap are sky/out-of-range, not surfaces,
    so they are masked out before back-projection. Returns
    {"planes": [{"type", "normal", "height", "inliers"}], "n_navigable"}."""
    floor_z = float(renderer_meta.get("floor_z", 0.0))
    eye_height = float(renderer_meta["eye_height"])
    z = np.asarray(depth, dtype=np.float64)
    max_depth = renderer_meta.get("max_depth")
    if max_depth is not None:
        z = np.where(z >= 0.999 * float(max_depth), np.nan, z)
    pts = world_point_cloud(z, pose, fov_x=float(renderer_meta["fov_x"]),
                            fov_y=float(renderer_meta["fov_y"]),
                            eye_height=eye_height,
                            pitch_rad=float(renderer_meta["pitch_rad"]),
                            floor_z=floor_z)
    eye = np.array([pose.x, pose.y, floor_z + eye_height])
    nrm = estimate_normals(pts, view_point=eye)
    planes = extract_planes(pts, nrm)
    agent_height = float(renderer_meta.get("agent_height", eye_height))
    nav = navigable_points(planes, pts, nrm, floor_z=floor_z,
                           agent_height=agent_height)
    return {
        "planes": [{
            "type": classify_plane(p, floor_z=floor_z, agent_height=agent_height),
            "normal": [float(v) for v in p.normal],
            "height": float(p.centroid[2]) - floor_z,
            "inliers": p.inliers,
        } for p in planes],
        "n_navigable": int(nav.shape[0]),
    }
