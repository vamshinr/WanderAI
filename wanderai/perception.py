"""Vision perception — turn a rendered RGB+depth frame into the symbolic
`Observation` the text policy already consumes.

This is the bridge between Phase B (pixels) and the existing text-RFT pipeline:
instead of computing the egocentric view from privileged geometry, we *see* it.
  * the red ball is found by segmenting red pixels in the RGB image and keeping the
    largest compact, roughly-circular blob (so red furniture or distant specks are
    rejected); its bearing comes from the blob centroid, its distance from the
    depth buffer;
  * left / centre / right wall clearance is read directly from the depth buffer
    (this is the "depth perception for distance calculations").
The episodic memory (visited cells / recent moves) is carried by the env, exactly
as in the geometric path, so `observation_text(obs)` renders an identical format —
the policy can't tell which perception produced it.
"""
from __future__ import annotations
import math

import numpy as np

from .observation import Observation, visit_key, VISIT_LOOKAHEAD, DEFAULT_HISTORY

MIN_BLOB_PIXELS = 8        # smaller red blobs are noise/distant specks, not the ball
MIN_FILL = 0.40           # blob area / bounding-box area (a disk is ~0.79; a couch isn't)
ASPECT_RANGE = (0.4, 2.6)  # bounding-box w/h — a ball is roughly square, a sofa isn't
HORIZON_LO, HORIZON_HI = 0.28, 0.60   # rows (fraction of H) used for wall clearance


def red_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of red pixels: strongly red, clearly above green/blue.
    Tolerant to MuJoCo shading (the lit ball reads roughly (200, 40, 40))."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 110) & (r - g > 55) & (r - b > 55)


def largest_blob(mask: np.ndarray):
    """Largest 4-connected component of `mask` → (ys, xs) pixel coords, or None.
    Plain BFS — red pixels are few, so this is cheap and needs no scipy."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    best: list | None = None
    rys, rxs = np.where(mask)
    for y0, x0 in zip(rys.tolist(), rxs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if best is None or len(comp) > len(best):
            best = comp
    if best is None:
        return None
    arr = np.array(best)
    return arr[:, 0], arr[:, 1]


def detect_ball(rgb: np.ndarray):
    """Find the ball blob → (col_centroid, ys, xs) or None. Rejects small specks,
    elongated shapes, and sparse (non-disk) blobs so red furniture isn't the ball."""
    blob = largest_blob(red_mask(rgb))
    if blob is None:
        return None
    ys, xs = blob
    area = len(xs)
    if area < MIN_BLOB_PIXELS:
        return None
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    fill = area / float(bw * bh)
    aspect = bw / float(bh)
    if fill < MIN_FILL or not (ASPECT_RANGE[0] <= aspect <= ASPECT_RANGE[1]):
        return None
    return float(xs.mean()), ys, xs


def _clearance_from_depth(depth: np.ndarray, lo_col: int, hi_col: int,
                          max_depth: float) -> float:
    """Nearest surface in a vertical column band, sampled across the horizon rows
    (so the floor below and ceiling above don't masquerade as obstacles)."""
    h = depth.shape[0]
    r0, r1 = int(h * HORIZON_LO), int(h * HORIZON_HI)
    band = depth[r0:r1, lo_col:hi_col]
    band = band[np.isfinite(band)]
    if band.size == 0:
        return max_depth
    # 5th percentile ≈ nearest robust surface (ignores a few stray near pixels)
    return float(min(max_depth, np.percentile(band, 5)))


def perceive(renderer, scene, pose, history=None, visited=None) -> Observation:
    """Render the agent's view and decode it into an `Observation`."""
    rgb, depth = renderer.render_rgb_depth(scene, pose)
    h, w = depth.shape
    max_depth = float(getattr(renderer, "max_depth", 10.0))
    fov_x = float(getattr(renderer, "fov_x", math.pi / 2))
    half_tan = math.tan(fov_x / 2.0)

    # --- red ball ---
    hit = detect_ball(rgb)
    ball_visible = hit is not None
    bearing = distance = None
    if ball_visible:
        col, ys, xs = hit
        ndc = 2.0 * col / w - 1.0                 # -1 (left) .. +1 (right)
        bearing = -math.atan(ndc * half_tan)      # + = left, matches geometry
        distance = float(np.median(depth[ys, xs]))

    # --- wall clearance from depth: left / centre / right thirds ---
    t = w // 3
    clearance = {
        "left": _clearance_from_depth(depth, 0, t, max_depth),
        "center": _clearance_from_depth(depth, t, 2 * t, max_depth),
        "right": _clearance_from_depth(depth, 2 * t, w, max_depth),
    }

    # --- episodic memory (carried by the env, same as the geometric path) ---
    recent = [getattr(a, "name", str(a)) for a in (history or [])][-DEFAULT_HISTORY:]
    visited = visited or set()

    def seen(angle):
        px = pose.x + VISIT_LOOKAHEAD * math.cos(angle)
        py = pose.y + VISIT_LOOKAHEAD * math.sin(angle)
        return visit_key(px, py) in visited

    explored = {"left": seen(pose.heading + fov_x / 2),
                "center": seen(pose.heading),
                "right": seen(pose.heading - fov_x / 2)}
    return Observation(ball_visible, bearing, distance, clearance, recent,
                       explored, len(visited))
