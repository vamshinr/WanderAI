"""Import a MuJoCo MJCF scene (e.g. exported from Antim/Gizmo) into our `Scene`.

Gizmo exports rich scenes; we extract what our 2D search env needs: the floor
extent (bounds), box obstacle footprints, and the ball position. Geometry is
projected to the floor plane (rotations ignored — an axis-aligned approximation).
A free, reachable agent start is chosen so the imported scene is always solvable.

This is format-driven (standard MJCF), not Gizmo-specific, so it also imports MJCF
from any other source — including the Phase-B MuJoCo renderer's own scenes."""

from __future__ import annotations
import math
import os
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from .geometry import AABB, Pose
from .scene import Scene
from .occupancy import OccupancyGrid
from .distance_field import DistanceField


def _floats(s, n=3, default=0.0):
    parts = [float(v) for v in (s or "").split()]
    parts += [default] * (n - len(parts))
    return parts[:n]


def _collect_geoms(elem, ox=0.0, oy=0.0, out=None):
    """Recursively gather geoms with world (x,y) accumulated from body offsets."""
    if out is None:
        out = []
    for child in elem:
        tag = child.tag.split("}")[-1]
        if tag == "body":
            bx, by, _ = _floats(child.get("pos"), 3)
            _collect_geoms(child, ox + bx, oy + by, out)
        elif tag == "geom":
            gx, gy, _ = _floats(child.get("pos"), 3)
            out.append({
                "type": child.get("type", "sphere"),
                "x": ox + gx, "y": oy + gy,
                "size": _floats(child.get("size"), 3),
                "rgba": _floats(child.get("rgba"), 4, default=1.0) if child.get("rgba") else None,
                "name": (child.get("name") or "").lower(),
            })
        else:
            _collect_geoms(child, ox, oy, out)
    return out


def _is_redish(rgba):
    return rgba is not None and rgba[0] > 0.5 and rgba[1] < 0.4 and rgba[2] < 0.4


def _place_agent(bounds, obstacles, agent_radius, ball, step=0.4):
    """Pick a free, reachable point far from the ball, facing roughly toward it."""
    probe = Scene(bounds, obstacles, ball, Pose(0, 0, 0), agent_radius)
    grid = OccupancyGrid.from_scene(probe, 0.1)
    field = DistanceField.from_grid(grid, ball)
    best, best_d = None, -1.0
    y = bounds.min_y + agent_radius
    while y < bounds.max_y:
        x = bounds.min_x + agent_radius
        while x < bounds.max_x:
            if probe.is_free(x, y) and math.isfinite(field.query(x, y)):
                d = math.hypot(x - ball[0], y - ball[1])
                if d > best_d:
                    best_d, best = d, (x, y)
            x += step
        y += step
    if best is None:                       # degenerate: fall back to a corner
        best = (bounds.min_x + agent_radius * 2, bounds.min_y + agent_radius * 2)
    heading = math.atan2(ball[1] - best[1], ball[0] - best[0])
    return Pose(best[0], best[1], heading)


def mjcf_to_scene(source: str, agent_radius: float = 0.2, margin: float = 0.2) -> Scene:
    """Parse MJCF (an XML string or a path to a .xml) into a Scene."""
    xml = source
    if os.path.exists(source) and source.endswith(".xml"):
        with open(source) as fh:
            xml = fh.read()
    root = ET.fromstring(xml)
    world = root.find(".//worldbody")
    if world is None:
        raise ValueError("MJCF has no <worldbody>")
    geoms = _collect_geoms(world)

    boxes = [g for g in geoms if g["type"] == "box"]
    spheres = [g for g in geoms if g["type"] == "sphere"]
    planes = [g for g in geoms if g["type"] == "plane"]
    if not spheres:
        raise ValueError("no sphere found to use as the ball")

    # Bounds: a finite floor plane if present, else the bbox of all geoms + margin.
    if planes and planes[0]["size"][0] > 0 and planes[0]["size"][1] > 0:
        p = planes[0]
        hx, hy = p["size"][0], p["size"][1]
        raw = AABB(p["x"] - hx, p["y"] - hy, p["x"] + hx, p["y"] + hy)
    else:
        xs, ys = [], []
        for g in boxes + spheres:
            r = g["size"][0]
            xs += [g["x"] - r, g["x"] + r]
            ys += [g["y"] - r, g["y"] + r]
        raw = AABB(min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    # Translate so the min corner is the origin (keeps coordinates non-negative).
    dx, dy = -raw.min_x, -raw.min_y
    bounds = AABB(0, 0, raw.max_x - raw.min_x, raw.max_y - raw.min_y)

    obstacles = []
    for b in boxes:
        hx, hy = b["size"][0], b["size"][1]
        ob = AABB(b["x"] + dx - hx, b["y"] + dy - hy, b["x"] + dx + hx, b["y"] + dy + hy)
        # Clip to room; drop anything degenerate or fully outside.
        ob = AABB(max(ob.min_x, bounds.min_x), max(ob.min_y, bounds.min_y),
                  min(ob.max_x, bounds.max_x), min(ob.max_y, bounds.max_y))
        if ob.max_x - ob.min_x > 1e-3 and ob.max_y - ob.min_y > 1e-3:
            obstacles.append(ob)

    # Ball: prefer a red sphere, then one named ball/sphere, else the smallest.
    ball_geom = (next((s for s in spheres if _is_redish(s["rgba"])), None)
                 or next((s for s in spheres if "ball" in s["name"] or "sphere" in s["name"]), None)
                 or min(spheres, key=lambda s: s["size"][0]))
    ball = (ball_geom["x"] + dx, ball_geom["y"] + dy)

    start = _place_agent(bounds, obstacles, agent_radius, ball)
    return Scene(bounds, obstacles, ball, start, agent_radius)


def mjcf_zip_to_scene(zip_path: str, **kw) -> Scene:
    """Import the first .xml found inside an exported MJCF archive."""
    with zipfile.ZipFile(zip_path) as z:
        xmls = [n for n in z.namelist() if n.endswith(".xml")]
        if not xmls:
            raise ValueError(f"no .xml in {zip_path}")
        return mjcf_to_scene(z.read(xmls[0]).decode("utf-8", "replace"), **kw)
