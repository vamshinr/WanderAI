"""Import a MuJoCo MJCF scene (e.g. exported from Antim/Gizmo) into our `Scene`.

Reality of Gizmo exports (learned from a real one): the room *shell* (floor,
walls, baseboards, window frames) is inline `mesh` geoms with vertex data, but
**furniture bodies carry only a position — their meshes are referenced externally
and aren't in the export.** And Gizmo doesn't place our goal object. So we:
  * take room bounds from the floor mesh,
  * skip the wall shell,
  * turn each floor-level furniture *body* into a default-sized box obstacle
    (we only know where it is, not its exact shape),
  * drop our own red ball into a free, reachable spot (unless the MJCF already
    has a red/named sphere).
The result is always a solvable search scene.

Footprints are axis-aligned (rotations ignored). Plain box/sphere/plane
primitives are handled too, so non-Gizmo MJCF (incl. our Phase-B MuJoCo scenes)
imports as well."""

from __future__ import annotations
import math
import os
import xml.etree.ElementTree as ET
import zipfile
from .geometry import AABB, Pose
from .scene import Scene
from .occupancy import OccupancyGrid
from .distance_field import DistanceField

_SHELL = ("wall", "baseboard", "window", "frame", "glass", "ceiling",
          "door", "curtain", "blind", "skirting", "floor")
_PASSABLE = ("rug", "mat", "carpet")
_FLOOR_Z_MAX = 1.2              # ignore wall-mounted / ceiling objects above this
_DEFAULT_HALF = 0.3            # half-extent for furniture whose shape is unknown


def _floats(s, n=3, default=0.0):
    parts = [float(v) for v in (s or "").split()]
    parts += [default] * (n - len(parts))
    return parts[:n]


def _tag(e):
    return e.tag.split("}")[-1]


def _mesh_aabbs(root):
    out = {}
    asset = root.find("asset")
    for m in (asset.findall("mesh") if asset is not None else []):
        if m.get("vertex"):
            v = [float(x) for x in m.get("vertex").split()]
            xs, ys = v[0::3], v[1::3]
            if xs and ys:
                out[m.get("name")] = (min(xs), min(ys), max(xs), max(ys))
    return out


def _materials(root):
    out = {}
    asset = root.find("asset")
    for mat in (asset.findall("material") if asset is not None else []):
        if mat.get("rgba"):
            out[mat.get("name")] = _floats(mat.get("rgba"), 4, default=1.0)
    return out


def _is_redish(rgba):
    return rgba is not None and rgba[0] > 0.5 and rgba[1] < 0.4 and rgba[2] < 0.4


def _is_shell(name):
    return any(k in name for k in _SHELL)


def _collect(elem, ox, oy, oz, body, meshes, mats, geoms, bodies):
    """Recursively gather geom footprints and body world positions."""
    for child in elem:
        t = _tag(child)
        if t == "body":
            bx, by, bz = _floats(child.get("pos"), 3)
            wx, wy, wz = ox + bx, oy + by, oz + bz
            name = child.get("name", body)
            bodies.append({"name": (name or "").lower(), "x": wx, "y": wy, "z": wz})
            _collect(child, wx, wy, wz, name, meshes, mats, geoms, bodies)
            continue
        if t == "geom":
            gx, gy, gz = _floats(child.get("pos"), 3)
            wx, wy, wz = ox + gx, oy + gy, oz + gz
            gtype = child.get("type", "mesh")
            rgba = _floats(child.get("rgba"), 4, default=1.0) if child.get("rgba") else None
            if rgba is None and child.get("material"):
                rgba = mats.get(child.get("material"))
            fp = None
            if gtype == "mesh":
                la = meshes.get(child.get("mesh"))
                if la:
                    fp = AABB(wx + la[0], wy + la[1], wx + la[2], wy + la[3])
            elif gtype in ("box", "plane"):
                hx, hy, _ = _floats(child.get("size"), 3)
                if hx > 0 and hy > 0:
                    fp = AABB(wx - hx, wy - hy, wx + hx, wy + hy)
            geoms.append({"type": gtype, "x": wx, "y": wy, "z": wz, "rgba": rgba,
                          "name": (child.get("name") or body or "").lower(),
                          "body": (body or "").lower(), "footprint": fp})
        else:
            _collect(child, ox, oy, oz, body, meshes, mats, geoms, bodies)


def _place_agent(bounds, obstacles, agent_radius, ball, step=0.4):
    probe = Scene(bounds, obstacles, ball, Pose(0, 0, 0), agent_radius)
    field = DistanceField.from_grid(OccupancyGrid.from_scene(probe, 0.1), ball)
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
    if best is None:
        best = (bounds.min_x + agent_radius * 2, bounds.min_y + agent_radius * 2)
    return Pose(best[0], best[1], math.atan2(ball[1] - best[1], ball[0] - best[0]))


def _place_ball(bounds, obstacles, agent_radius, step=0.3):
    """Free cell farthest from room center (a corner) — a good search target."""
    probe = Scene(bounds, obstacles, (0, 0), Pose(0, 0, 0), agent_radius)
    cx, cy = (bounds.min_x + bounds.max_x) / 2, (bounds.min_y + bounds.max_y) / 2
    best, best_d = None, -1.0
    y = bounds.min_y + agent_radius
    while y < bounds.max_y:
        x = bounds.min_x + agent_radius
        while x < bounds.max_x:
            if probe.is_free(x, y):
                d = (x - cx) ** 2 + (y - cy) ** 2
                if d > best_d:
                    best_d, best = d, (x, y)
            x += step
        y += step
    if best is None:
        raise ValueError("imported room has no free space for a ball")
    return best


def mjcf_to_scene(source: str, agent_radius: float = 0.2, margin: float = 0.2) -> Scene:
    """Parse MJCF (XML string or path to .xml) into a solvable Scene."""
    xml = source
    if "\n" not in source and os.path.exists(source) and source.endswith(".xml"):
        with open(source) as fh:
            xml = fh.read()
    root = ET.fromstring(xml)
    world = root.find(".//worldbody")
    if world is None:
        raise ValueError("MJCF has no <worldbody>")
    geoms, bodies = [], []
    _collect(world, 0.0, 0.0, 0.0, "", _mesh_aabbs(root), _materials(root), geoms, bodies)

    # Bounds from the floor: a plane geom (primitive scenes) or a geom named
    # 'floor' (Gizmo's floor mesh), else the bbox of everything with a footprint.
    floor = (next((g for g in geoms if g["type"] == "plane" and g["footprint"]), None)
             or next((g for g in geoms
                      if "floor" in (g["name"] + " " + g["body"]) and g["footprint"]), None))
    if floor is not None:
        raw = floor["footprint"]
    else:
        fps = [g["footprint"] for g in geoms if g["footprint"]]
        if not fps:
            raise ValueError("MJCF has no usable geometry for bounds")
        raw = AABB(min(f.min_x for f in fps) - margin, min(f.min_y for f in fps) - margin,
                   max(f.max_x for f in fps) + margin, max(f.max_y for f in fps) + margin)

    dx, dy = -raw.min_x, -raw.min_y
    bounds = AABB(0, 0, raw.max_x - raw.min_x, raw.max_y - raw.min_y)
    room_area = bounds.max_x * bounds.max_y

    def clip(fp):
        ob = AABB(max(fp.min_x + dx, 0), max(fp.min_y + dy, 0),
                  min(fp.max_x + dx, bounds.max_x), min(fp.max_y + dy, bounds.max_y))
        if ob.max_x - ob.min_x <= 1e-3 or ob.max_y - ob.min_y <= 1e-3:
            return None
        if (ob.max_x - ob.min_x) * (ob.max_y - ob.min_y) > 0.8 * room_area:
            return None                # spans the room — it's the floor, not an obstacle
        return ob

    # Ball from an existing red/named sphere (if any), so we don't turn it into
    # an obstacle below.
    ball_geom = next((g for g in geoms if g["type"] == "sphere"
                      and (_is_redish(g["rgba"]) or "ball" in g["name"] or "sphere" in g["name"])),
                     None)
    ball_body = ball_geom["body"] if ball_geom else None

    obstacles = []
    geom_obstacle_bodies = set()
    for g in geoms:
        if g is floor or g["footprint"] is None or g["type"] in ("sphere", "plane"):
            continue
        name = g["name"] + " " + g["body"]
        if _is_shell(name) or any(k in name for k in _PASSABLE) or g["z"] > _FLOOR_Z_MAX:
            continue
        ob = clip(g["footprint"])
        if ob:
            obstacles.append(ob)
            geom_obstacle_bodies.add(g["body"])

    # Furniture bodies with no inline geometry → default-sized box at their spot.
    for b in bodies:
        name = b["name"]
        if (_is_shell(name) or any(k in name for k in _PASSABLE) or b["z"] > _FLOOR_Z_MAX
                or name == ball_body or name in geom_obstacle_bodies):
            continue
        h = _DEFAULT_HALF
        ob = clip(AABB(b["x"] - h, b["y"] - h, b["x"] + h, b["y"] + h))
        if ob:
            obstacles.append(ob)

    if ball_geom is not None:
        ball = (ball_geom["x"] + dx, ball_geom["y"] + dy)
    else:
        ball = _place_ball(bounds, obstacles, agent_radius)
    start = _place_agent(bounds, obstacles, agent_radius, ball)
    return Scene(bounds, obstacles, ball, start, agent_radius)


def mjcf_zip_to_scene(zip_path: str, **kw) -> Scene:
    """Import the first .xml found inside an exported MJCF archive."""
    with zipfile.ZipFile(zip_path) as z:
        xmls = [n for n in z.namelist() if n.endswith(".xml")]
        if not xmls:
            raise ValueError(f"no .xml in {zip_path}")
        return mjcf_to_scene(z.read(xmls[0]).decode("utf-8", "replace"), **kw)
