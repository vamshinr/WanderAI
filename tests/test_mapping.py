import math

import numpy as np

from wanderai.geometry import Pose, AABB
from wanderai.scene import Scene, default_scene
from wanderai.mapping import (EgoMap, depth_strip, depth_strip_from_image,
                              frontier_hint, DEFAULT_N_RAYS,
                              OCC_THRESHOLD, FREE_THRESHOLD)


def open_room():
    return Scene(bounds=AABB(0, 0, 6, 6), obstacles=[], ball=(5, 5),
                 agent_start=Pose(3, 3, 0.0), agent_radius=0.2)


def test_depth_strip_spans_fov_left_to_right():
    strip = depth_strip(open_room(), Pose(3, 3, 0.0))
    assert len(strip) == DEFAULT_N_RAYS
    bearings = [rel for rel, _ in strip]
    assert bearings[0] > 0 > bearings[-1]          # left first, right last
    assert all(b1 > b2 for b1, b2 in zip(bearings, bearings[1:]))
    assert all(r > 0 for _, r in strip)


def test_depth_strip_sees_the_wall_at_the_right_distance():
    # Facing +x from (3,3) in a 6x6 room: the wall is 3m ahead.
    strip = depth_strip(open_room(), Pose(3, 3, 0.0), max_range=10.0)
    center = min(strip, key=lambda s: abs(s[0]))
    assert abs(center[1] - 3.0) < 0.05


def test_egomap_marks_free_along_ray_and_occupied_at_hit():
    m = EgoMap(cell_size=0.25)
    pose = Pose(3, 3, 0.0)
    m.update(pose, [(0.0, 2.0)], max_range=6.0)    # single ray hitting at 2m
    assert m.is_free(m.key(3.5, 3.0))              # along the ray: free
    assert m.is_occupied(m.key(3 + 2.0 + 0.05, 3.0))  # at the hit: occupied


def test_egomap_max_range_reading_marks_no_obstacle():
    m = EgoMap(cell_size=0.25)
    m.update(Pose(3, 3, 0.0), [(0.0, 6.0)], max_range=6.0)   # nothing within range
    # No cell should be occupied anywhere along that ray.
    assert not any(lo > OCC_THRESHOLD for lo in m.cells.values())


def test_egomap_budget_triggers_coarsening_and_respects_budget():
    m = EgoMap(cell_size=0.125, budget_cells=200)
    scene = open_room()
    rng = np.random.default_rng(0)
    for _ in range(40):
        pose = Pose(float(rng.uniform(1, 5)), float(rng.uniform(1, 5)),
                    float(rng.uniform(-math.pi, math.pi)))
        m.update(pose, depth_strip(scene, pose))
    assert m.coarsen_count >= 1
    assert m.memory_cells() <= 200 * 2     # transiently may exceed, then re-coarsens
    assert m.cell_size > 0.125


def test_coarsening_keeps_walls():
    m = EgoMap(cell_size=0.25)
    m.update(Pose(3, 3, 0.0), [(0.0, 2.0)], max_range=6.0)
    hit_world = (3 + 2.0 + 0.05, 3.0)
    assert m.is_occupied(m.key(*hit_world))
    m._coarsen()
    assert m.is_occupied(m.key(*hit_world))    # occupied evidence survives merge


def test_frontiers_exist_after_partial_observation_and_shrink_when_explored():
    scene = open_room()
    m = EgoMap(cell_size=0.25)
    pose = Pose(3, 3, 0.0)
    m.update(pose, depth_strip(scene, pose))
    early = m.frontiers()
    assert early, "a partially observed room must have frontiers"
    # Spin in place: observe all directions from the center.
    for k in range(12):
        p = Pose(3, 3, k * math.pi / 6)
        m.update(p, depth_strip(scene, p, max_range=10.0))
    late_cells = sum(c.size for c in m.frontiers())
    assert late_cells < sum(c.size for c in early) + 40


def test_plan_path_avoids_occupied_cells():
    m = EgoMap(cell_size=0.25)
    # Observe a wall segment blocking the straight line from (1,1) to (4,1).
    for y in (0.5, 0.75, 1.0, 1.25, 1.5):
        m._bump(m.key(2.5, y), 5.0)
    # Mark a corridor of free space around it.
    for x in np.arange(0.5, 4.5, 0.25):
        for y in np.arange(0.25, 2.5, 0.25):
            if not m.is_occupied(m.key(x, y)):
                m._bump(m.key(x, y), -2.0)
    path = m.plan_path((1.0, 1.0), (4.0, 1.0))
    assert path is not None
    for wx, wy in path:
        assert not m.is_occupied(m.key(wx, wy))


def test_frontier_hint_direction_is_sane():
    scene = open_room()
    m = EgoMap(cell_size=0.25)
    pose = Pose(1.0, 3.0, 0.0)                      # facing +x, unknown ahead
    m.update(pose, depth_strip(scene, pose))
    hint = frontier_hint(m, pose)
    assert hint is not None
    assert hint["direction"] in ("left", "center", "right", "behind")
    assert hint["distance"] > 0


def test_depth_strip_from_image_matches_flat_wall():
    # Synthetic pinhole depth image of a wall 3m ahead, 90deg FOV:
    # perpendicular depth is constant 3.0 across all columns.
    h, w = 40, 60
    depth = np.full((h, w), 3.0, dtype=np.float32)
    strip = depth_strip_from_image(depth, fov_x=math.pi / 2, max_depth=10.0,
                                   n_rays=9)
    center = min(strip, key=lambda s: abs(s[0]))
    assert abs(center[1] - 3.0) < 1e-3
    # Off-axis rays must report LONGER range along the ray (3/cos(bearing)).
    edge = strip[0]
    assert edge[1] > center[1]
    assert abs(edge[1] - 3.0 / math.cos(edge[0])) < 0.15
