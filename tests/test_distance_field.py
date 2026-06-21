import math
import numpy as np
import pytest
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.occupancy import OccupancyGrid
from wanderai.distance_field import DistanceField


def _empty_grid():
    s = Scene(AABB(0, 0, 4, 4), [], (3.5, 0.5), Pose(0.5, 0.5, 0), 0.0)
    return OccupancyGrid.from_scene(s, 0.5), (3.5, 0.5)


def test_distance_zero_at_ball():
    g, ball = _empty_grid()
    df = DistanceField.from_grid(g, ball)
    assert df.query(*ball) < 0.5


def test_open_field_matches_euclidean():
    g, ball = _empty_grid()
    df = DistanceField.from_grid(g, ball)
    d = df.query(0.5, 0.5)
    euclid = math.hypot(3.5 - 0.5, 0.5 - 0.5)
    assert abs(d - euclid) < 0.6   # 8-connected approx on a coarse grid


def test_geodesic_exceeds_euclidean_with_wall():
    # Wall from y=0..3 at x~2 forces a detour around the top.
    s = Scene(AABB(0, 0, 4, 4), [AABB(1.9, 0.0, 2.1, 3.0)], (3.5, 0.5), Pose(0.5, 0.5, 0), 0.0)
    g = OccupancyGrid.from_scene(s, 0.25)
    df = DistanceField.from_grid(g, (3.5, 0.5))
    geo = df.query(0.5, 0.5)
    euclid = math.hypot(3.0, 0.0)
    assert geo > euclid + 1.0      # must detour around the wall


def test_unreachable_is_inf():
    # Ball sealed in a corner box.
    s = Scene(AABB(0, 0, 4, 4),
              [AABB(2, 0, 2.2, 4), AABB(0, 2, 4, 2.2)], (3.5, 3.5), Pose(0.5, 0.5, 0), 0.0)
    g = OccupancyGrid.from_scene(s, 0.2)
    df = DistanceField.from_grid(g, (3.5, 3.5))
    assert math.isinf(df.query(0.5, 0.5))


def test_diagonal_neighbors_do_not_cut_blocked_corners():
    blocked = np.array([
        [False, True],
        [True, False],
    ], dtype=bool)
    g = OccupancyGrid(blocked, cell_size=1.0, origin=(0.0, 0.0))

    df = DistanceField.from_grid(g, (1.5, 1.5))

    assert math.isinf(df.query(0.5, 0.5))


def test_blocked_distance_field_goal_is_rejected():
    blocked = np.array([[True]], dtype=bool)
    g = OccupancyGrid(blocked, cell_size=1.0, origin=(0.0, 0.0))

    with pytest.raises(ValueError, match="goal"):
        DistanceField.from_grid(g, (0.5, 0.5))
