from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.occupancy import OccupancyGrid


def _scene():
    return Scene(AABB(0, 0, 4, 4), [AABB(1, 1, 2, 2)], (3.5, 3.5), Pose(0.5, 0.5, 0), 0.0)


def test_grid_shape_and_roundtrip():
    g = OccupancyGrid.from_scene(_scene(), cell_size=0.5)
    assert g.blocked.shape == (8, 8)
    r, c = g.world_to_cell(0.25, 0.25)
    assert (r, c) == (0, 0)
    x, y = g.cell_to_world(0, 0)
    assert abs(x - 0.25) < 1e-9 and abs(y - 0.25) < 1e-9


def test_obstacle_cells_blocked():
    g = OccupancyGrid.from_scene(_scene(), cell_size=0.5)
    assert g.is_blocked_world(1.5, 1.5)     # inside obstacle
    assert not g.is_blocked_world(0.25, 0.25)
    assert g.is_blocked_world(-1, 2)        # out of bounds


def test_world_to_cell_uses_floor_for_negative_offsets():
    s = Scene(AABB(0, 0, 1, 1), [], (0.5, 0.5), Pose(0.5, 0.5, 0), 0.0)
    g = OccupancyGrid.from_scene(s, cell_size=0.5)

    assert g.world_to_cell(-0.01, 0.25) == (0, -1)
    assert g.is_blocked_world(-0.01, 0.25)


def test_grid_dimensions_cover_partial_edge_cells():
    s = Scene(AABB(0, 0, 1, 1), [], (0.95, 0.95), Pose(0.05, 0.05, 0), 0.0)
    g = OccupancyGrid.from_scene(s, cell_size=0.3)

    assert g.blocked.shape == (4, 4)
    assert not g.is_blocked_world(0.95, 0.95)


def test_cell_size_must_be_positive():
    s = Scene(AABB(0, 0, 1, 1), [], (0.5, 0.5), Pose(0.5, 0.5, 0), 0.0)

    try:
        OccupancyGrid.from_scene(s, cell_size=0.0)
    except ValueError as exc:
        assert "cell_size" in str(exc)
    else:
        raise AssertionError("expected ValueError for zero cell_size")
