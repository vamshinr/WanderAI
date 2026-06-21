import math
from wanderai.antim_import import mjcf_to_scene
from wanderai.occupancy import OccupancyGrid
from wanderai.distance_field import DistanceField

SAMPLE = """
<mujoco model="room">
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 0"/>
    <body name="box1" pos="1 0 0.25"><geom type="box" size="0.3 0.5 0.25"/></body>
    <body name="box2" pos="-1.5 1 0.25"><geom type="box" size="0.4 0.4 0.25"/></body>
    <body name="red_ball" pos="2 -2 0.1"><geom type="sphere" size="0.1" rgba="1 0 0 1"/></body>
  </worldbody>
</mujoco>
"""


def test_bounds_from_plane_translated_to_origin():
    s = mjcf_to_scene(SAMPLE)
    assert math.isclose(s.bounds.min_x, 0.0) and math.isclose(s.bounds.min_y, 0.0)
    assert math.isclose(s.bounds.max_x, 6.0) and math.isclose(s.bounds.max_y, 6.0)


def test_obstacles_and_ball_positions():
    s = mjcf_to_scene(SAMPLE)
    assert len(s.obstacles) == 2
    # red ball at world (2,-2) -> translated (+3,+3) -> (5,1)
    assert math.isclose(s.ball[0], 5.0, abs_tol=1e-6)
    assert math.isclose(s.ball[1], 1.0, abs_tol=1e-6)


def test_imported_scene_is_solvable():
    s = mjcf_to_scene(SAMPLE)
    assert s.is_free(*s.ball)
    assert s.is_free(s.agent_start.x, s.agent_start.y)
    grid = OccupancyGrid.from_scene(s, 0.1)
    field = DistanceField.from_grid(grid, s.ball)
    assert math.isfinite(field.query(s.agent_start.x, s.agent_start.y))


def test_red_sphere_preferred_over_others():
    xml = """
    <mujoco><worldbody>
      <geom type="plane" size="2 2 0.1" pos="0 0 0"/>
      <body pos="0 1 0.1"><geom type="sphere" size="0.3" rgba="0 0 1 1"/></body>
      <body pos="1 -1 0.1"><geom type="sphere" size="0.1" rgba="1 0 0 1"/></body>
    </worldbody></mujoco>
    """
    s = mjcf_to_scene(xml)
    # red ball at world (1,-1) -> +2 -> (3,1)
    assert math.isclose(s.ball[0], 3.0, abs_tol=1e-6)
    assert math.isclose(s.ball[1], 1.0, abs_tol=1e-6)
