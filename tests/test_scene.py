from wanderai.scene import Scene, default_scene
from wanderai.geometry import AABB, Pose


def test_is_free_respects_bounds_and_obstacles():
    s = Scene(bounds=AABB(0, 0, 6, 6), obstacles=[AABB(2, 2, 4, 4)],
              ball=(5.5, 5.5), agent_start=Pose(0.5, 0.5, 0.0), agent_radius=0.2)
    assert s.is_free(0.5, 0.5)         # open floor
    assert not s.is_free(3, 3)         # inside obstacle
    assert not s.is_free(1.9, 3)       # within agent_radius of obstacle
    assert not s.is_free(-0.1, 3)      # outside bounds


def test_default_scene_valid():
    s = default_scene()
    assert s.is_free(s.agent_start.x, s.agent_start.y)
    assert s.is_free(*s.ball)
    assert s.bounds.contains(*s.ball)
