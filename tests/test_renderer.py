import math
import numpy as np
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.renderer import StubRenderer, ball_visible


def test_visible_when_facing_clear():
    s = Scene(AABB(0, 0, 6, 6), [], (5, 3), Pose(1, 3, 0.0), 0.0)  # ball straight ahead (+x)
    assert ball_visible(s, Pose(1, 3, 0.0), math.pi / 2, 8.0)
    assert not ball_visible(s, Pose(1, 3, math.pi), math.pi / 2, 8.0)  # facing away


def test_occluded_by_obstacle():
    s = Scene(AABB(0, 0, 6, 6), [AABB(2.8, 2.5, 3.2, 3.5)], (5, 3), Pose(1, 3, 0.0), 0.0)
    assert not ball_visible(s, Pose(1, 3, 0.0), math.pi / 2, 8.0)  # box between agent & ball


def test_render_shape_and_red_when_visible():
    r = StubRenderer()
    s = Scene(AABB(0, 0, 6, 6), [], (5, 3), Pose(1, 3, 0.0), 0.0)
    img = r.render(s, Pose(1, 3, 0.0))
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8
    assert img[:, :, 0].max() > 150 and img[:, :, 1].max() < 120  # has red, little green
    blank = r.render(s, Pose(1, 3, math.pi))
    assert blank[:, :, 0].max() < 130   # no red band when facing away
