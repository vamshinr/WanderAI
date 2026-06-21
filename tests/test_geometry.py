import math
from wanderai.geometry import Pose, AABB, segment_intersects_aabb, wrap_angle


def test_aabb_contains_and_inflate():
    b = AABB(0, 0, 2, 2)
    assert b.contains(1, 1)
    assert not b.contains(3, 1)
    big = b.inflate(0.5)
    assert big.min_x == -0.5 and big.max_x == 2.5
    assert big.contains(-0.4, 1)


def test_segment_intersects_aabb():
    b = AABB(1, 1, 2, 2)
    assert segment_intersects_aabb(0, 1.5, 3, 1.5, b)      # passes through
    assert not segment_intersects_aabb(0, 0, 0.5, 0.5, b)  # misses
    assert segment_intersects_aabb(1.5, 0, 1.5, 3, b)      # vertical through


def test_wrap_angle():
    assert math.isclose(wrap_angle(3 * math.pi), math.pi)
    assert math.isclose(wrap_angle(-3 * math.pi), math.pi)
    assert math.isclose(wrap_angle(0.5), 0.5)
