import math
from wanderai.geometry import (
    Pose, AABB, segment_intersects_aabb, wrap_angle, ray_aabb,
)


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


def test_ray_aabb_entry_from_outside():
    # Ray from (0,1) heading +x toward a box spanning x in [2,3].
    box = AABB(2, 0, 3, 2)
    hit = ray_aabb(0, 1, 1, 0, box)
    assert hit is not None
    tmin, tmax = hit
    assert math.isclose(tmin, 2.0)   # enters front face at x=2
    assert math.isclose(tmax, 3.0)   # exits back face at x=3


def test_ray_aabb_exit_from_inside():
    # Origin inside the box -> tmin negative, tmax positive (exit distance).
    box = AABB(0, 0, 4, 4)
    hit = ray_aabb(2, 2, 1, 0, box)
    assert hit is not None
    tmin, tmax = hit
    assert tmin < 0 and math.isclose(tmax, 2.0)   # wall at x=4 is 2 away


def test_ray_aabb_miss():
    box = AABB(2, 2, 3, 3)
    assert ray_aabb(0, 0, 1, 0, box) is None   # heading +x along y=0, misses
