import math
import numpy as np
from wanderai.scene_gen import random_scene, make_split
from wanderai.occupancy import OccupancyGrid
from wanderai.distance_field import DistanceField


def _reachable(scene):
    grid = OccupancyGrid.from_scene(scene, 0.1)
    field = DistanceField.from_grid(grid, scene.ball)
    return math.isfinite(field.query(scene.agent_start.x, scene.agent_start.y))


def test_random_scene_is_valid_and_solvable():
    rng = np.random.default_rng(0)
    for _ in range(10):
        s = random_scene(rng)
        assert s.is_free(s.agent_start.x, s.agent_start.y)   # start in free space
        assert s.is_free(*s.ball)                            # ball in free space
        assert s.bounds.contains(*s.ball)
        assert _reachable(s)                                 # ball reachable from start
        assert len(s.obstacles) >= 1


def test_random_scene_is_deterministic_by_seed():
    a = random_scene(np.random.default_rng(42))
    b = random_scene(np.random.default_rng(42))
    assert a.ball == b.ball
    assert a.agent_start == b.agent_start
    assert [tuple(o.__dict__.values()) for o in a.obstacles] == \
           [tuple(o.__dict__.values()) for o in b.obstacles]


def test_start_and_ball_are_separated():
    rng = np.random.default_rng(1)
    for _ in range(10):
        s = random_scene(rng)
        d = math.hypot(s.ball[0] - s.agent_start.x, s.ball[1] - s.agent_start.y)
        assert d >= 2.0   # not trivially adjacent


def test_make_split_disjoint_and_deterministic():
    train1, test1 = make_split(n_train=8, n_test=4, seed=7)
    train2, test2 = make_split(n_train=8, n_test=4, seed=7)
    assert len(train1) == 8 and len(test1) == 4
    # Deterministic given seed.
    assert train1[0].ball == train2[0].ball
    assert test1[0].agent_start == test2[0].agent_start
    # Train and test scenes differ (held-out set is genuinely separate).
    train_balls = {s.ball for s in train1}
    test_balls = {s.ball for s in test1}
    assert train_balls.isdisjoint(test_balls)
