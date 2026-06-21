import math
from wanderai.scene import Scene
from wanderai.geometry import AABB, Pose
from wanderai.environment import Action
from wanderai.observation import cast_ray, observe, observation_text


def _empty_room():
    return Scene(AABB(0, 0, 4, 4), [], (3.5, 2.0), Pose(2, 2, 0.0), 0.0)


def test_cast_ray_hits_room_wall():
    s = _empty_room()
    # Facing +x from center -> wall at x=4 is 2.0 away.
    assert math.isclose(cast_ray(s, 2, 2, 0.0, max_range=10), 2.0, abs_tol=1e-6)
    # Facing +y -> wall at y=4 is 2.0 away.
    assert math.isclose(cast_ray(s, 2, 2, math.pi / 2, max_range=10), 2.0, abs_tol=1e-6)


def test_cast_ray_hits_obstacle_before_wall():
    s = Scene(AABB(0, 0, 4, 4), [AABB(3.0, 1.0, 3.5, 3.0)], (3.7, 3.7), Pose(2, 2, 0), 0.0)
    # Facing +x from (2,2): obstacle front face at x=3 is 1.0 away (closer than wall).
    assert math.isclose(cast_ray(s, 2, 2, 0.0, max_range=10), 1.0, abs_tol=1e-6)


def test_cast_ray_capped_at_max_range():
    s = _empty_room()
    assert cast_ray(s, 2, 2, 0.0, max_range=0.5) == 0.5


def test_cast_ray_accounts_for_agent_radius():
    # Obstacle front face at x=3; with a 0.2m agent radius the body hits at x=2.8.
    s = Scene(AABB(0, 0, 4, 4), [AABB(3.0, 1.0, 3.5, 3.0)], (3.7, 3.7), Pose(2, 2, 0), 0.2)
    # facing +x from (2,2): clearance should be 0.8 (to inflated face), not 1.0.
    assert math.isclose(cast_ray(s, 2, 2, 0.0, max_range=10), 0.8, abs_tol=1e-6)


def test_observe_ball_visible_ahead():
    # Ball straight ahead (+x), clear line of sight.
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 3.0), Pose(1.0, 3.0, 0.0), 0.0)
    obs = observe(s, Pose(1.0, 3.0, 0.0))
    assert obs.ball_visible
    assert abs(obs.ball_bearing) < 1e-6          # dead ahead
    assert math.isclose(obs.ball_distance, 4.0, abs_tol=1e-6)


def test_observe_ball_hidden_when_facing_away():
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 3.0), Pose(1.0, 3.0, 0.0), 0.0)
    obs = observe(s, Pose(1.0, 3.0, math.pi))    # facing -x, away from ball
    assert not obs.ball_visible
    assert obs.ball_distance is None


def test_observe_clearance_keys_and_memory():
    s = _empty_room()
    obs = observe(s, Pose(2, 2, 0.0), history=[Action.MOVE_FORWARD, Action.TURN_LEFT])
    assert set(obs.clearance) == {"left", "center", "right"}
    assert all(v > 0 for v in obs.clearance.values())
    assert obs.recent_actions[-1] == "TURN_LEFT"


def test_observe_visited_areas_memory():
    from wanderai.observation import visit_key
    s = _empty_room()
    # Mark the cell ~1m ahead (+x) as already visited.
    ahead = visit_key(2 + 1.0, 2)
    obs = observe(s, Pose(2, 2, 0.0), visited={ahead})
    assert obs.explored["center"] is True       # straight ahead is explored
    assert obs.explored["left"] is False         # sides are new
    assert obs.n_visited == 1


def test_observation_text_includes_explored():
    s = _empty_room()
    txt = observation_text(observe(s, Pose(2, 2, 0.0), visited=set()))
    assert "Explored" in txt and "NEW" in txt


def test_observation_text_is_readable():
    s = Scene(AABB(0, 0, 6, 6), [], (5.0, 3.0), Pose(1.0, 3.0, 0.0), 0.0)
    txt = observation_text(observe(s, Pose(1.0, 3.0, 0.0)))
    assert "ball" in txt.lower()
    assert "VISIBLE" in txt
    assert "clearance" in txt.lower()
