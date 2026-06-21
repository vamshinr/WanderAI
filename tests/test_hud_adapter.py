import asyncio
import math

from wanderai.geometry import AABB, Pose
from wanderai.scene import Scene


PRIVILEGED_KEYS = {
    "ball",
    "geodesic",
    "goal",
    "hidden",
    "optimal",
    "path_length",
    "reward",
}


def _assert_no_privileged_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key not in PRIVILEGED_KEYS
            _assert_no_privileged_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_privileged_keys(nested)


def test_adapter_reset_clears_episode_state():
    from wanderai.hud_adapter import SceneSearchHudAdapter

    adapter = SceneSearchHudAdapter()
    adapter.reset_episode(scene_id="default", seed=7)
    stepped = adapter.move_forward()
    assert stepped["episode"]["steps"] == 1

    reset = adapter.reset_episode(scene_id="default", seed=7)

    assert reset["episode"]["steps"] == 0
    assert reset["episode"]["terminated"] is False
    assert reset["episode"]["scene_id"] == "default"
    assert reset["episode"]["seed"] == 7


def test_tool_responses_do_not_expose_privileged_state():
    from wanderai.hud_adapter import SceneSearchHudAdapter

    adapter = SceneSearchHudAdapter()
    responses = [
        adapter.look(scene_id="default", seed=11),
        adapter.move_forward(),
        adapter.turn_left(),
        adapter.turn_right(),
        adapter.episode_status(),
    ]

    for response in responses:
        _assert_no_privileged_keys(response)


def test_action_tools_step_episode():
    from wanderai.hud_adapter import SceneSearchHudAdapter

    adapter = SceneSearchHudAdapter()
    adapter.look(scene_id="default", seed=3)

    first = adapter.move_forward()
    second = adapter.turn_left()
    third = adapter.turn_right()

    assert first["episode"]["steps"] == 1
    assert second["episode"]["steps"] == 2
    assert third["episode"]["steps"] == 3
    assert first["action"] == "move_forward"
    assert second["action"] == "turn_left"
    assert third["action"] == "turn_right"


def test_mcp_server_and_env_exports_hud_surface():
    import importlib

    from wanderai.hud_adapter import SceneSearchHudAdapter, create_mcp_server

    server = create_mcp_server(SceneSearchHudAdapter())
    assert {"look", "move_forward", "turn_left", "turn_right", "episode_status"} <= set(
        server.tools
    )

    hud_env = importlib.import_module("env")
    assert hud_env.env.name == "wanderai-scene-search"
    assert callable(hud_env.find_red_ball)


def test_find_red_ball_template_resets_and_yields_reward():
    import importlib

    hud_env = importlib.import_module("env")

    async def _drive_template():
        template = getattr(hud_env.find_red_ball, "func", hud_env.find_red_ball)
        template_run = template(scene_id="default", seed=23)
        prompt = await template_run.__anext__()
        reward = await template_run.asend("done")
        return prompt, reward

    prompt, reward = asyncio.run(_drive_template())

    assert "red ball" in prompt
    assert hud_env.adapter.hidden_episode_state()["seed"] == 23
    assert reward == 0.0


def test_observation_summary_detects_red_target_bearing():
    from wanderai.hud_adapter import SceneSearchHudAdapter

    scene = Scene(
        bounds=AABB(0, 0, 6, 6),
        obstacles=[],
        ball=(3.0, 1.0),
        agent_start=Pose(1.0, 1.0, 0.0),
        agent_radius=0.2,
    )
    adapter = SceneSearchHudAdapter(scene_factory=lambda _scene_id, _seed: scene)

    response = adapter.look(scene_id="bearing-room", seed=5)

    assert response["target_visible"] is True
    assert response["target_bearing"] == "center"
    assert "center" in response["observation"]


def test_final_reward_uses_bounded_spl_from_hidden_episode_state():
    from wanderai.hud_adapter import SceneSearchHudAdapter

    scene = Scene(
        bounds=AABB(0, 0, 3, 3),
        obstacles=[],
        ball=(1.25, 1.0),
        agent_start=Pose(1.0, 1.0, 0.0),
        agent_radius=0.2,
    )
    adapter = SceneSearchHudAdapter(scene_factory=lambda _scene_id, _seed: scene)
    adapter.reset_episode(scene_id="short-room", seed=13)
    adapter.move_forward()

    reward = adapter.final_reward()
    hidden = adapter.hidden_episode_state()
    expected = hidden["optimal"] / max(hidden["path_length"], hidden["optimal"])

    assert hidden["success"] is True
    assert math.isclose(reward, expected)
    assert 0.0 <= reward <= 1.0
