from wanderai.environment import Action, SceneSearchEnv
from wanderai.scene import default_scene
from wanderai.llm_policy import LLMPolicy, parse_action, build_messages


def test_parse_action_explicit_line():
    assert parse_action("I see open space.\nACTION=TURN_LEFT") == Action.TURN_LEFT
    assert parse_action("ACTION = MOVE_FORWARD") == Action.MOVE_FORWARD


def test_parse_action_falls_back_to_last_token():
    # No ACTION= line: take the last token mentioned.
    assert parse_action("maybe TURN_LEFT, but better TURN_RIGHT") == Action.TURN_RIGHT


def test_parse_action_default_when_unparseable():
    assert parse_action("hmm, not sure") == Action.MOVE_FORWARD
    assert parse_action("") == Action.MOVE_FORWARD


def test_build_messages_includes_observation():
    msgs = build_messages("Red ball: not visible.")
    assert msgs[0]["role"] == "system"
    assert "Red ball: not visible." in msgs[1]["content"]


def test_llm_policy_act_uses_completion(monkeypatch):
    # Stub the network call: act() must turn the model text into an Action.
    env = SceneSearchEnv(default_scene())
    env.reset()
    pol = LLMPolicy(api_key="test")
    monkeypatch.setattr(pol, "_complete", lambda messages: "scan around\nACTION=TURN_RIGHT")
    assert pol.act(None, env) == Action.TURN_RIGHT


def test_llm_policy_never_crashes_on_error(monkeypatch):
    env = SceneSearchEnv(default_scene())
    env.reset()
    pol = LLMPolicy(api_key="test")

    def boom(messages):
        raise RuntimeError("network down")
    monkeypatch.setattr(pol, "_complete", boom)
    assert pol.act(None, env) == Action.MOVE_FORWARD     # graceful fallback
