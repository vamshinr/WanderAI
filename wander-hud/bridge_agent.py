"""Bridge agent: drive the HUD scene-search env with OUR flat-text model.

Our fine-tuned model speaks `ACTION=<token>` text, not OpenAI tool-calls — so the
stock openai_compatible agent scored 0 (its tool call leaked as text). This agent
overrides get_response: it pulls the latest observation out of the conversation,
asks our model with the exact flat prompt it was trained on, parses the action,
and emits the matching move() tool call. HUD still records a real trace + reward.
"""
import asyncio
import os

from hud.agents.openai_compatible.agent import OpenAIChatAgent
from hud.agents.types import AgentStep
from hud.types import MCPToolCall

from wanderai.llm_policy import LLMPolicy, build_messages, parse_action

_ACTION_INT = {"MOVE_FORWARD": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2}
_OBS_MARKER = "Clearance"          # appears in every observation
_DONE_MARKERS = ("FOUND THE RED BALL", "episode over")


def _msg_text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


class WanderBridgeAgent(OpenAIChatAgent):
    def __init__(self, config=None, wander_model=None):
        super().__init__(config=config)
        # None for our llama fine-tunes; "low" lets reasoning models (gpt-oss) answer cleanly.
        reasoning = os.environ.get("WANDER_REASONING") or None
        self._pol = LLMPolicy(model=wander_model, reasoning_effort=reasoning, temperature=0.0)

    def _latest_observation(self, state) -> str:
        for m in reversed(state.messages):
            text = _msg_text(m)
            if _OBS_MARKER in text:
                return text
        return _msg_text(state.messages[-1]) if state.messages else ""

    async def get_response(self, state, *, system_prompt=None, citations_enabled=False) -> AgentStep:
        obs = self._latest_observation(state)
        # Our model's inference path, exactly as trained (blocking → thread).
        text = await asyncio.to_thread(self._pol._complete, build_messages(obs))
        if any(mk in obs for mk in _DONE_MARKERS):
            return AgentStep(content=text, tool_calls=[], done=True)
        action = parse_action(text)
        call = MCPToolCall(name="move", arguments={"action": _ACTION_INT[action.name]})
        return AgentStep(content=text, tool_calls=[call], done=False)
