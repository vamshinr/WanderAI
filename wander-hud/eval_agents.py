"""Baseline agents for the WanderAI HUD eval suite.

RandomBridgeAgent = the FLOOR: it ignores the observation and emits a uniformly
random move() each step (seeded for reproducibility), stopping when the env says
the ball was found or the episode ended. It produces a real HUD trace + reward,
so "random" sits on the same leaderboard as our RFT model and the Claude ceiling.

Self-contained (no import from bridge_agent) so it's robust to edits there.
"""
import random

from hud.agents.openai_compatible.agent import OpenAIChatAgent
from hud.agents.types import AgentStep
from hud.types import MCPToolCall

_OBS_MARKER = "Clearance"                       # appears in every observation
_DONE_MARKERS = ("FOUND THE RED BALL", "episode over")


def _msg_text(m) -> str:
    c = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
    return ""


class RandomBridgeAgent(OpenAIChatAgent):
    def __init__(self, config=None, seed: int = 0):
        super().__init__(config=config)
        self._rng = random.Random(seed)         # reproducible floor

    def _latest_obs(self, state) -> str:
        for m in reversed(state.messages):
            t = _msg_text(m)
            if _OBS_MARKER in t:
                return t
        return _msg_text(state.messages[-1]) if state.messages else ""

    async def get_response(self, state, *, system_prompt=None, citations_enabled=False) -> AgentStep:
        obs = self._latest_obs(state)
        if any(mk in obs for mk in _DONE_MARKERS):
            return AgentStep(content="done", tool_calls=[], done=True)
        action = self._rng.choice([0, 1, 2])     # 0=fwd 1=left 2=right
        call = MCPToolCall(name="move", arguments={"action": action})
        return AgentStep(content=f"random:{action}", tool_calls=[call], done=False)
