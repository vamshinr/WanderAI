"""LLM navigation policy backed by a Fireworks chat model.

Reads the env's symbolic text observation, asks a Fireworks-hosted model for one
action, and parses it. This is the *student* that A5 will reinforcement-fine-tune
(RFT): the model proposes actions, the env scores them with the geodesic reward,
and weights move toward higher return. For A4 we just run the (untrained) model
to establish a baseline and prove the loop end-to-end.

Default model: gpt-oss-20b (serverless on Fireworks AND RL-tunable). It's a
reasoning model, so we use reasoning_effort=low and a parse-friendly format."""

from __future__ import annotations
import json
import os
import re
import ssl
import urllib.request
import urllib.error
from .environment import Action

FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = os.environ.get("FIREWORKS_MODEL", "accounts/fireworks/models/gpt-oss-20b")

SYSTEM_PROMPT = (
    "You control an agent in a room, searching for a RED BALL. You only see a "
    "first-person description: whether the ball is visible (and its bearing/distance), "
    "how much open space is to your left/center/right, which of those directions you "
    "have already explored this run, and your recent moves. "
    "Goal: reach the red ball quickly. If it is visible, head toward it. If not, "
    "explore: move into open space and prefer directions marked NEW over ones marked "
    "explored, so you cover new ground instead of circling. "
    "Think in at most one short sentence, then end with a line exactly:\n"
    "ACTION=<MOVE_FORWARD|TURN_LEFT|TURN_RIGHT>"
)

_ACTIONS = {"MOVE_FORWARD": Action.MOVE_FORWARD,
            "TURN_LEFT": Action.TURN_LEFT,
            "TURN_RIGHT": Action.TURN_RIGHT}
_ACTION_RE = re.compile(r"ACTION\s*=\s*(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)")


def build_messages(obs_text: str):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text + "\nWhat is your next action?"}]


def parse_action(text: str) -> Action:
    """Extract the chosen action: prefer the explicit ACTION= line, else the last
    action token mentioned, else default to MOVE_FORWARD."""
    up = (text or "").upper()
    m = _ACTION_RE.search(up)
    if m:
        return _ACTIONS[m.group(1)]
    hits = [(up.rfind(k), a) for k, a in _ACTIONS.items() if k in up]
    if hits:
        return max(hits)[1]            # the token appearing last
    return Action.MOVE_FORWARD


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class LLMPolicy:
    def __init__(self, model: str | None = None, api_key: str | None = None,
                 reasoning_effort: str = "low", max_tokens: int = 256,
                 temperature: float = 0.0, timeout: int = 60):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._ctx = _ssl_context()

    def _complete(self, messages) -> str:
        """One chat completion -> the assistant's text. Separated out so tests can
        stub it without hitting the network."""
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        body = {"model": self.model, "messages": messages,
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        req = urllib.request.Request(
            FIREWORKS_BASE + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
            d = json.loads(r.read())
        return d["choices"][0]["message"].get("content") or ""

    def act(self, obs, env) -> Action:
        try:
            return parse_action(self._complete(build_messages(env.text_observation())))
        except Exception:
            return Action.MOVE_FORWARD     # never crash the rollout on a bad call
