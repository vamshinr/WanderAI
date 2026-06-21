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
import time
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
                 temperature: float = 0.0, timeout: int = 60, retries: int = 8):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries                # ride out scale-to-zero cold starts
        self._ctx = _ssl_context()
        self.last_error: str | None = None   # set when a call fails (don't hide it)

    def _complete(self, messages) -> str:
        """One chat completion -> the assistant's text. Separated out so tests can
        stub it without hitting the network. Retries transient 503/429/5xx with
        backoff: a scale-to-zero LoRA deployment returns 503 for ~30-60s while its
        replica spins up, so the first call after idle would otherwise fail."""
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
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as r:
                    d = json.loads(r.read())
                return d["choices"][0]["message"].get("content") or ""
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_exc = e
                if attempt == self.retries:
                    raise
            time.sleep(min(12, 3 * (attempt + 1)))    # 3,6,9,12,... up to ~75s total
        raise last_exc        # unreachable, but keeps type-checkers happy

    def act(self, obs, env) -> Action:
        try:
            action = parse_action(self._complete(build_messages(env.text_observation())))
            self.last_error = None
            return action
        except Exception as e:
            # Record the failure so callers can surface it — a silent MOVE_FORWARD
            # fallback once masked a 404 as "the model always goes forward".
            self.last_error = f"{type(e).__name__}: {e}"
            return Action.MOVE_FORWARD


class AssistedLLMPolicy:
    """The trained LLM with a geodesic safety-net, so it never gets stuck and always
    reaches the ball. When the model stalls (no geodesic progress) or ping-pongs
    LEFT/RIGHT, we take ONE privileged geodesic-descent step to break the loop, then
    hand control back to the model. Demo mode: guarantees convergence; the nudge
    uses ground-truth geometry the model itself never sees."""

    def __init__(self, model: str | None = None, stuck_patience: int = 2):
        from .policies import OraclePolicy
        self.llm = LLMPolicy(model=model, reasoning_effort=None, temperature=0.0)
        self.oracle = OraclePolicy()
        self.stuck_patience = stuck_patience
        self._env = None
        self.prev_d = None
        self.stall = 0
        self.recent: list = []
        self.last_error: str | None = None
        self.assists = 0

    def act(self, obs, env) -> Action:
        if env is not self._env:                      # new episode → reset state
            self._env, self.prev_d, self.stall, self.recent, self.assists = env, None, 0, [], 0

        d = env.field.query(env.pose.x, env.pose.y)
        progressed = self.prev_d is None or d < self.prev_d - 1e-3
        self.stall = 0 if progressed else self.stall + 1
        self.prev_d = d
        ping_pong = len(self.recent) >= 3 and all(a is not Action.MOVE_FORWARD for a in self.recent[-3:])

        if self.stall >= self.stuck_patience or ping_pong:
            action = self.oracle.act(obs, env)        # guaranteed-progress nudge
            self.stall, self.last_error = 0, None
            self.assists += 1
        else:
            action = self.llm.act(obs, env)
            self.last_error = self.llm.last_error
        self.recent.append(action)
        return action
