"""Vision-language-model navigation policy — the agent acts from PIXELS.

Where `llm_policy` feeds the model a symbolic text observation, this policy feeds
it the actual first-person frame: the rendered RGB image goes into the prompt as a
data-URL PNG (encoded here with stdlib zlib+struct — no PIL), alongside a small
text scaffold carrying the agent's episodic memory (recent moves + which
directions it has already explored). That memory must be in-context — a core
WanderAI thesis is that where the agent has been *this episode* cannot live in the
model's weights.

Honesty contract — the policy senses the world ONLY through:
  * the rendered first-person RGB frame (the image the VLM looks at);
  * its own pose-derived episodic memory (recent actions, visited-cell flags) —
    GPS+Compass state that is part of the standard ObjectNav task spec.
It never reads the environment's occupancy grid, geodesic field, the ball's
hidden position, or the scene's obstacle list. The scaffold deliberately omits
the symbolic ball-visibility line: if the model wants the ball, it must SEE it.

`VisionFrontierPolicy` names the complementary classical route: the existing
FrontierPolicy run in vision mode, where all sensing already flows through
`perceive()` (red-blob detection) and the rendered depth buffer."""
from __future__ import annotations

import base64
import json
import os
import struct
import time
import urllib.error
import urllib.request
import zlib

import numpy as np

from .environment import Action
from .frontier_policy import FrontierConfig, FrontierPolicy
from .llm_policy import parse_action, _ssl_context
from .observation import observe

DEFAULT_VLM_MODEL = os.environ.get(
    "WANDER_VLM_MODEL", "accounts/fireworks/models/qwen3-vl-8b-instruct")
DEFAULT_VLM_BASE = os.environ.get(
    "WANDER_VLM_BASE_URL", "https://api.fireworks.ai/inference/v1")

SYSTEM_PROMPT = (
    "You control an agent in a room, searching for a RED BALL. The image is your "
    "first-person view right now. If the red ball appears anywhere in the image, "
    "turn until it is centred and then move toward it. If it is not in the image, "
    "explore: move into open space and prefer directions marked NEW over ones "
    "marked explored, so you cover new ground instead of circling. The text below "
    "the image lists your recent moves and which directions you have already "
    "explored this run. "
    "Think in at most one short sentence, then end with a line exactly:\n"
    "ACTION=<MOVE_FORWARD|TURN_LEFT|TURN_RIGHT>"
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def encode_png(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 image as a minimal valid PNG (stdlib only, no PIL).

    8-bit truecolor, one zlib-compressed IDAT, filter type 0 (None) on every
    scanline — the simplest bitstream every decoder accepts. Compression is worse
    than PIL's adaptive filtering, but our frames are tiny (64x64) and the point
    is zero dependencies."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("encode_png expects an HxWx3 uint8 array")
    rgb = np.ascontiguousarray(rgb)
    h, w = rgb.shape[:2]
    # bit depth 8, color type 2 (RGB), compression 0, filter 0, interlace 0
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    return (PNG_SIGNATURE + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(raw))
            + _png_chunk(b"IEND", b""))


def build_vision_messages(rgb: np.ndarray, obs_text_scaffold: str) -> list:
    """OpenAI-format chat messages: system prompt + [image, text] user content.
    The image is the agent's first-person frame as a data-URL PNG; the text is the
    episodic-memory scaffold (recent moves, explored flags) plus the question."""
    b64 = base64.b64encode(encode_png(rgb)).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text",
             "text": obs_text_scaffold + "\nWhat is your next action?"},
        ]},
    ]


class VLMPolicy:
    """`act(obs, env) -> Action` driven by a vision-language model.

    A `transport` callable (messages -> assistant text) can be injected for
    testing; the default transport POSTs to a Fireworks-compatible
    /chat/completions endpoint via urllib, retrying transient failures the same
    way `LLMPolicy` does. On any transport failure the policy falls back to a
    deterministic TURN_LEFT scan and records `.last_error` (never hide failures —
    a silent fallback once masked a 404 as model behaviour)."""

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0,
                 max_tokens: int = 256, timeout: int = 60, retries: int = 8,
                 transport=None):
        self.model = model or DEFAULT_VLM_MODEL
        self.base_url = (base_url or DEFAULT_VLM_BASE).rstrip("/")
        self.api_key = api_key or os.environ.get("FIREWORKS_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries               # ride out scale-to-zero cold starts
        self.transport = transport or self._http_transport
        self._ctx = _ssl_context()
        self.last_error: str | None = None

    def _http_transport(self, messages) -> str:
        """One chat completion -> the assistant's text (default transport).
        Mirrors LLMPolicy._complete: transient 503/429/5xx are retried with
        backoff because a scale-to-zero deployment 503s while spinning up."""
        if not self.api_key:
            raise RuntimeError("FIREWORKS_API_KEY not set")
        body = {"model": self.model, "messages": messages,
                "max_tokens": self.max_tokens, "temperature": self.temperature}
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=json.dumps(body).encode(),
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
            time.sleep(min(12, 3 * (attempt + 1)))
        raise last_exc        # unreachable, but keeps type-checkers happy

    @staticmethod
    def _frame(env) -> np.ndarray:
        """The agent's current first-person RGB frame — the only world channel."""
        if hasattr(env.renderer, "render_rgb_depth"):
            rgb, _depth = env.renderer.render_rgb_depth(env.scene, env.pose)
            return rgb
        return env.renderer.render(env.scene, env.pose)

    @staticmethod
    def _scaffold(env) -> str:
        """Episodic-memory text: recent moves + explored-direction flags. Built
        from the agent's own history/visited state — the ball's position and the
        symbolic clearance are deliberately NOT included (the model must read the
        world from the image, not from privileged text)."""
        obs = observe(env.scene, env.pose, history=env.history,
                      visited=env.visited)
        tag = lambda b: "explored" if b else "NEW"
        e = obs.explored
        mem = (f"Explored — left: {tag(e['left'])}, center: {tag(e['center'])}, "
               f"right: {tag(e['right'])} ({obs.n_visited} cells seen).")
        moves = ", ".join(obs.recent_actions) if obs.recent_actions else "none"
        return f"{mem} Recent moves: {moves}."

    def messages_for(self, env) -> list:
        """The exact prompt for the env's current state — a pure function of that
        state (no wall clock, no randomness), so identical states prompt
        identically and episodes replay deterministically."""
        return build_vision_messages(self._frame(env), self._scaffold(env))

    def act(self, obs, env) -> Action:
        messages = self.messages_for(env)
        try:
            action = parse_action(self.transport(messages))
            self.last_error = None
            return action
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return Action.TURN_LEFT       # deterministic scan, not a fake "forward"


class VisionFrontierPolicy(FrontierPolicy):
    """Pixels-only CLASSICAL pipeline, named for the benchmark table.

    This is just `FrontierPolicy` — no new behaviour. When the env runs with
    `EnvConfig(perception="vision")` and a renderer exposing `render_rgb_depth`,
    FrontierPolicy already senses exclusively through pixels: the ball via
    `perceive()`'s red-blob detector on RGB, and the map via
    `mapping.obstacle_strip_from_depth` — a height-aware projection of the full
    depth image, so furniture below the horizon still registers. The alias
    exists so "frontier exploration from pixels" has a name distinct from the
    VLM route."""


def make_pixels_only_policy(config: FrontierConfig | None = None) -> FrontierPolicy:
    """Factory for the pixels-only classical baseline (see VisionFrontierPolicy).
    Pair it with an env configured with perception="vision"."""
    return VisionFrontierPolicy(config)
