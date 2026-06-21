from __future__ import annotations

import asyncio
import math
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .environment import Action, EnvConfig, SceneSearchEnv
from .geometry import AABB, Pose
from .renderer import StubRenderer
from .scene import Scene, default_scene


DEFAULT_SCENE_ID = "default"
FIND_RED_BALL_PROMPT = (
    "Find the red ball in the scene. Use look to inspect the egocentric view, "
    "turn_left or turn_right to scan, move_forward to navigate, and "
    "episode_status to check whether the episode is complete."
)

SceneFactory = Callable[[str, int | None], Scene]
RendererFactory = Callable[[], StubRenderer]


try:  # pragma: no cover - exercised only when FastMCP is installed.
    from fastmcp import FastMCP as FastMCP
except Exception:  # pragma: no cover - local shim is tested instead.

    class FastMCP:
        """Tiny FastMCP-compatible shim for local tests without HUD deps."""

        def __init__(self, name: str):
            self.name = name
            self.tools: dict[str, Callable[..., Any]] = {}

        def tool(self, fn: Callable[..., Any] | None = None, **_kwargs):
            def _register(func: Callable[..., Any]):
                self.tools[func.__name__] = func
                return func

            if fn is None:
                return _register
            return _register(fn)

        async def run_async(self, *_args, **_kwargs):
            raise RuntimeError("fastmcp is required to serve the MCP tools")


@dataclass(frozen=True)
class LocalCapability:
    name: str
    protocol: str
    url: str
    params: dict[str, Any]

    @classmethod
    def mcp(cls, *, name: str, url: str, auth_token: str | None = None):
        params: dict[str, Any] = {}
        if auth_token is not None:
            params["auth_token"] = auth_token
        return cls(name=name, protocol="mcp/2025-11-25", url=url, params=params)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "url": self.url,
            "params": self.params,
        }


class LocalEnvironment:
    """Minimal HUD Environment shim so env.py imports in local CI."""

    def __init__(
        self,
        name: str = "environment",
        version: str = "0.0.1",
        capabilities: list[Any] | None = None,
    ):
        self.name = name
        self.version = version
        self.capabilities = list(capabilities or [])
        self.initializers: list[Callable[[], Any]] = []
        self.shutdowns: list[Callable[[], Any]] = []
        self.templates: dict[str, Callable[..., Any]] = {}

    def add_capability(self, capability: Any) -> None:
        cap_name = getattr(capability, "name", None)
        self.capabilities = [
            existing
            for existing in self.capabilities
            if getattr(existing, "name", None) != cap_name
        ]
        self.capabilities.append(capability)

    def initialize(self, fn: Callable[[], Any]):
        self.initializers.append(fn)
        return fn

    def shutdown(self, fn: Callable[[], Any]):
        self.shutdowns.append(fn)
        return fn

    def template(self, id: str | None = None, **_kwargs):
        def _register(fn: Callable[..., Any]):
            template_id = id or fn.__name__
            self.templates[template_id] = fn
            setattr(fn, "template_id", template_id)
            return fn

        return _register

    async def start(self):
        for fn in self.initializers:
            result = fn()
            if hasattr(result, "__await__"):
                await result

    async def stop(self):
        for fn in reversed(self.shutdowns):
            result = fn()
            if hasattr(result, "__await__"):
                await result


@dataclass(frozen=True)
class ObservationSummary:
    text: str
    target_visible: bool
    target_bearing: str
    image: dict[str, int]


def default_scene_factory(scene_id: str, _seed: int | None = None) -> Scene:
    if scene_id == DEFAULT_SCENE_ID:
        return default_scene()
    if scene_id == "open-room":
        return Scene(
            bounds=AABB(0, 0, 5, 5),
            obstacles=[],
            ball=(4.4, 2.5),
            agent_start=Pose(0.6, 2.5, 0.0),
            agent_radius=0.2,
        )
    if scene_id == "turn-room":
        return Scene(
            bounds=AABB(0, 0, 5, 5),
            obstacles=[AABB(1.6, 0.0, 1.9, 3.4), AABB(3.1, 1.6, 3.4, 5.0)],
            ball=(4.4, 0.7),
            agent_start=Pose(0.6, 4.2, -math.pi / 2),
            agent_radius=0.2,
        )
    if scene_id == "occlusion-room":
        return Scene(
            bounds=AABB(0, 0, 6, 6),
            obstacles=[AABB(2.2, 1.8, 3.2, 4.3), AABB(4.2, 0.8, 4.8, 2.0)],
            ball=(5.3, 1.2),
            agent_start=Pose(0.7, 3.0, 0.0),
            agent_radius=0.2,
        )
    raise ValueError(
        f"Unknown scene_id {scene_id!r}; expected one of default, open-room, turn-room, occlusion-room"
    )


def summarize_observation(obs: np.ndarray) -> ObservationSummary:
    arr = np.asarray(obs)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return ObservationSummary(
            text="Observation image is unavailable; only episode state is known.",
            target_visible=False,
            target_bearing="unknown",
            image={},
        )

    height, width, channels = arr.shape[:3]
    red = arr[:, :, 0].astype(np.int16)
    green = arr[:, :, 1].astype(np.int16)
    blue = arr[:, :, 2].astype(np.int16)
    red_mask = (red >= 150) & (red >= green * 2) & (red >= blue * 2) & (green <= 120)
    visible = int(red_mask.sum()) >= max(4, int(height * width * 0.002))

    image = {"height": int(height), "width": int(width), "channels": int(channels)}
    if not visible:
        return ObservationSummary(
            text="No red target is visible in the current view.",
            target_visible=False,
            target_bearing="none",
            image=image,
        )

    xs = np.nonzero(red_mask)[1]
    center_x = float(xs.mean()) if xs.size else width / 2
    if center_x < width / 3:
        bearing = "left"
    elif center_x > (2 * width) / 3:
        bearing = "right"
    else:
        bearing = "center"

    return ObservationSummary(
        text=f"Red target is visible near the {bearing} of the view.",
        target_visible=True,
        target_bearing=bearing,
        image=image,
    )


class SceneSearchHudAdapter:
    def __init__(
        self,
        *,
        scene_factory: SceneFactory = default_scene_factory,
        renderer_factory: RendererFactory | None = None,
        config: EnvConfig | None = None,
    ):
        self._scene_factory = scene_factory
        self._renderer_factory = renderer_factory or StubRenderer
        self._config = config or EnvConfig()
        self._env: SceneSearchEnv | None = None
        self._scene_id: str | None = None
        self._seed: int | None = None
        self._last_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}
        self._last_action: str | None = None
        self._last_collision = False
        self._terminated = False

    def reset_episode(
        self, scene_id: str = DEFAULT_SCENE_ID, seed: int | None = None
    ) -> dict[str, Any]:
        scene = self._scene_factory(scene_id, seed)
        self._env = SceneSearchEnv(
            scene=scene,
            renderer=self._renderer_factory(),
            config=self._config,
        )
        self._scene_id = scene_id
        self._seed = seed
        self._last_action = None
        self._last_collision = False
        self._terminated = False
        self._last_obs, self._last_info = self._env.reset()
        return self._observation_payload(action="reset")

    def look(
        self, scene_id: str | None = None, seed: int | None = None
    ) -> dict[str, Any]:
        self._ensure_episode(scene_id=scene_id, seed=seed)
        return self._observation_payload(action="look")

    def move_forward(self) -> dict[str, Any]:
        return self._step(Action.MOVE_FORWARD, "move_forward")

    def turn_left(self) -> dict[str, Any]:
        return self._step(Action.TURN_LEFT, "turn_left")

    def turn_right(self) -> dict[str, Any]:
        return self._step(Action.TURN_RIGHT, "turn_right")

    def episode_status(self) -> dict[str, Any]:
        return {"episode": self._public_episode_state()}

    def final_reward(self) -> float:
        state = self.hidden_episode_state()
        if not state["success"]:
            return 0.0
        optimal = state["optimal"]
        path_length = state["path_length"]
        if not math.isfinite(optimal) or optimal <= 0.0:
            return 0.0
        denom = max(path_length, optimal)
        if not math.isfinite(denom) or denom <= 0.0:
            return 0.0
        return max(0.0, min(1.0, optimal / denom))

    def hidden_episode_state(self) -> dict[str, Any]:
        return {
            "scene_id": self._scene_id,
            "seed": self._seed,
            "success": bool(self._last_info.get("success", False)),
            "optimal": float(self._last_info.get("optimal", math.inf)),
            "path_length": float(self._last_info.get("path_length", 0.0)),
            "steps": int(self._last_info.get("steps", 0)),
            "terminated": self._terminated,
        }

    def _ensure_episode(
        self, scene_id: str | None = None, seed: int | None = None
    ) -> None:
        effective_scene_id = scene_id if scene_id is not None else self._scene_id
        effective_scene_id = effective_scene_id or DEFAULT_SCENE_ID
        effective_seed = seed if seed is not None else self._seed

        needs_reset = self._env is None
        if scene_id is not None and scene_id != self._scene_id:
            needs_reset = True
        if seed is not None and seed != self._seed:
            needs_reset = True

        if needs_reset:
            self.reset_episode(scene_id=effective_scene_id, seed=effective_seed)

    def _step(self, action: Action, action_name: str) -> dict[str, Any]:
        self._ensure_episode()
        assert self._env is not None

        if self._terminated:
            payload = self._observation_payload(action=action_name)
            payload["message"] = "Episode is already complete; reset before acting again."
            return payload

        obs, _reward, done, info = self._env.step(action)
        self._last_obs = obs
        self._last_info = info
        self._terminated = bool(done)
        self._last_action = action_name
        self._last_collision = bool(info.get("collision", False))
        payload = self._observation_payload(action=action_name)
        payload["collision"] = self._last_collision
        return payload

    def _observation_payload(self, action: str) -> dict[str, Any]:
        summary = summarize_observation(self._last_obs)
        return {
            "action": action,
            "observation": summary.text,
            "target_visible": summary.target_visible,
            "target_bearing": summary.target_bearing,
            "image": summary.image,
            "episode": self._public_episode_state(),
        }

    def _public_episode_state(self) -> dict[str, Any]:
        if self._env is None:
            return {
                "started": False,
                "scene_id": None,
                "seed": None,
                "steps": 0,
                "max_steps": self._config.max_steps,
                "terminated": False,
                "success": False,
                "last_action": None,
                "last_collision": False,
            }

        return {
            "started": True,
            "scene_id": self._scene_id,
            "seed": self._seed,
            "steps": int(self._last_info.get("steps", 0)),
            "max_steps": self._env.config.max_steps,
            "terminated": self._terminated,
            "success": bool(self._last_info.get("success", False)),
            "last_action": self._last_action,
            "last_collision": self._last_collision,
        }


def create_mcp_server(
    adapter: SceneSearchHudAdapter | None = None,
    *,
    name: str = "wanderai-tools",
) -> FastMCP:
    adapter = adapter or SceneSearchHudAdapter()
    server = FastMCP(name=name)

    def register(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool = getattr(server, "tool")
        try:
            registered = tool(fn)
        except TypeError:
            registered = tool()(fn)
        return registered or fn

    @register
    def look() -> dict[str, Any]:
        """Inspect the current egocentric view without advancing the episode."""
        return adapter.look()

    @register
    def move_forward() -> dict[str, Any]:
        """Move one step forward if the cell ahead is traversable."""
        return adapter.move_forward()

    @register
    def turn_left() -> dict[str, Any]:
        """Rotate left in place."""
        return adapter.turn_left()

    @register
    def turn_right() -> dict[str, Any]:
        """Rotate right in place."""
        return adapter.turn_right()

    @register
    def episode_status() -> dict[str, Any]:
        """Return non-privileged episode progress and completion state."""
        return adapter.episode_status()

    if not hasattr(server, "tools"):
        setattr(
            server,
            "tools",
            {
                "look": look,
                "move_forward": move_forward,
                "turn_left": turn_left,
                "turn_right": turn_right,
                "episode_status": episode_status,
            },
        )

    return server


async def wait_for_port(host: str, port: int, *, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.1)
            if sock.connect_ex((host, port)) == 0:
                return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")
