from __future__ import annotations

import asyncio
import contextlib
import os
import socket

try:  # pragma: no cover - exercised only when HUD is installed.
    from hud.capabilities import Capability
    from hud.environment import Environment
except Exception:  # pragma: no cover - local tests use this shim.
    from wanderai.hud_adapter import LocalCapability as Capability
    from wanderai.hud_adapter import LocalEnvironment as Environment

from wanderai.hud_adapter import (
    DEFAULT_SCENE_ID,
    FIND_RED_BALL_PROMPT,
    SceneSearchHudAdapter,
    create_mcp_server,
    wait_for_port,
)


MCP_HOST = os.getenv("WANDERAI_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("WANDERAI_MCP_PORT", "0"))

adapter = SceneSearchHudAdapter()
mcp_server = create_mcp_server(adapter)
env = Environment(name="wanderai-scene-search")
_mcp_task: asyncio.Task | None = None


@env.initialize
async def _start_mcp_tools() -> None:
    global _mcp_task
    port = MCP_PORT or _choose_free_port(MCP_HOST)
    if _mcp_task is None:
        _mcp_task = asyncio.create_task(
            mcp_server.run_async(
                transport="http",
                host=MCP_HOST,
                port=port,
            )
        )
        await wait_for_port(MCP_HOST, port)

    env.add_capability(
        Capability.mcp(
            name="wanderai-tools",
            url=f"http://{MCP_HOST}:{port}/mcp",
        )
    )


@env.shutdown
async def _stop_mcp_tools() -> None:
    global _mcp_task
    if _mcp_task is not None:
        _mcp_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _mcp_task
        _mcp_task = None


def _choose_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@env.template(
    id="find_red_ball",
    description="Navigate the egocentric scene and find the red ball.",
)
async def find_red_ball(scene_id: str = DEFAULT_SCENE_ID, seed: int | None = None):
    adapter.reset_episode(scene_id=scene_id, seed=seed)
    _answer = yield FIND_RED_BALL_PROMPT
    yield adapter.final_reward()
