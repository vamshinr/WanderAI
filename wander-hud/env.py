"""WanderAI scene-search as a HUD v6 environment.

Exposes our SceneSearchEnv to a HUD agent: the agent calls reset_room / move
tools (served over an in-process MCP capability) to navigate a generated room and
find the red ball. Reward (0-1) = success, else the fraction of geodesic distance
it closed. Run:  hud eval tasks.py claude --gateway
"""
import asyncio
import contextlib
import math
import socket

from hud import Environment
from hud.capabilities import Capability

from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.scene_gen import make_split

env = Environment(name="wander-scene-search")

# One container per evaluation, so a module-global episode is safe.
_state = {"env": None, "optimal": 0.0, "final_geo": math.inf, "success": False, "steps": 0}
_ACTIONS = {0: Action.MOVE_FORWARD, 1: Action.TURN_LEFT, 2: Action.TURN_RIGHT}


# --- agent-facing tools ---
async def reset_room(seed: int = 0) -> str:
    """Start a new search episode in a generated room (seed selects the room).
    Returns the first egocentric observation."""
    _, scenes = make_split(0, 1, seed=int(seed))
    e = SceneSearchEnv(scenes[0], config=EnvConfig(max_steps=60))
    _, info = e.reset()
    _state.update(env=e, optimal=info["optimal"], final_geo=info["geodesic"],
                  success=False, steps=0)
    return f"New room. {info['obs_text']}"


async def move(action: int) -> str:
    """Take one action: 0 = MOVE_FORWARD, 1 = TURN_LEFT, 2 = TURN_RIGHT.
    Returns the new observation plus whether the red ball was found."""
    e = _state["env"]
    if e is None:
        return "Call reset_room first."
    _, reward, done, info = e.step(_ACTIONS.get(int(action), Action.MOVE_FORWARD))
    _state.update(final_geo=info["geodesic"], success=info["success"], steps=info["steps"])
    tag = "FOUND THE RED BALL!" if info["success"] else ("episode over" if done else "searching")
    return f"{info['obs_text']} [{tag}]"


# --- serve the tools over an in-process MCP capability ---
_MCP_PORT, _SRV = 0, None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _wait(host, port, timeout=10.0):
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        try:
            socket.create_connection((host, port), timeout=0.2).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"MCP server never came up on {host}:{port}")


@env.initialize
async def _up():
    from fastmcp import FastMCP
    global _MCP_PORT, _SRV
    if _SRV is None:
        server = FastMCP(name="scene-search")
        server.tool(reset_room)
        server.tool(move)
        _MCP_PORT = _free_port()
        _SRV = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False))
        await _wait("127.0.0.1", _MCP_PORT)
    env.add_capability(Capability.mcp(name="scene-search", url=f"http://127.0.0.1:{_MCP_PORT}/mcp"))


@env.shutdown
async def _down():
    global _SRV
    if _SRV is not None:
        _SRV.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _SRV
        _SRV = None


# --- the task ---
@env.template(id="find-red-ball")
async def find_red_ball(seed: int = 0):
    """Navigate a generated room to reach the red ball. Reward = success, else the
    fraction of the geodesic distance to the ball that was closed."""
    obs = await reset_room(seed)
    _ = yield (
        "You control an agent in a room and must reach a RED BALL.\n"
        f"{obs}\n"
        "Repeatedly call move(action) — 0=forward, 1=turn left, 2=turn right — using "
        "the observation each step (it tells you if the ball is visible, its bearing, "
        "and the clearance left/center/right). Keep going until you FIND THE RED BALL."
    )
    opt, fg = _state["optimal"], _state["final_geo"]
    if _state["success"]:
        reward = 1.0
    elif opt > 0 and math.isfinite(fg):
        reward = max(0.0, min(1.0, (opt - fg) / opt))
    else:
        reward = 0.0
    yield reward
