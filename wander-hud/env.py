"""WanderAI scene-search as a HUD v6 environment.

Exposes our SceneSearchEnv to a HUD agent: the agent calls reset_room / move
tools (served over an in-process MCP capability) to navigate a generated room and
find the red ball. Reward (0-1) = success, else the fraction of geodesic distance
it closed. Run:  hud eval tasks.py claude --gateway
"""
import asyncio
import contextlib
import math
import os
import socket

from hud import Environment
from hud.capabilities import Capability

from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.scene_gen import make_split

env = Environment(name="wander-scene-search")

# --- 3D vision scenes (Phase B): the agent sees the room via MuJoCo RGB+depth ---
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCENES_3D = {
    "test": os.path.join(_REPO, "examples", "js76kpb923w3tnvv8thabsdcw58931g0.xml"),
    "train": os.path.join(_REPO, "examples", "js755rrf6gmkwj444nzqh6ermx89394v.xml"),
}
_scene3d_cache = {}


def _load_3d(which):
    """Load (and cache) a 3D scene + MuJoCo renderer — recompiling per episode is
    wasteful and the renderer holds a GL context."""
    if which not in _scene3d_cache:
        from wanderai.mujoco_renderer import load_mjcf_3d
        _scene3d_cache[which] = load_mjcf_3d(_SCENES_3D[which])
    return _scene3d_cache[which]

# One container per evaluation, so a module-global episode is safe.
_state = {"env": None, "optimal": 0.0, "final_geo": math.inf, "success": False, "steps": 0}
_ACTIONS = {0: Action.MOVE_FORWARD, 1: Action.TURN_LEFT, 2: Action.TURN_RIGHT}
_STR_ACTIONS = {"MOVE_FORWARD": Action.MOVE_FORWARD, "TURN_LEFT": Action.TURN_LEFT,
                "TURN_RIGHT": Action.TURN_RIGHT, "0": Action.MOVE_FORWARD,
                "1": Action.TURN_LEFT, "2": Action.TURN_RIGHT}


def _parse_action(action):
    """Accept either an int (0/1/2) or a string token (MOVE_FORWARD/...). Our
    multi-turn RFT models were trained on the wander_lake move(action: str) tool, so
    HUD eval must accept the string form too — not just the int."""
    if isinstance(action, str):
        s = action.strip().upper()
        for tok, a in _STR_ACTIONS.items():
            if tok in s:
                return a
        return Action.MOVE_FORWARD
    return _ACTIONS.get(int(action), Action.MOVE_FORWARD)


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


async def reset_room_3d(scene: str = "test") -> str:
    """Start a search episode in a 3D MuJoCo room ('test' or 'train'); the agent
    perceives it through rendered RGB+depth. Returns the first observation."""
    s3, renderer = _load_3d(scene)
    e = SceneSearchEnv(s3, renderer=renderer,
                       config=EnvConfig(max_steps=120, perception="vision"))
    _, info = e.reset()
    _state.update(env=e, optimal=info["optimal"], final_geo=info["geodesic"],
                  success=False, steps=0)
    return f"New 3D room ({scene}). {info['obs_text']}"


_scene3d_geo_cache = {}


def _load_3d_geo(scene: str, seed: int):
    """Load a 3D scene's GEOMETRY (no rendering) with a reachable randomized start —
    matches how the 3D multi-turn model was TRAINED (symbolic obs of the real 3D
    scene; the rollout cluster is headless so training used geometry, not vision)."""
    import math
    import numpy as np
    from dataclasses import replace
    from wanderai.antim_import import mjcf_to_scene
    from wanderai.occupancy import OccupancyGrid
    from wanderai.distance_field import DistanceField
    from wanderai.geometry import Pose
    if scene not in _scene3d_geo_cache:
        _scene3d_geo_cache[scene] = mjcf_to_scene(_SCENES_3D[scene])
    base = _scene3d_geo_cache[scene]
    rng = np.random.default_rng(seed)
    grid = OccupancyGrid.from_scene(base, 0.1)
    field = DistanceField.from_grid(grid, base.ball)
    b = base.bounds
    fb = None
    for _ in range(600):
        x = rng.uniform(b.min_x + 0.4, b.max_x - 0.4)
        y = rng.uniform(b.min_y + 0.4, b.max_y - 0.4)
        if not base.is_free(x, y):
            continue
        d = field.query(x, y)
        if not math.isfinite(d):
            continue
        fb = (x, y)
        if 3.0 <= d <= 9.0:
            return replace(base, agent_start=Pose(x, y, rng.uniform(-math.pi, math.pi)))
    return replace(base, agent_start=Pose(fb[0], fb[1], 0.0)) if fb else base


async def reset_room_3d_sym(scene: str = "train", seed: int = 0) -> str:
    """3D scene via SYMBOLIC geometry observation (matches the multi-turn 3D model's
    training distribution). Returns the first observation."""
    s = _load_3d_geo(scene, int(seed))
    e = SceneSearchEnv(s, config=EnvConfig(max_steps=80))
    _, info = e.reset()
    _state.update(env=e, optimal=info["optimal"], final_geo=info["geodesic"],
                  success=False, steps=0)
    return f"New 3D room ({scene}, symbolic). {info['obs_text']}"


async def move(action: str) -> str:
    """Take one step. action must be one of: MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
    (the strings 0, 1, 2 are also accepted). Returns the new observation plus
    whether the red ball was found."""
    e = _state["env"]
    if e is None:
        return "Call reset_room first."
    _, reward, done, info = e.step(_parse_action(action))
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
        server.tool(reset_room_3d)
        server.tool(reset_room_3d_sym)
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


@env.template(id="find-red-ball-3d-sym")
async def find_red_ball_3d_sym(scene: str = "train", seed: int = 0):
    """Navigate a real 3D MuJoCo scene via a SYMBOLIC observation — the track the
    multi-turn 3D RFT model was trained on. Reward = success, else fraction closed."""
    obs = await reset_room_3d_sym(scene, seed)
    _ = yield (
        "You control an agent in a 3D room and must reach a RED BALL.\n"
        f"{obs}\n"
        "Repeatedly call move(action) — MOVE_FORWARD, TURN_LEFT, or TURN_RIGHT — using "
        "the observation each step (ball visibility/bearing, clearance left/center/right). "
        "Keep going until you FIND THE RED BALL."
    )
    opt, fg = _state["optimal"], _state["final_geo"]
    if _state["success"]:
        reward = 1.0
    elif opt > 0 and math.isfinite(fg):
        reward = max(0.0, min(1.0, (opt - fg) / opt))
    else:
        reward = 0.0
    yield reward


@env.template(id="find-red-ball-3d")
async def find_red_ball_3d(scene: str = "test"):
    """Navigate a real 3D MuJoCo room — perceived through rendered RGB+depth — to
    reach the red ball. Reward = success, else the fraction of geodesic closed."""
    obs = await reset_room_3d(scene)
    _ = yield (
        "You control an agent in a 3D room and must reach a RED BALL. You perceive "
        "the room through a first-person camera (RGB + depth).\n"
        f"{obs}\n"
        "Repeatedly call move(action) — 0=forward, 1=turn left, 2=turn right — using "
        "the observation each step (it tells you if the ball is visible, its bearing, "
        "and the clearance left/center/right from depth). Keep going until you FIND "
        "THE RED BALL."
    )
    opt, fg = _state["optimal"], _state["final_geo"]
    if _state["success"]:
        reward = 1.0
    elif opt > 0 and math.isfinite(fg):
        reward = max(0.0, min(1.0, (opt - fg) / opt))
    else:
        reward = 0.0
    yield reward
