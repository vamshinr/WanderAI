"""WanderAI visualizer — zero-dependency (stdlib only) web server.

Run:  python serve.py   then open http://localhost:8000

Lets you generate scenes, drive the agent (buttons / arrow keys), or run the
oracle / random policy, while watching the geodesic field, the agent's FOV, and
the live *symbolic observation* the RFT text policy will consume.
"""
from __future__ import annotations
import base64
import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from wanderai.scene_gen import random_scene
from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.observation import DEFAULT_FOV, VISIT_CELL
from wanderai.policies import OraclePolicy, RandomPolicy

HERE = os.path.dirname(os.path.abspath(__file__))

# 3D vision scenes (Phase B): the agent sees the room via MuJoCo RGB+depth.
_SCENES_3D = {
    "test": os.path.join(HERE, "examples", "js76kpb923w3tnvv8thabsdcw58931g0.xml"),
    "train": os.path.join(HERE, "examples", "js755rrf6gmkwj444nzqh6ermx89394v.xml"),
}
_scene3d_cache: dict = {}


def _png_data_url(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _depth_to_rgb(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Colourise a depth buffer with a jet colormap: near = red, far = blue."""
    n = np.clip(depth / max(max_depth, 1e-3), 0, 1)      # 0 near .. 1 far
    v = 1.0 - n                                          # 1 near .. 0 far
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def _fpv_payload(env) -> dict | None:
    """The agent's first-person RGB + colourised depth, as data URLs — only when
    the env is backed by the MuJoCo (vision) renderer."""
    r = getattr(env, "renderer", None)
    if r is None or not hasattr(r, "render_rgb_depth"):
        return None
    rgb, depth = r.render_rgb_depth(env.scene, env.pose)
    return {"rgb": _png_data_url(rgb),
            "depth": _png_data_url(_depth_to_rgb(depth, r.max_depth))}
# Scene-aware trained policies: the 2D model drives the symbolic/2D rooms, the 3D
# model the MuJoCo rooms — both multi-turn RFT models. A LoRA addon on a dedicated
# deployment must be addressed as "<model>#<deployment>" (the bare id 404s). Override
# per scene via env vars or .trained_model*.txt (so addresses can change without code).
_ST1 = ("accounts/vamshinr5899-p0wudhc/models/wander-rft-st1"
        "#accounts/vamshinr5899-p0wudhc/deployments/t67nt9dm")        # 2D single-turn (proven, fast)
_M2DMTQ = ("accounts/vamshinr5899-p0wudhc/models/wander-rft-2dmt-q"
           "#accounts/vamshinr5899-p0wudhc/deployments/y9y8cwq3")     # 2D multi-turn (slow reasoning)
_M3DMTQ = ("accounts/vamshinr5899-p0wudhc/models/wander-rft-3dmt-q"
           "#accounts/vamshinr5899-p0wudhc/deployments/bdl8xznb")     # 3D multi-turn


def _model_from(env_var: str, filename: str, default: str) -> str:
    v = os.environ.get(env_var)
    if v:
        return v
    path = os.path.join(HERE, filename)
    if os.path.exists(path):
        txt = open(path).read().strip()
        if txt:
            return txt
    return default


# Scene-aware defaults (what "auto" uses): st1 for 2D rooms, 3dmt-q for 3D rooms.
TRAINED_MODEL_2D = _model_from("WANDER_TRAINED_MODEL", ".trained_model.txt", _ST1)
TRAINED_MODEL_3D = _model_from("WANDER_TRAINED_MODEL_3D", ".trained_model_3d.txt", _M3DMTQ)
# Explicit dropdown picks (key -> address); "auto" resolves scene-aware above.
_MODEL_ADDR = {"st1": _ST1, "2dmt-q": _M2DMTQ, "3dmt-q": _M3DMTQ}


def _field_payload(env: SceneSearchEnv):
    """Geodesic distance field as a JSON-safe grid (inf -> None) for the heatmap."""
    f = env.field
    data = [[None if not math.isfinite(v) else round(float(v), 3) for v in row]
            for row in f.dist]
    return {"cell_size": f.grid.cell_size, "origin": list(f.grid.origin),
            "nrows": f.grid.nrows, "ncols": f.grid.ncols, "data": data}


def _visited_payload(env: SceneSearchEnv):
    """The agent's episodic memory: the coarse cells it has actually stepped into
    this run (env.visited). Lets the UI shade explored ground and show the agent
    favouring NEW cells — the same memory the NEW/explored observation flags use."""
    cells = sorted(env.visited) if env.visited else []
    return {"cell": VISIT_CELL, "cells": [[cx, cy] for cx, cy in cells]}


def _scene_payload(env: SceneSearchEnv, info: dict):
    s = env.scene
    fpv = _fpv_payload(env)
    return {
        "bounds": s.bounds.__dict__,
        "obstacles": [o.__dict__ for o in s.obstacles],
        "ball": list(s.ball),
        "agent": {"x": env.pose.x, "y": env.pose.y, "heading": env.pose.heading},
        "agent_radius": s.agent_radius,
        "fov": DEFAULT_FOV,
        "field": _field_payload(env),
        "info": _info_payload(info),
        "visited": _visited_payload(env),
        "fpv": fpv,
        "mode": "3d" if fpv else "2d",
    }


def _info_payload(info: dict):
    return {
        "obs_text": info.get("obs_text", ""),
        "geodesic": None if not math.isfinite(info.get("geodesic", math.inf))
        else round(float(info["geodesic"]), 3),
        "optimal": None if not math.isfinite(info.get("optimal", math.inf))
        else round(float(info["optimal"]), 3),
        "path_length": round(float(info.get("path_length", 0.0)), 3),
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "steps": int(info.get("steps", 0)),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        payload = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode() or "{}")

    # --- routes ---
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "ui", "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        st = self.server.state
        try:
            if self.path == "/api/reset":
                body = self._read_json()
                seed = body.get("seed")
                if body.get("scene") == "default" or seed is None:
                    scene = default_scene()
                else:
                    scene = random_scene(np.random.default_rng(int(seed)))
                env = SceneSearchEnv(scene, config=EnvConfig(max_steps=3000))
                _, info = env.reset()
                st["env"] = env
                st["cumulative"] = 0.0
                st["random_policy"] = RandomPolicy(seed=int(seed or 0))
                self._json({**_scene_payload(env, info), "cumulative": 0.0,
                            "done": False})

            elif self.path == "/api/import_mjcf":
                from wanderai.antim_import import mjcf_to_scene, mjcf_zip_to_scene
                body = self._read_json()
                if body.get("xml"):
                    scene = mjcf_to_scene(body["xml"])
                elif body.get("path"):
                    p = body["path"]
                    scene = (mjcf_zip_to_scene(p) if p.endswith(".zip")
                             else mjcf_to_scene(open(p).read()))
                else:
                    return self._json({"error": "provide 'xml' or 'path'"}, 400)
                env = SceneSearchEnv(scene, config=EnvConfig(max_steps=3000))
                _, info = env.reset()
                st["env"] = env
                st["cumulative"] = 0.0
                st["random_policy"] = RandomPolicy()
                self._json({**_scene_payload(env, info), "cumulative": 0.0,
                            "done": False, "source": "antim"})

            elif self.path == "/api/load_3d":
                try:
                    from wanderai.mujoco_renderer import load_mjcf_3d
                except ModuleNotFoundError as e:
                    import sys
                    return self._json({"error": (
                        f"3D vision needs MuJoCo, missing in this interpreter ({e.name}). "
                        f"Install it for the python running serve.py:  "
                        f"{sys.executable} -m pip install mujoco")}, 500)
                body = self._read_json()
                which = body.get("scene", "test")
                # The policy only ever reads the symbolic text observation, so a 3D room
                # is "just a 2D scene" to it. Default to perception="geometry": the SAME
                # clean observation as 2D (accurate clearance/bearing), so the 2D-trained
                # model navigates it well. The RGB+depth view still renders for display.
                # "vision" (decode the obs from rendered pixels) is available but lossy —
                # its noisy bearings made the model just walk forward.
                percept = body.get("perception", "geometry")
                if which not in _SCENES_3D:
                    return self._json({"error": f"unknown 3D scene '{which}'"}, 400)
                if which not in _scene3d_cache:
                    _scene3d_cache[which] = load_mjcf_3d(_SCENES_3D[which])
                scene3d, renderer = _scene3d_cache[which]
                env = SceneSearchEnv(scene3d, renderer=renderer,
                                     config=EnvConfig(max_steps=3000, perception=percept))
                _, info = env.reset()
                st["env"] = env
                st["cumulative"] = 0.0
                st["random_policy"] = RandomPolicy()
                self._json({**_scene_payload(env, info), "cumulative": 0.0,
                            "done": False, "source": f"3d:{which}"})

            elif self.path == "/api/step":
                env = st.get("env")
                if env is None:
                    return self._json({"error": "call /api/reset first"}, 400)
                action = int(self._read_json().get("action", 0))
                _, reward, done, info = env.step(action)
                st["cumulative"] += reward
                self._json({
                    "agent": {"x": env.pose.x, "y": env.pose.y, "heading": env.pose.heading},
                    "reward": round(float(reward), 4),
                    "cumulative": round(float(st["cumulative"]), 4),
                    "done": bool(done),
                    "info": _info_payload(info),
                    "visited": _visited_payload(env),
                    "fpv": _fpv_payload(env),
                })

            elif self.path == "/api/policy_action":
                env = st.get("env")
                if env is None:
                    return self._json({"error": "call /api/reset first"}, 400)
                req = self._read_json()
                name = req.get("policy", "oracle")
                if name == "oracle":
                    policy = OraclePolicy()
                elif name == "llm":
                    if st.get("llm_policy") is None:
                        from wanderai.llm_policy import LLMPolicy
                        st["llm_policy"] = LLMPolicy()
                    policy = st["llm_policy"]
                elif name == "trained":
                    # Model choice from the UI dropdown. An explicit pick wins; "auto"
                    # (or none) is scene-aware: 3D (MuJoCo renderer) -> 3D model, else 2D.
                    key = req.get("model") or "auto"
                    if key in _MODEL_ADDR:
                        model = _MODEL_ADDR[key]
                    else:
                        is_3d = hasattr(getattr(env, "renderer", None), "render_rgb_depth")
                        model = TRAINED_MODEL_3D if is_3d else TRAINED_MODEL_2D
                    pol = st.get("trained_policy")
                    if pol is None or getattr(pol, "_addr", None) != model:
                        from wanderai.llm_policy import GuidedLLMPolicy
                        # Drives from the egocentric view only; the sole assist is
                        # clearance-based obstacle avoidance (no field/oracle/hidden ball).
                        pol = GuidedLLMPolicy(model=model)
                        pol._addr = model
                        st["trained_policy"] = pol
                    policy = pol
                else:
                    policy = st["random_policy"]
                action = int(policy.act(None, env))
                err = getattr(policy, "last_error", None)
                if err:                       # surface model failures, don't fake a step
                    return self._json({"error": f"policy '{name}' call failed: {err}"}, 502)
                self._json({"action": action, "action_name": Action(action).name})

            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:  # surface errors to the UI rather than 500-silent
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)


def _load_dotenv(path=os.path.join(HERE, ".env")):
    """Load .env so 'Run LLM' finds FIREWORKS_API_KEY without manual export."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _reexec_with_mujoco_if_needed():
    """3D vision scenes need MuJoCo, which lives in .venv-hud (python3.12), not the
    system python. If MuJoCo isn't importable here but that venv has it, re-launch
    serve.py under it so `python3 serve.py` works for BOTH 2D and 3D. The guard
    (executable != venv python) prevents an infinite re-exec loop."""
    import importlib.util
    if importlib.util.find_spec("mujoco") is not None:
        return
    venv_py = os.path.join(HERE, ".venv-hud", "bin", "python")
    if os.path.exists(venv_py) and os.path.realpath(sys.executable) != os.path.realpath(venv_py):
        print("MuJoCo not in this interpreter → relaunching under .venv-hud for 3D support…", flush=True)
        os.execv(venv_py, [venv_py, os.path.abspath(__file__), *sys.argv[1:]])


def main():
    _reexec_with_mujoco_if_needed()
    _load_dotenv()
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")    # set HOST=0.0.0.0 in a container (HF Space)
    # Single-threaded: the MuJoCo GL context is thread-affine (rendering from a
    # second thread segfaults on macOS), and a local single-user demo serialises
    # requests anyway. The 2D path is unaffected.
    server = HTTPServer((host, port), Handler)
    server.state = {"env": None, "cumulative": 0.0, "random_policy": RandomPolicy(),
                    "llm_policy": None}
    print(f"WanderAI visualizer → http://{host}:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
