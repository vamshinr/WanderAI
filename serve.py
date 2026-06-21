"""WanderAI visualizer — zero-dependency (stdlib only) web server.

Run:  python serve.py   then open http://localhost:8000

Lets you generate scenes, drive the agent (buttons / arrow keys), or run the
oracle / random policy, while watching the geodesic field, the agent's FOV, and
the live *symbolic observation* the RFT text policy will consume.
"""
from __future__ import annotations
import json
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from wanderai.scene_gen import random_scene
from wanderai.scene import default_scene
from wanderai.environment import SceneSearchEnv, EnvConfig, Action
from wanderai.observation import DEFAULT_FOV
from wanderai.policies import OraclePolicy, RandomPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
# The RFT-trained policy. A LoRA addon on a dedicated deployment must be addressed
# as "<model>#<deployment>" (the bare model id 404s). scripts/deploy_trained.py
# writes the current address to .trained_model.txt; env var overrides that.
def _trained_model() -> str:
    env = os.environ.get("WANDER_TRAINED_MODEL")
    if env:
        return env
    path = os.path.join(HERE, ".trained_model.txt")
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return ("accounts/vamshinr5899-p0wudhc/models/wander-rft-v1"
            "#accounts/vamshinr5899-p0wudhc/deployments/ykcmxh3l")


TRAINED_MODEL = _trained_model()


def _field_payload(env: SceneSearchEnv):
    """Geodesic distance field as a JSON-safe grid (inf -> None) for the heatmap."""
    f = env.field
    data = [[None if not math.isfinite(v) else round(float(v), 3) for v in row]
            for row in f.dist]
    return {"cell_size": f.grid.cell_size, "origin": list(f.grid.origin),
            "nrows": f.grid.nrows, "ncols": f.grid.ncols, "data": data}


def _scene_payload(env: SceneSearchEnv, info: dict):
    s = env.scene
    return {
        "bounds": s.bounds.__dict__,
        "obstacles": [o.__dict__ for o in s.obstacles],
        "ball": list(s.ball),
        "agent": {"x": env.pose.x, "y": env.pose.y, "heading": env.pose.heading},
        "agent_radius": s.agent_radius,
        "fov": DEFAULT_FOV,
        "field": _field_payload(env),
        "info": _info_payload(info),
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
                env = SceneSearchEnv(scene, config=EnvConfig(max_steps=400))
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
                env = SceneSearchEnv(scene, config=EnvConfig(max_steps=400))
                _, info = env.reset()
                st["env"] = env
                st["cumulative"] = 0.0
                st["random_policy"] = RandomPolicy()
                self._json({**_scene_payload(env, info), "cumulative": 0.0,
                            "done": False, "source": "antim"})

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
                })

            elif self.path == "/api/policy_action":
                env = st.get("env")
                if env is None:
                    return self._json({"error": "call /api/reset first"}, 400)
                name = self._read_json().get("policy", "oracle")
                if name == "oracle":
                    policy = OraclePolicy()
                elif name == "llm":
                    if st.get("llm_policy") is None:
                        from wanderai.llm_policy import LLMPolicy
                        st["llm_policy"] = LLMPolicy()
                    policy = st["llm_policy"]
                elif name == "trained":
                    if st.get("trained_policy") is None:
                        from wanderai.llm_policy import LLMPolicy
                        st["trained_policy"] = LLMPolicy(model=TRAINED_MODEL,
                                                         reasoning_effort=None)
                    policy = st["trained_policy"]
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


def main():
    _load_dotenv()
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.state = {"env": None, "cumulative": 0.0, "random_policy": RandomPolicy(),
                    "llm_policy": None}
    print(f"WanderAI visualizer → http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
