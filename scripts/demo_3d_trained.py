"""Headless proof that the trained RFT model converges on a 3D scene, driven
through the SAME localhost server + endpoints the browser UI uses.

Loads the 3D scene, then repeatedly asks the server for the trained policy's
action and steps it, logging the geodesic distance shrinking to the ball. Saves
the agent's first-person RGB + depth at the start, midpoint, and finish as
evidence frames.

  PORT=8011 python scripts/demo_3d_trained.py [test|train]
"""
import base64
import json
import os
import sys
import time
import urllib.request

PORT = os.environ.get("PORT", "8011")
SCENE = sys.argv[1] if len(sys.argv) > 1 else "test"
BASE = f"http://localhost:{PORT}"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "demo_frames")


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=180).read())


def save_fpv(fpv, tag):
    if not fpv:
        return
    os.makedirs(OUT, exist_ok=True)
    for kind in ("rgb", "depth"):
        data = fpv[kind].split(",", 1)[1]
        with open(os.path.join(OUT, f"{tag}_{kind}.png"), "wb") as fh:
            fh.write(base64.b64decode(data))


def main():
    print(f">>> loading 3D '{SCENE}' scene via {BASE}")
    d = post("/api/load_3d", {"scene": SCENE})
    opt = d["info"]["optimal"]
    print(f"    obstacles={len(d['obstacles'])} ball={[round(c,1) for c in d['ball']]} "
          f"optimal_geodesic={opt:.1f}  mode={d['mode']}")
    save_fpv(d.get("fpv"), "start")

    t0 = time.time()
    done = False
    steps = 0
    first_seen = None
    saved_mid = False
    while not done and steps < 120:
        a = post("/api/policy_action", {"policy": "trained"})
        if "error" in a:
            print("    policy error:", a["error"])
            break
        d = post("/api/step", {"action": a["action"]})
        steps += 1
        info = d["info"]
        if "VISIBLE" in info["obs_text"] and first_seen is None:
            first_seen = steps
        if steps % 5 == 0 or d["done"]:
            print(f"    step {steps:3d}  act={a['action_name']:<12} "
                  f"geodesic={info['geodesic']}  success={info['success']}")
        if not saved_mid and info.get("geodesic") and info["geodesic"] <= opt / 2:
            save_fpv(d.get("fpv"), "mid")
            saved_mid = True
        done = d["done"]

    save_fpv(d.get("fpv"), "finish")
    print(f"\n=== RESULT ({SCENE}) ===")
    print(f"  success={info['success']}  steps={steps}  path={info['path_length']:.1f} "
          f"optimal={opt:.1f}  efficiency={min(1.0, opt/max(info['path_length'],1e-6)):.2f}")
    print(f"  first saw ball at step {first_seen};  wall-clock {time.time()-t0:.0f}s")
    print(f"  evidence frames in {os.path.normpath(OUT)}/")


if __name__ == "__main__":
    main()
