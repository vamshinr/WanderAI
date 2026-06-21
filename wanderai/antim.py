"""Antim Labs / Gizmo client.

Gizmo is a scene *generator*: prompt -> 3D scene -> export (MJCF / USD / SDF).
Generation is slow (minutes) and not always reliable, so the intended use here is
**pre-generate + cache + import**, not live-on-stage: batch a few scenes, cache the
exported archives, and load them into our env via `antim_import.mjcf_to_scene`.

Endpoints (verified against the live OpenAPI, 2026-06-20):
  POST /v1/scenes                  {prompt, asset_pipeline, persist} -> {scene_id, job_id, status}
  GET  /v1/scenes/{id}/status      -> {pipeline_status, pipeline_stages, ...}
  GET  /v1/jobs/{id}               -> {job: {status, ...}}
  POST /v1/scenes/{id}/export      {format} -> archive bytes
Auth: Bearer $GIZMO_API_KEY.
"""
from __future__ import annotations
import json
import os
import ssl
import time
import urllib.request
import urllib.error

GIZMO_BASE = "https://api.gizmo.antimlabs.com/v1"
_TERMINAL_OK = {"succeeded", "completed", "done", "ready"}
_TERMINAL_BAD = {"failed", "error", "cancelled"}


class GizmoError(RuntimeError):
    pass


def _ssl_context() -> ssl.SSLContext:
    """Prefer certifi's CA bundle — the python.org framework build ships without
    a usable system bundle and otherwise fails TLS verification."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


class GizmoClient:
    def __init__(self, api_key: str | None = None, base: str = GIZMO_BASE):
        self.api_key = api_key or os.environ.get("GIZMO_API_KEY")
        if not self.api_key:
            raise GizmoError("GIZMO_API_KEY not set (put it in .env or pass api_key=)")
        self.base = base
        self._ctx = _ssl_context()

    def _request(self, method: str, path: str, body=None, raw: bool = False, timeout: int = 90):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=self._ctx) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            raise GizmoError(f"{method} {path} -> HTTP {e.code}: {e.read()[:300]!r}")
        return payload if raw else json.loads(payload)

    # --- API surface ---
    def whoami(self) -> dict:
        return self._request("GET", "/whoami")

    def generate(self, prompt: str, asset_pipeline: str = "auto", persist: bool = True) -> dict:
        return self._request("POST", "/scenes",
                             {"prompt": prompt, "asset_pipeline": asset_pipeline, "persist": persist})

    def scene_status(self, scene_id: str) -> dict:
        return self._request("GET", f"/scenes/{scene_id}/status")

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/jobs/{job_id}?include_result=true")

    def export(self, scene_id: str, fmt: str = "mjcf") -> bytes:
        return self._request("POST", f"/scenes/{scene_id}/export", {"format": fmt}, raw=True)

    def wait(self, scene_id: str, timeout: int = 900, interval: int = 8, on_poll=None) -> str:
        """Block until the scene's pipeline reaches a terminal state. Returns the
        terminal status; raises GizmoError on failure or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            st = self.scene_status(scene_id)
            status = st.get("pipeline_status")
            if on_poll:
                on_poll(status, st.get("pipeline_stages", []))
            if status in _TERMINAL_OK:
                return status
            if status in _TERMINAL_BAD:
                raise GizmoError(f"scene {scene_id} {status}: {st.get('error')!r}")
            time.sleep(interval)
        raise GizmoError(f"scene {scene_id} timed out after {timeout}s")

    def generate_export(self, prompt: str, fmt: str = "mjcf", out_path: str | None = None,
                        asset_pipeline: str = "auto", timeout: int = 900, on_poll=None) -> str:
        """Full flow: generate -> wait -> export -> cache archive to out_path."""
        job = self.generate(prompt, asset_pipeline=asset_pipeline)
        scene_id = job["scene_id"]
        self.wait(scene_id, timeout=timeout, on_poll=on_poll)
        archive = self.export(scene_id, fmt=fmt)
        out_path = out_path or os.path.join("scenes_cache", f"{scene_id}.{fmt}.zip")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "wb") as fh:
            fh.write(archive)
        return out_path
