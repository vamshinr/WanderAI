from __future__ import annotations

import re
import tomllib
from pathlib import Path

import tasks


ROOT = Path(__file__).resolve().parents[1]


def test_tasks_exports_scene_search_rows():
    assert tasks.TASKSET_NAME == "wanderai-scene-search"
    assert len(tasks.tasks) >= 4

    slugs = [task.slug for task in tasks.tasks]
    assert len(slugs) == len(set(slugs))
    scene_ids = {task.args["scene_id"] for task in tasks.tasks}
    assert {"default", "open-room", "turn-room", "occlusion-room"} <= scene_ids

    for task in tasks.tasks:
        assert task.env == "wanderai-scene-search"
        assert task.id == "find_red_ball"
        assert isinstance(task.args["seed"], int)
        assert task.columns["task"] == "find_red_ball"
        assert task.columns["scene_id"] == task.args["scene_id"]
        assert task.columns["seed"] == task.args["seed"]
        assert task.agent_config["max_steps"] >= 80


def test_hud_dockerfile_serves_env_on_expected_port():
    dockerfile = (ROOT / "Dockerfile.hud").read_text()
    normalized = re.sub(r"\s+", " ", dockerfile)

    assert "EXPOSE 8765" in dockerfile
    assert "hud" in normalized
    assert "serve" in normalized
    assert "env:env" in normalized
    assert "0.0.0.0" in normalized
    assert "8765" in normalized


def test_hud_dependencies_are_declared():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    dev_deps = data["project"]["optional-dependencies"]["dev"]

    assert "numpy" in deps
    assert any(dep.startswith("hud-python") for dep in deps)
    assert any(dep.startswith("fastmcp") for dep in deps)
    assert "pytest" in dev_deps
