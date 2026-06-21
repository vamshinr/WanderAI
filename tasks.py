from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from env import env, find_red_ball


TASKSET_NAME = "wanderai-scene-search"


@dataclass(frozen=True)
class _LocalTaskRow:
    env: str
    id: str
    args: dict[str, Any]
    slug: str
    columns: dict[str, Any]
    agent_config: dict[str, Any]


def _row(
    slug: str,
    *,
    scene_id: str,
    seed: int,
    split: str,
    difficulty: str,
    agent_max_steps: int,
):
    args = {"scene_id": scene_id, "seed": seed}
    columns = {
        "task": "find_red_ball",
        "scene_id": scene_id,
        "seed": seed,
        "split": split,
        "difficulty": difficulty,
        "agent_max_steps": agent_max_steps,
    }
    agent_config = {"max_steps": agent_max_steps}

    if hasattr(find_red_ball, "env") and hasattr(find_red_ball, "id"):
        task = find_red_ball(**args)
        task.slug = slug
        task.columns = columns
        task.agent_config = agent_config
        return task

    return _LocalTaskRow(
        env=getattr(env, "name", TASKSET_NAME),
        id="find_red_ball",
        args=args,
        slug=slug,
        columns=columns,
        agent_config=agent_config,
    )


tasks = [
    _row(
        "find-red-ball-default-seed-0",
        scene_id="default",
        seed=0,
        split="smoke",
        difficulty="medium",
        agent_max_steps=80,
    ),
    _row(
        "find-red-ball-open-room-seed-1",
        scene_id="open-room",
        seed=1,
        split="eval",
        difficulty="easy",
        agent_max_steps=120,
    ),
    _row(
        "find-red-ball-turn-room-seed-2",
        scene_id="turn-room",
        seed=2,
        split="eval",
        difficulty="medium",
        agent_max_steps=160,
    ),
    _row(
        "find-red-ball-occlusion-room-seed-3",
        scene_id="occlusion-room",
        seed=3,
        split="eval",
        difficulty="hard",
        agent_max_steps=180,
    ),
]
