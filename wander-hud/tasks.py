"""WanderAI scene-search tasks for HUD.

Run locally:   hud eval tasks.py claude --gateway
Each task is a generated room (by seed); the agent must reach the red ball.
"""
from env import find_red_ball, env  # noqa: F401  (env re-exported for `hud eval`)

_t0 = find_red_ball(seed=0); _t0.slug = "find-ball-room-0"
_t1 = find_red_ball(seed=1); _t1.slug = "find-ball-room-1"
_t2 = find_red_ball(seed=2); _t2.slug = "find-ball-room-2"

tasks = [_t0, _t1, _t2]
