"""WanderAI scene-search tasks for HUD.

Run locally:   hud eval tasks.py claude --gateway
Each task is a generated room (by seed); the agent must reach the red ball.
"""
from env import find_red_ball, find_red_ball_3d, env  # noqa: F401  (env re-exported for `hud eval`)

_t0 = find_red_ball(seed=0); _t0.slug = "find-ball-room-0"
_t1 = find_red_ball(seed=1); _t1.slug = "find-ball-room-1"
_t2 = find_red_ball(seed=2); _t2.slug = "find-ball-room-2"

# 3D vision tasks (Phase B): the agent perceives a real MuJoCo room via RGB+depth.
_t3d_test = find_red_ball_3d(scene="test"); _t3d_test.slug = "find-ball-3d-test"
_t3d_train = find_red_ball_3d(scene="train"); _t3d_train.slug = "find-ball-3d-train"

tasks = [_t0, _t1, _t2, _t3d_test, _t3d_train]
