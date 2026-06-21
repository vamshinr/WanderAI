"""WanderAI demo EVAL SUITE — the 3D VISION track of the unified leaderboard.

Same agents, same reward, same HUD env as the 2D symbolic track (eval_tasks.py),
but here the agent perceives a real MuJoCo room through rendered RGB+depth instead
of a privileged symbolic view. The 'test' room is the held-out scene the model was
NOT trained on; 'train' is the scene v5 RFT data came from.

Run one agent:   hud eval eval_tasks_3d.py claude --gateway
Run the suite:   python run_eval_suite.py   (runs both tracks)
"""
from env import find_red_ball_3d, env  # noqa: F401  (env re-exported for `hud eval`)

VISION_SCENES = ["test", "train"]

tasks = []
for _s in VISION_SCENES:
    _t = find_red_ball_3d(scene=_s)
    _t.slug = f"find-ball-3d-{_s}"
    tasks.append(_t)
