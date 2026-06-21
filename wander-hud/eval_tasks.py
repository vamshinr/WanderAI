"""WanderAI demo EVAL SUITE — a fixed, held-out HUD taskset for the leaderboard.

These 8 seeds were NEVER used to train any model (RFT training used seeds 1-24),
so the score measures generalization, not memorization. Kept separate from
tasks.py (which also carries the 3D-vision stretch tasks) so the eval is a clean,
apples-to-apples 2D symbolic comparison across agents.

Run one agent:   hud eval eval_tasks.py claude --gateway
Run the suite:   python run_eval_suite.py
"""
from env import find_red_ball, env  # noqa: F401  (env re-exported for `hud eval`)

# Held-out eval seeds (verified solvable, mean optimal geodesic ~4.4).
EVAL_SEEDS = [300, 301, 302, 303, 304, 305, 306, 307]

tasks = []
for _s in EVAL_SEEDS:
    _t = find_red_ball(seed=_s)
    _t.slug = f"find-ball-room-{_s}"
    tasks.append(_t)
