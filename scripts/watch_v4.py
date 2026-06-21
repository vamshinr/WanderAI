"""Poll the v4 multi-turn RFT job and emit ONE stdout line per meaningful change
(state transition, score milestone, new deployable checkpoint, terminal state).

Designed to be driven by a Monitor: each printed line becomes a notification.
All work is under __main__ so importing this file (e.g. pytest collection) is a
no-op — unlike the old await_v3.py, which ran its poll loop at import time.

Run:  python3 scripts/watch_v4.py            # job c8ir3r0p by default
"""
import json
import os
import subprocess
import sys
import time

ACCT = "vamshinr5899-p0wudhc"
JOB = os.environ.get("WANDER_V4_JOB", "c8ir3r0p")
BASE = f"https://api.fireworks.ai/v1/accounts/{ACCT}"
TERMINAL = {"JOB_STATE_COMPLETED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}


def _curl(url, key):
    out = subprocess.run(["curl", "-s", "-m", "25", url, "-H", f"Authorization: Bearer {key}"],
                         capture_output=True, text=True).stdout
    try:
        return json.loads(out)
    except Exception:
        return {}


def job_status(key):
    d = _curl(f"{BASE}/reinforcementFineTuningJobs/{JOB}", key)
    score = None
    try:
        curve = json.loads(d["outputMetrics"])["curves"]["average"]["Score"]
        if curve:
            score = round(float(curve[-1]), 3)
    except Exception:
        pass
    pct = d.get("jobProgress", {}).get("percent")
    return d.get("state", "?"), pct, score


def v4_checkpoints(key):
    """Deployable checkpoint models named wander-rft-v4*."""
    d = _curl(f"{BASE}/models?pageSize=200", key)
    names = []
    for m in d.get("models", []):
        nm = m.get("name", "").split("/")[-1]
        if nm.startswith("wander-rft-v4"):
            names.append(nm)
    return sorted(names)


def main():
    key = os.environ["FIREWORKS_API_KEY"]
    last_state, last_score_bucket, seen_ckpts = None, None, set()
    print(f"V4 WATCH start job={JOB}", flush=True)
    for _ in range(2000):                       # ~10h at 18s; Monitor bounds the real lifetime
        try:
            state, pct, score = job_status(key)
        except Exception as e:
            print(f"V4 poll error: {type(e).__name__} {e}", flush=True)
            time.sleep(18)
            continue

        if state != last_state:
            print(f"V4 STATE -> {state} (progress={pct}%, score={score})", flush=True)
            last_state = state

        if score is not None:
            bucket = round(score, 1)
            if bucket != last_score_bucket:
                print(f"V4 score ~{score} (progress={pct}%)", flush=True)
                last_score_bucket = bucket

        for ck in v4_checkpoints(key):
            if ck not in seen_ckpts:
                seen_ckpts.add(ck)
                print(f"V4 CHECKPOINT ready: {ck}  (deploy via scripts/deploy_trained.py {ck})", flush=True)

        if state in TERMINAL:
            print(f"V4 DONE: {state}  bestScore~{score}  checkpoints={sorted(seen_ckpts)}", flush=True)
            return
        time.sleep(18)


if __name__ == "__main__":
    main()
