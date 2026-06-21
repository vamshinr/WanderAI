"""Poll the v5 single-step RFT job and emit one stdout line per meaningful change
(state transition, Score milestone, terminal state). Run in the background; each
line is a notification. Importing this file is a no-op (work is under __main__).

Run:  python3 scripts/watch_v5.py            # job i4saoqyp by default
"""
import json
import os
import subprocess
import time

ACCT = "vamshinr5899-p0wudhc"
JOB = os.environ.get("WANDER_V5_JOB", "i4saoqyp")
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


def main():
    key = os.environ["FIREWORKS_API_KEY"]
    last_state, last_bucket = None, None
    print(f"V5 WATCH start job={JOB}", flush=True)
    for _ in range(3000):
        try:
            state, pct, score = job_status(key)
        except Exception as e:
            print(f"V5 poll error: {type(e).__name__} {e}", flush=True)
            time.sleep(20)
            continue
        if state != last_state:
            print(f"V5 STATE -> {state} (progress={pct}%, score={score})", flush=True)
            last_state = state
        if score is not None:
            bucket = round(score, 2)
            if bucket != last_bucket:
                print(f"V5 score ~{score} (progress={pct}%)", flush=True)
                last_bucket = bucket
        if state in TERMINAL:
            print(f"V5 DONE: {state}  finalScore~{score}", flush=True)
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
