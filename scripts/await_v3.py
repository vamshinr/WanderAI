"""Poll the v3 RFT job for one ~9-minute window. Prints a marker so the caller
knows whether to deploy (V3_DONE), keep waiting (V3_RUNNING), or stop (V3_FAILED).
Kept short because background commands are capped ~10 min — relaunch until done.
"""
import json
import os
import subprocess
import sys
import time

ACCT = "vamshinr5899-p0wudhc"
JOB = os.environ.get("WANDER_V3_JOB", "hvsj77kq")
KEY = os.environ["FIREWORKS_API_KEY"]
URL = f"https://api.fireworks.ai/v1/accounts/{ACCT}/reinforcementFineTuningJobs/{JOB}"


def status():
    out = subprocess.run(["curl", "-s", "-m", "20", URL, "-H", f"Authorization: Bearer {KEY}"],
                         capture_output=True, text=True).stdout
    d = json.loads(out)
    score = []
    try:
        score = json.loads(d["outputMetrics"])["curves"]["average"]["Score"]
    except Exception:
        pass
    return d.get("state", "?"), d.get("jobProgress", {}).get("percent"), score


for i in range(36):                       # ~9 min at 15s
    state, pct, score = status()
    last = round(score[-1], 3) if score else "?"
    print(f"poll {i}: {state} {pct}% Score~{last}", flush=True)
    if state == "JOB_STATE_COMPLETED":
        print("V3_DONE", flush=True)
        sys.exit(0)
    if "FAILED" in str(state).upper() or "CANCEL" in str(state).upper():
        print("V3_FAILED", flush=True)
        sys.exit(1)
    time.sleep(15)
print(f"V3_RUNNING {pct}%", flush=True)
