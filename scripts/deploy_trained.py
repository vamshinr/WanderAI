"""Deploy the fine-tuned wander-rft-v1 (LoRA addon, live-merged) on a dedicated
GPU so it can be served for inference, then wait until it's READY.

Run:     python scripts/deploy_trained.py
Teardown (stop paying): python scripts/deploy_trained.py --delete
"""
import os
import sys
import time
from fireworks import Fireworks

ACCT = os.environ.get("FIREWORKS_ACCOUNT_ID", "vamshinr5899-p0wudhc")
MODEL = f"accounts/{ACCT}/models/wander-rft-v1"


def main():
    fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

    if "--delete" in sys.argv:
        for d in fw.deployments.list(account_id=ACCT):
            if getattr(d, "base_model", "") == MODEL:
                fw.deployments.delete(account_id=ACCT, deployment_id=d.name.split("/")[-1])
                print("deleted deployment:", d.name)
        return

    print(">>> creating deployment for", MODEL)
    dep = fw.deployments.create(account_id=ACCT, base_model=MODEL,
                                accelerator_type="NVIDIA_A100_80GB", accelerator_count=1)
    dep_id = dep.name.split("/")[-1]
    print("    deployment:", dep.name)

    print(">>> waiting for READY")
    for i in range(60):
        d = fw.deployments.get(account_id=ACCT, deployment_id=dep_id)
        state = getattr(d, "state", "?")
        print(f"    poll {i}: {state}")
        if str(state).upper().endswith("READY") or str(state).upper() == "READY":
            print("\n✅ DEPLOYMENT READY:", dep.name)
            print("   inference model id:", MODEL)
            return
        if "FAILED" in str(state).upper():
            raise SystemExit(f"deployment failed: {state}")
        time.sleep(10)
    print("still not ready after timeout; check the dashboard")


if __name__ == "__main__":
    main()
