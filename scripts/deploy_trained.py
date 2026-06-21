"""Deploy a fine-tuned model (LoRA addon, live-merged) on a dedicated GPU, wait
until READY, then write its inference address (model#deployment) to
.trained_model.txt so the visualizer's "Trained" button uses it.

Run:      python scripts/deploy_trained.py wander-rft-v3
          (defaults to $WANDER_DEPLOY_MODEL or wander-rft-v1)
Teardown: python scripts/deploy_trained.py wander-rft-v3 --delete
"""
import os
import sys
import time
from fireworks import Fireworks

ACCT = os.environ.get("FIREWORKS_ACCOUNT_ID", "vamshinr5899-p0wudhc")
# model name = first non-flag arg, else env, else v1
_args = [a for a in sys.argv[1:] if not a.startswith("-")]
NAME = _args[0] if _args else os.environ.get("WANDER_DEPLOY_MODEL", "wander-rft-v1")
MODEL = f"accounts/{ACCT}/models/{NAME}"
TRAINED_FILE = os.path.join(os.path.dirname(__file__), "..", ".trained_model.txt")


def main():
    fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"])

    if "--delete" in sys.argv:
        for d in fw.deployments.list(account_id=ACCT):
            if getattr(d, "base_model", "") == MODEL:
                fw.deployments.delete(account_id=ACCT, deployment_id=d.name.split("/")[-1])
                print("deleted deployment:", d.name)
        return

    print(">>> creating deployment for", MODEL)
    # H100 (A100 hit a Fireworks INTERNAL error for this account).
    dep = fw.deployments.create(account_id=ACCT, base_model=MODEL,
                                accelerator_type="NVIDIA_H100_80GB", accelerator_count=1)
    dep_id = dep.name.split("/")[-1]
    print("    deployment:", dep.name)

    print(">>> waiting for READY")
    for i in range(90):
        d = fw.deployments.get(account_id=ACCT, deployment_id=dep_id)
        state = str(getattr(d, "state", "?"))
        print(f"    poll {i}: {state}", flush=True)
        if "READY" in state.upper():
            # A LoRA addon must be addressed as model#deployment.
            addr = f"{MODEL}#accounts/{ACCT}/deployments/{dep_id}"
            with open(TRAINED_FILE, "w") as fh:
                fh.write(addr + "\n")
            print("\n✅ DEPLOYMENT READY")
            print("   inference address:", addr)
            print("   wrote", TRAINED_FILE, "→ the UI 'Trained' button now uses this model")
            return
        if "FAILED" in state.upper():
            raise SystemExit(f"deployment failed: {state}")
        time.sleep(10)
    print("still not ready after timeout; check the dashboard")


if __name__ == "__main__":
    main()
