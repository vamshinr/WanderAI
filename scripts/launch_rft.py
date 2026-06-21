"""Launch the WanderAI RFT job directly via the Fireworks SDK.

eval-protocol's `ep create rft` uploads the evaluator fine but then 404-loops on a
buggy evaluator-status poll (it builds the URL from the file path, not the
evaluator id). This does the same job creation it would, correctly: upload the
dataset, wait for the (already-uploaded) evaluator to go ACTIVE, then create the
reinforcement_fine_tuning_job.

Run: python scripts/launch_rft.py
"""
import os
import time
import requests
from fireworks import Fireworks
from eval_protocol.fireworks_rft import (
    create_dataset_from_jsonl, get_fireworks_api_base, get_fireworks_api_key,
)

ACCOUNT_ID = os.environ.get("FIREWORKS_ACCOUNT_ID", "vamshinr5899-p0wudhc")
EVALUATOR_ID = "rft-evaltest-wander-rftpy"
DATASET_ID = os.environ.get("WANDER_DATASET_ID", "wander-rft-train-v2")
# llama-v3p1-8b-instruct was DEPRECATED 2025-11-26 → RFT jobs die at ~2% with an
# internal error. llama-v3-8b-instruct is the current rlLoraTunable 8B-instruct base.
BASE_MODEL = os.environ.get("WANDER_BASE_MODEL", "accounts/fireworks/models/llama-v3-8b-instruct")
OUTPUT_MODEL = os.environ.get("WANDER_OUTPUT_MODEL", "wander-rft-v2")  # full path built below
JSONL = os.environ.get("WANDER_JSONL", "data/rft_train.jsonl")
EPOCHS = int(os.environ.get("WANDER_EPOCHS", "10"))
LORA_RANK = int(os.environ.get("WANDER_LORA_RANK", "16"))


def main():
    api_key = get_fireworks_api_key()
    api_base = get_fireworks_api_base()
    headers = {"Authorization": f"Bearer {api_key}"}
    evaluator_resource = f"accounts/{ACCOUNT_ID}/evaluators/{EVALUATOR_ID}"

    print(f">>> uploading dataset from {JSONL}")
    try:
        dsid, _ = create_dataset_from_jsonl(ACCOUNT_ID, api_key, api_base, DATASET_ID,
                                            "WanderAI RFT train", JSONL)
        dataset_resource = f"accounts/{ACCOUNT_ID}/datasets/{dsid}"
    except Exception as e:                       # already uploaded on a prior run
        print("    upload note:", str(e)[:160])
        dataset_resource = f"accounts/{ACCOUNT_ID}/datasets/{DATASET_ID}"
    print("    dataset:", dataset_resource)

    print(f">>> waiting for evaluator {EVALUATOR_ID} to become ACTIVE")
    for i in range(90):
        r = requests.get(f"{api_base}/v1/{evaluator_resource}", headers=headers, timeout=30)
        state = r.json().get("state") if r.ok else f"http {r.status_code}"
        print(f"    poll {i}: state={state}")
        if state == "ACTIVE":
            break
        if state in ("BUILD_FAILED", "FAILED"):
            raise SystemExit(f"evaluator build failed: {r.text[:300]}")
        time.sleep(10)

    print(">>> creating RFT job")
    fw = Fireworks(api_key=api_key, base_url=api_base)
    job = fw.reinforcement_fine_tuning_jobs.create(
        account_id=ACCOUNT_ID,
        evaluator=evaluator_resource,
        dataset=dataset_resource,
        training_config={"base_model": BASE_MODEL,
                         "output_model": f"accounts/{ACCOUNT_ID}/models/{OUTPUT_MODEL}",
                         "epochs": EPOCHS, "lora_rank": LORA_RANK},
        inference_parameters={"max_output_tokens": 128, "temperature": 0.7},
    )
    print("\n✅ RFT JOB CREATED:", job.name)
    print("   dashboard: https://app.fireworks.ai/dashboard/fine-tuning")


if __name__ == "__main__":
    main()
