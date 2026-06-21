#!/usr/bin/env bash
# Launch the v4 *multi-turn (episodic)* RFT job on Fireworks via eval-protocol.
#
# Unlike v1/v3 (single-step contextual-bandit RFT), v4 trains on whole episodes:
# the model drives our McpGym env (wander_lake/) via the move() tool turn after
# turn, and GRPO sees the whole-episode return. Overshooting / oscillating now
# tanks the return, so the policy learns to actually search and reach the ball.
#
# Self-gating: ep uploads + BUILDS the evaluator on Fireworks and aborts BEFORE
# any paid training if that build fails (BUILD_FAILED). So a wrong package set
# costs nothing — training only starts once the evaluator is ACTIVE.
set -euo pipefail
cd "$(dirname "$0")/.."

# Fireworks' MCPGymRolloutProcessor launches the env with `python server.py`;
# make `python` resolve to a real python3 (anaconda's lacks our deps locally).
mkdir -p /tmp/pybin && ln -sf "$(command -v python3)" /tmp/pybin/python
export PATH=/tmp/pybin:$PATH
export PYTHONUNBUFFERED=1

# Load secrets (FIREWORKS_API_KEY) without printing them.
set -a; [ -f .env ] && . ./.env; set +a

ACCT="${WANDER_ACCOUNT:-vamshinr5899-p0wudhc}"
OUT="${WANDER_OUTPUT_MODEL:-accounts/${ACCT}/models/wander-rft-v4}"
EPOCHS="${WANDER_EPOCHS:-10}"
LORA="${WANDER_LORA_RANK:-16}"

echo "Launching v4 multi-turn RFT -> ${OUT}  (epochs=${EPOCHS}, lora_rank=${LORA})"
# NOTE: do NOT pass --dataset (it expects an existing dataset *id*, not a path).
# ep infers the JSONL from the evaluator's input_dataset=[wander_lake/data/...],
# runs it through the dataset_adapter, and uploads it as a proper dataset resource.
exec ep create rft \
  --evaluator test-wander-rl-wander-rl \
  --training-config-base-model accounts/fireworks/models/llama-v3p1-8b-instruct \
  --training-config-output-model "${OUT}" \
  --training-config-epochs "${EPOCHS}" \
  --training-config-lora-rank "${LORA}" \
  --skip-validation \
  --force \
  --yes
