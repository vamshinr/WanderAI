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
# Base model must be BOTH rlLoraTunable (RFT-eligible — NOT just `tunable`, which is
# only SFT) AND not deprecated. The original llama-v3p1-8b-instruct was RFT-eligible
# but DEPRECATED 2025-11-26 → jobs died at ~2% with "Rollout job failed: Internal
# error" (v3 hvsj77kq, v4 c8ir3r0p/n2gu8jnx). llama-v3p2-3b-instruct is NOT
# rlLoraTunable → clean 400 "not supported for reinforcement fine-tuning".
# llama-v3-8b-instruct: rlLoraTunable=True, dep=None, trainingContextLength=131072,
# same 8B Llama-instruct class as the original — the correct current base.
# Multi-turn RFT needs a TOOL-CALLING base: the model drives the env via the move()
# tool. llama-v3-8b-instruct is rlLoraTunable but emits NO tool calls in the rollout
# → every episode is 0 steps / reward 0.0 → job fails. qwen3-4b is rlLoraTunable AND
# supportsTools AND non-deprecated AND <16B (free RFT tier) — the right multi-turn base.
BASE="${WANDER_BASE_MODEL:-accounts/fireworks/models/qwen3-4b}"
MAXTOK="${WANDER_MAX_TOKENS:-1536}"      # reasoning models need headroom for think + tool call

echo "Launching v4 multi-turn RFT -> ${OUT}  (base=${BASE##*/}, epochs=${EPOCHS}, lora_rank=${LORA}, max_tokens=${MAXTOK})"
# NOTE: do NOT pass --dataset (it expects an existing dataset *id*, not a path).
# ep infers the JSONL from the evaluator's input_dataset=[wander_lake/data/...],
# runs it through the dataset_adapter, and uploads it as a proper dataset resource.
exec ep create rft \
  --evaluator "${WANDER_EVALUATOR:-test-wander-rl-wander-rl}" \
  --training-config-base-model "${BASE}" \
  --training-config-output-model "${OUT}" \
  --training-config-epochs "${EPOCHS}" \
  --training-config-lora-rank "${LORA}" \
  --inference-parameters-max-output-tokens "${MAXTOK}" \
  --skip-validation \
  --force \
  --yes
