"""Fireworks RFT evaluator for WanderAI (self-contained — no wanderai import).

Each dataset row carries the precomputed reward for every action, so scoring is a
lookup: parse the action the model chose, return its reward. The model is sampled
multiple times per prompt by Fireworks; GRPO turns these rewards into advantages
and updates the weights toward the better-scoring actions.

Local check:   ep local-test rft_eval/test_wander_rft.py
Launch (RFT):  ep create rft --dataset data/rft_train.jsonl \\
                 --evaluator rft_eval/test_wander_rft.py \\
                 --training-config-base-model accounts/fireworks/models/llama-v3p1-8b-instruct \\
                 --training-config-output-model wander-rft-v1
"""
import json
import os
import re

from eval_protocol import evaluation_test, EvaluationRow, SingleTurnRolloutProcessor
from eval_protocol.models import EvaluateResult

_ACTIONS = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT")
_ACTION_RE = re.compile(r"ACTION\s*=\s*(MOVE_FORWARD|TURN_LEFT|TURN_RIGHT)")
_DATASET = os.path.join(os.path.dirname(__file__), "..", "data", "rft_train.jsonl")
_MODEL = os.environ.get("WANDER_RFT_MODEL",
                        "fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct")


def parse_action(text: str) -> str:
    up = (text or "").upper()
    m = _ACTION_RE.search(up)
    if m:
        return m.group(1)
    hits = [(up.rfind(a), a) for a in _ACTIONS if a in up]
    return max(hits)[1] if hits else "MOVE_FORWARD"


@evaluation_test(
    input_dataset=[_DATASET],
    completion_params=[{"model": _MODEL, "max_tokens": 128, "temperature": 0.7}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def wander_rft(row: EvaluationRow) -> EvaluationRow:
    gt = row.ground_truth
    rewards = (json.loads(gt) if isinstance(gt, str) else gt)["rewards"]
    action = parse_action(row.messages[-1].content if row.messages else "")
    score = float(rewards.get(action, 0.0))
    row.evaluation_result = EvaluateResult(
        score=score, reason=f"chose {action} -> reward {score}", is_score_valid=True)
    return row
