"""Multi-turn (episodic) RFT evaluator for WanderAI scene-search.

Each row is a generated room; MCPGymRolloutProcessor launches our McpGym env
(server.py), the model plays a full episode via the move() tool, and the reward
is the whole-episode return (1.0 reach the ball, else fraction of distance closed).

Local check:  ep local-test wander_lake/test_wander_rl.py
Launch (RFT): ep create rft --dataset wander_lake/data/wander_dataset.jsonl \\
                --evaluator wander_lake/test_wander_rl.py \\
                --training-config-base-model accounts/fireworks/models/llama-v3p1-8b-instruct \\
                --training-config-output-model wander-rft-v4
"""
import os
from typing import Any, Dict, List

from eval_protocol.models import EvaluateResult, EvaluationRow, InputMetadata, Message
from eval_protocol.pytest import evaluation_test
from eval_protocol.pytest.default_mcp_gym_rollout_processor import MCPGymRolloutProcessor

# Serverless model for LOCAL validation; the RFT trains the --training-config-base-model.
_LOCAL_MODEL = os.environ.get("WANDER_RL_MODEL", "fireworks_ai/accounts/fireworks/models/gpt-oss-20b")


def wander_to_rows(data: List[Dict[str, Any]]) -> List[EvaluationRow]:
    rows = []
    for row in data:
        rows.append(EvaluationRow(
            messages=[Message(role="system", content=row["system_prompt"])],
            input_metadata=InputMetadata(
                row_id=row["id"],
                dataset_info={"environment_context": row["environment_context"],
                              "user_prompt_template": row["user_prompt_template"]},
            ),
        ))
    return rows


@evaluation_test(
    input_dataset=["wander_lake/data/wander_dataset.jsonl"],
    dataset_adapter=wander_to_rows,
    completion_params=[{"temperature": 0.7, "max_tokens": 512, "model": _LOCAL_MODEL,
                        "extra_body": {"reasoning_effort": "low"}}],
    rollout_processor=MCPGymRolloutProcessor(),
    passed_threshold=0.4,
    num_runs=1,
    max_concurrent_rollouts=3,
    mode="pointwise",
    server_script_path="wander_lake/server.py",
)
def wander_rl(row: EvaluationRow) -> EvaluationRow:
    score = row.get_total_reward()
    row.evaluation_result = EvaluateResult(
        score=score,
        reason="reached the ball" if score >= 0.99 else f"partial (return {score:.2f})")
    return row
