# A5 — RFT launch recipe

Reinforcement-fine-tune the navigation policy on our verifiable geodesic reward.
**Generate** (the model acts in `SceneSearchEnv`) → **score** (`wanderai/rft.py`
`episode_reward`, 0–1) → **update** (GRPO on Fireworks). The generate + score
halves are built and verified locally; the weight update runs on Fireworks and is
the one paid, multi-hour step.

## What's built and verified (no cost)

- **Reward** — `wanderai/rft.py:episode_reward`. Dense + bounded [0,1]: partial
  credit for the fraction of geodesic distance closed, plus an SPL bonus on
  success. Dense because an untrained model rarely reaches the ball; pure success
  would be all-zero (no gradient).
- **Rollout** — `run_scored(scene, policy)` runs one episode and scores it. The
  policy is `LLMPolicy` (the model being trained), driving the env via the text
  observation.
- **Training signal** — `grpo_preview(scene, policy_factory, group_size)` samples
  a group of trajectories, scores each, and returns **GRPO advantages**
  (rewards standardized to mean 0). Positive = reinforce, negative = suppress.
  Run it live against the real model:

  ```bash
  python scripts/grpo_preview.py --group 4 --max-steps 14 --temperature 0.8
  ```

  Non-zero spread in the advantages = a real learning signal exists.

## The weight update on Fireworks (the paid step)

- **Model:** `accounts/fireworks/models/gpt-oss-20b` — serverless (so we can serve
  it) **and** `rlTunable` (so we can RFT it).
- **Tooling:** `pip install eval-protocol` → `ep create` / `firectl rftj create`.
- **Evaluator** to wire up (uses our reward verbatim):

  ```python
  # rft_evaluator.py — scaffold; verify against current eval-protocol docs before launch
  from eval_protocol import evaluation_test, EvaluationRow
  from wanderai.scene_gen import make_split
  from wanderai.environment import EnvConfig
  from wanderai.llm_policy import LLMPolicy
  from wanderai.rft import run_scored

  TRAIN, _ = make_split(n_train=24, n_test=0, seed=1)

  @evaluation_test(input_dataset=[EvaluationRow(scene_index=i) for i in range(len(TRAIN))])
  def wander_reward(row):
      scene = TRAIN[row.scene_index]
      # The harness serves the in-training model; LLMPolicy points at it.
      roll = run_scored(scene, LLMPolicy(), EnvConfig(max_steps=80))
      return roll.reward            # already in [0, 1]
  ```

- **Launch:**

  ```bash
  ep upload                                   # register the evaluator
  ep create rft \
    --base-model accounts/fireworks/models/gpt-oss-20b \
    --output-model wander-rft-v1
  ```

- **Cost / time:** gpt-oss-20b is 20B → **not** in the free-RFT (<16B) tier →
  per-GPU-hour (~$7/hr), and a run takes **hours**. A fresh Fireworks account needs
  billing/credits enabled before this will start. To stay free, swap to a
  `rlTunable` model under 16B (re-check the catalog filter in
  `wander-integration-architecture` memory).

## After training — measure generalization

Point the policy at the fine-tuned model and run the held-out eval:

```bash
FIREWORKS_MODEL=accounts/<account>/models/wander-rft-v1 \
  python -m wanderai.evaluate --llm --test 10
```

Success = **held-out SPL above the untrained baseline**, trending toward the
oracle's 1.0. That is the scene-agnostic result — the whole project's payoff.

## As-built launch (single-step RFT) — what actually runs

We launch single-step (contextual-bandit) RFT — simpler and more robust than the
multi-turn MCP path, and it teaches greedy geodesic descent (which the oracle
proves solves the task):

1. **Dataset** — `python scripts/build_rft_dataset.py` → `data/rft_train.jsonl`.
   Each row = an observation prompt + a precomputed, direction-aware 0–1 reward
   for each action (`wanderai/rft.py:single_step_reward`).
2. **Evaluator** — `rft_eval/test_wander_rft.py`. Self-contained (reward lookup,
   no `wanderai` import) so it ships to Fireworks trivially. Needs an (empty)
   `requirements.txt` at repo root for `ep` to upload it.
3. **Launch** — `python scripts/launch_rft.py`. Uploads the dataset, waits for the
   evaluator to go ACTIVE, and creates the `reinforcement_fine_tuning_job` via the
   Fireworks SDK directly.
   - Base model: `accounts/fireworks/models/llama-v3p1-8b-instruct` (8B, open,
     RL-tunable). Output: `wander-rft-v1`.
   - **Why not `ep create rft`?** It uploads the evaluator fine but then 404-loops
     on a buggy evaluator-status poll (builds the URL from the file path, not the
     evaluator id). `scripts/launch_rft.py` does the same job creation correctly.

## Status (verified 2026-06-20)

Pipeline runs **end to end**: dataset uploaded ✓, evaluator built + ACTIVE ✓,
job-creation request well-formed and accepted ✓. The job is blocked by one
account-side gate: Fireworks returns `400 payment method is required` for RFT
(GPU-hours), **even with credits on the account**.

**To actually train:** add a payment method in Fireworks billing
(https://app.fireworks.ai → Billing), then run `python scripts/launch_rft.py`.
Training then runs on Fireworks (hours). After it finishes, deploy `wander-rft-v1`
and evaluate held-out: `FIREWORKS_MODEL=accounts/<acct>/models/wander-rft-v1
python -m wanderai.evaluate --llm --test 10`.

Reward, dataset, evaluator, and launcher: **built, tested, and validated against
the live API.** Only the payment-method toggle remains — not any code.
