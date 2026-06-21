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

## Status

Reward, rollout, and GRPO signal: **built, tested, runnable now.** The Fireworks
weight-update job is the remaining step, gated on account credits + wall-clock —
not on any code we still need to write.
