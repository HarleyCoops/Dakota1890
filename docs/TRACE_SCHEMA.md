# Trace schema (version 1)

`lexgym/tinker_env.py` writes one JSON line per rollout to `<log-path>/traces/train.jsonl` and `<log-path>/traces/eval.jsonl`. A row holds everything needed to read the rollout, recompute its reward, and reuse it for off-policy training.

| Field | Content |
|---|---|
| `trace_version` | `1` |
| `run.run_id`, `run.stage` | Run name and a stage label (for example `rl-after-sft`) |
| `run.base_model`, `run.load_checkpoint_path`, `run.renderer` | Model identity. The checkpoint path is the Tinker state the run started from. |
| `run.temperature`, `run.max_tokens`, `run.train_group_size`, `run.train_batch_size`, `run.seed` | Sampling and batch settings |
| `run.lexgym_commit`, `run.scorer_version`, `run.task_files_sha256` | Code and data versions |
| `time` | UTC time when the group was graded |
| `position.role` | `train` or `eval` |
| `position.pass_index` | Train: epoch. Eval: number of the evaluation pass; pass *k* runs after training step *k* × `eval_every`. |
| `position.batch_index`, `position.group_index`, `position.rollout_index`, `position.group_size` | Place of the rollout in the run. In `train`, `batch_index` is the training step. |
| `task` | The full task row: `id`, `task`, `split`, `prompt`, `answer`, `accepted`, and `info` with `entry_id`, `page`, `source_image`, `family` |
| `messages` | The whole conversation: system prompt, question, each `LOOKUP` line, each dictionary entry returned, and the final answer |
| `turns[]` | One element per model turn: `ob_tokens` (prompt token ids), `ac_tokens` (sampled token ids), `ac_logprobs` (log-probability of each sampled token under the sampling policy), `stop_reason`, `sequence_id` (Tinker's id for the sample), `reward` |
| `scoring.answer_span` | The text that the scorer extracted |
| `scoring.ledger` | Every reward component |
| `scoring.reward`, `scoring.group_mean_reward`, `scoring.advantage` | Reward, the mean of its group, and their difference (the advantage the cookbook loop uses) |

Notes:

- The sampling policy of a `train` row at step *s* is the model after *s* updates. An `eval` group has one rollout, so its advantage is 0.
- `ob_tokens` of a later turn contains the earlier turns, so the last turn's `ob_tokens` plus its `ac_tokens` is the full token sequence.
- The cookbook's own `rollout_summaries.jsonl` files hold the ledger but no text, no token ids, and no task id. The first pilot (10 November 2026) has only those.
