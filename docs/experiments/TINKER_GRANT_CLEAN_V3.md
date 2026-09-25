# Experiment note: grant-clean v3 train JSONL

Language: Dakota grammar and morphology from Stephen Return Riggs, *Dakota-English Dictionary* (1890).

This is a **data merge**, not a rubric change and not a Tinker launch from this checkout. The live scorer in `environments/dakota_grammar_translation/.../environment.py` is unchanged. Frozen holdout v1 (`dakota_rl_training/datasets/grammar_tasks_heldout.jsonl`) stays the eval set.

## Why v3 is not in this repo yet

`dakota_rl_training/datasets/adaption_adapted_v2_filtered.jsonl` is not checked in. Do not invent Adaptive rows or Dakota forms. The v3 train file already exists on the maintainer machine from the recipe below.

## Recipe

`scripts/rl/merge_grammar_tasks_v3.py`:

1. Keep every `grammar_tasks_complete_v2.jsonl` row (9,287).
2. Append Adaptive rows whose `(prompt, answer)` is not already in v2 and whose prompt is not in holdout v1.
3. Append a second copy of every `reverse_translation` row (2× upsample).

```bash
python scripts/rl/merge_grammar_tasks_v3.py
```

The script refuses to write v3 if the Adaptive filtered file is missing. It never writes holdout v1.

## Local launch (maintainer machine, 2026-08-14)

v3 was launched locally from this recipe. Counts:

| Metric | Count |
| --- | ---: |
| v2_rows | 9,287 |
| adaptive_filtered | 523 |
| adaptive_already_in_v2 | 192 |
| adaptive_holdout_blocked | 0 |
| adaptive_new | 331 |
| v3_unique_rows | 9,618 |
| reverse_translation_unique | 2,265 |
| v3_train_rows_after_reverse_upsample | 11,883 |
| holdout_v1_rows | 1,060 (untouched) |

`--eval-path` stays `dakota_rl_training/datasets/grammar_tasks_heldout.jsonl`. When `grammar_tasks_complete_v3.jsonl` is committed, point `--dataset-path` at that file and set `recipe_name="dakota1890_grant_clean_v3"` on `train.Config` (required by current `tinker_cookbook`). Until then, `tinker_train.py` keeps the v2 default.

Do not run Tinker from CI or a coding agent unless a human is paying for Tinker.
