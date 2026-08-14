# Dakota grammar train / held-out split

Registered split for grant-clean Tinker reruns. Frozen JSONL, not a live `train_test_split` of the same file.

| Field | Value |
| --- | --- |
| Source | `dakota_rl_training/datasets/grammar_tasks_complete.jsonl` (10,576 rows) |
| Seed | `42` |
| Held-out fraction | `0.1` per `task_type × difficulty` stratum |
| Used for GRPO advantages | **No** |
| Prompt overlap | Train drops any prompt that appears in held-out |

**Holdout v1 is frozen.** It is the `cebp9acs` eval set. Do not overwrite
`grammar_tasks_heldout.jsonl`. `scripts/rl/create_heldout_split.py` refuses to
replace it unless `--force` is passed.

Train JSONL was later repaired in `grammar_tasks_complete_v2.jsonl`. The same
repair applied to the v1 holdout rows is `grammar_tasks_heldout_v2.jsonl`.
Regenerate those files with:

```bash
python scripts/rl/repair_grammar_tasks.py
```

See `docs/experiments/TINKER_GRANT_CLEAN_RERUN.md` and `REPAIR_REPORT.json`.

Leaked gold `Examples:` / `Pattern:` lines are stripped at dataset load, not by rewriting the source JSONL.
