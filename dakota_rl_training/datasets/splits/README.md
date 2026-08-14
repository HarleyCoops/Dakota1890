# Dakota grammar train / held-out split

Registered split for grant-clean Tinker reruns. Frozen JSONL, not a live `train_test_split` of the same file.

| Field | Value |
| --- | --- |
| Source | `dakota_rl_training/datasets/grammar_tasks_complete.jsonl` (10,576 rows) |
| Seed | `42` |
| Held-out fraction | `0.1` per `task_type × difficulty` stratum |
| Used for GRPO advantages | **No** |
| Prompt overlap | Train drops any prompt that appears in held-out |

Regenerate (deterministic):

```bash
python scripts/rl/create_heldout_split.py
```

Leaked gold `Examples:` / `Pattern:` lines are stripped at dataset load, not by rewriting the source JSONL.
