# Dakota grammar train / held-out split

Registered split for grant-clean Tinker reruns.

| Field | Value |
| --- | --- |
| Source | `dakota_rl_training/datasets/grammar_tasks_complete.jsonl` (10,576 rows) |
| Seed | `42` |
| Held-out fraction | `0.1` |
| Algorithm | See `SPLIT_MANIFEST.json` → `algorithm` |
| Used for GRPO advantages | **No** |

Regenerate (deterministic):

```bash
python scripts/rl/create_heldout_split.py
```

Train is the source file minus held-out `(prompt, answer)` pairs. `tinker_train.py` defaults `--eval-path` to `grammar_tasks_heldout.jsonl` and excludes those items from the train bundle.
