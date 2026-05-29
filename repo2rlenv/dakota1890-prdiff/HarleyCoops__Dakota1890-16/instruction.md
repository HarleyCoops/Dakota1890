# Issue

**Title:** Add governance docs, offline CI, and reproducible eval kit

## Description

## Summary
- add Apache-2.0 licensing files, governance documents, and data stewardship notes
- introduce a small eval toolkit with fixtures, metrics, and documentation
- update onboarding docs, legacy notice, and CI to run offline lint/tests plus packaging

## Testing
- OFFLINE=1 pytest -q
- python eval/run_eval.py --pred eval/fixtures/sample_predictions.jsonl --truth eval/fixtures/sample_ground_truth.jsonl --out eval/report.md

------
https://chatgpt.com/codex/tasks/task_e_68e4386636d8832e921e2b6744fc5395

## Task

Modify the repository so that the issue described above is resolved. The repository is checked out at base commit `1976953518a0`. Edit files in place; the verifier captures your changes via `git diff` and scores them against an oracle patch using SWE-RL-style diff-similarity reward.