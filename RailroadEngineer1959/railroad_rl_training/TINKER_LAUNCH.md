# Railroad 1959 → Thinking Machines Tinker

This directory now mirrors the Dakota Tinker pipeline so you can run the **PrimeIntellect `railroad_1959` gym** (https://app.primeintellect.ai/dashboard/environments/harleycooper/railroad_1959) directly on Thinking Machines.

## What changed
- Added `railroad_rl_training/tinker_integration/` (env, dataset builder, ledger exporter).
- Replaced `tinker_train.py` with a real Tinker launcher wired to the railroad rubric and tasks.

## One‑command launch (defaults to safety_tasks_complete.json)
```bash
# 1) Env vars
export TINKER_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
# optional
export WANDB_API_KEY=...            # if you want W&B sync

# 2) Install deps if needed
pip install -r requirements.txt

# 3) Fire off a Tinker run (change model/paths as needed)
python RailroadEngineer1959/railroad_rl_training/tinker_train.py `
  --model-name Qwen/Qwen3-4B-Instruct-2507 `
  --log-path RailroadEngineer1959/outputs/tinker_railroad_run `
  --wandb-project railroad-1959-tinker `
  --wandb-name tinker-railroad-$(Get-Date -Format yyyyMMdd-HHmmss) `
  --batch-size 32 --group-size 8 --max-tokens 256 --learning-rate 5e-5 `
  --dataset-path RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json
```

## Outputs you get
- Tinker logs/checkpoints in `RailroadEngineer1959/outputs/tinker_railroad_run`
- Reward ledger CSV auto-exported to `RailroadEngineer1959/wandb_analysis/railroad_reward_ledger_tinker.csv`
- Optional `--sync-metrics-to-wandb` replays `metrics.jsonl` into the W&B run after training.

## Notes
- The rubric and prompts come from the PrimeIntellect gym: system prompt + Safety/Procedure/Terminology composite reward (now deterministic/local: exact-match + token-F1, no Anthropic calls).
- Evaluation split: provide `--eval-path` or rely on the default 10% held-out split; limit size with `--eval-examples`.
- To run a quick smoke test, add `--max-examples 64 --eval-examples 16` and reduce `--batch-size/--group-size`.
