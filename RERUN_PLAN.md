# Dakota1890 RL Rerun Plan

## Scope Decision

The canonical project focus is Dakota1890 and the RL grammar-training path:

1. 1890 Riggs source assets
2. Claude/VLM extraction
3. organized grammar rules
4. RL task generation
5. Dakota grammar verifier environment
6. Tinker/Qwen RL training and published adapter

The OpenAI SFT path is a comparison baseline only. Stoney Nakoda/OpenAI SFT work is a secondary project and should not be treated as the primary Dakota1890 claim.

## Verified State

- `data/rl_training_rules/all_rl_rules.json`: 1,497 rules
- `dakota_rl_training/datasets/grammar_tasks_complete.jsonl`: 10,576 tasks
- packaged environment dataset: 10,576 tasks
- `OpenAIFineTune/dakota_train.jsonl`: 980 examples
- `OpenAIFineTune/dakota_valid.jsonl`: 245 examples
- Tinker 30B run completed to final checkpoint:
  - `dakota_rl_training/outputs/tinker_qwen30b/checkpoints.jsonl`
  - final state: `tinker://da1ef918-d67a-5080-b500-dd1256db9ca7:train:0/weights/final`
  - 199 logged training steps, `progress/done_frac = 1.0`

OpenAI account state from a non-mutating fine-tune job list:

- Dakota job exists and succeeded, but it is old:
  - job: `ftjob-3bFeF3y6erpxy4NYPxYBGxav`
  - created: `2025-11-05T13:48:08+00:00`
  - base model: `gpt-3.5-turbo-0125`
  - files: `dakota_train.jsonl` / `dakota_valid.jsonl`
- The current repo default `gpt-4.1-mini-2025-04-14` Dakota OpenAI baseline has readiness coverage, but no matching launched job was found in the recent account job list.
- Stoney Nakoda jobs exist in the OpenAI account and are secondary to Dakota1890.

## Fixed Before Rerun

The RL verifier now treats bracketed grammar placeholders as literal patterns when regex matching does not match. The previous Tinker run predates this fix; its metrics show `env/all/ledger/pattern_raw = 0.0` at the final step. Do not use that run as evidence that the pattern reward component worked.

Regression coverage:

```bash
python -m pytest tests/test_verifier_integration.py -q
```

## Next Commands

Run local non-billing checks:

```bash
python -m pytest tests/test_verifier_integration.py tests/test_inference_configuration.py tests/test_training_dataset_builder.py tests/test_offline_eval.py tests/test_openai_finetune_readiness.py tests/test_sft_baseline.py -q
python dakota_rl_training/train.py --check-only
python scripts/rl/dakota_openai_finetune.py --check-only
```

PowerShell audit shortcut:

```powershell
.\scripts\check_windows_tooling.ps1 -UseSystemPython -Full
```

Rerun the canonical Dakota1890 Tinker RL path after the reward fix:

```bash
python dakota_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --wandb-project dakota-rl-grammar \
  --wandb-name qwen3-30b-dakota-pattern-reward-rerun \
  --log-path dakota_rl_training/outputs/tinker_qwen30b_pattern_fix \
  --batch-size 48 \
  --group-size 16 \
  --max-tokens 384 \
  --learning-rate 4e-5 \
  --lora-rank 32 \
  --eval-every 20 \
  --save-every 20
```

## Audit Fix Order

1. Rerun Dakota1890 RL after the verifier fix and archive metrics/checkpoint IDs.
2. Build `run_pipeline.py` as the single audited entry point with `--check-only`, `--from-existing-extraction`, `--skip-ocr`, and `--stage` options.
3. Create a small human-validated held-out Dakota test set with provenance and reviewer fields.
4. Use `scripts/check_windows_tooling.ps1` to keep Windows/PowerShell audit checks separate from WSL/Linux-first RL training.
5. Add language config YAML only after the Dakota path is stable.
6. Add community-in-the-loop review fields for register, temporal drift, and approval status.
