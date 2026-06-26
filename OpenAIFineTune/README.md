# OpenAI SFT Baseline

This folder is the maintained supervised fine-tuning baseline for Dakota1890.

It exists to answer a specific research question:

- Does the Dakota Grammar Gym RL path materially outperform a plain SFT baseline built from the same synthetic dataset?

## Files

- `dakota_train.jsonl` — OpenAI chat-format training split
- `dakota_valid.jsonl` — OpenAI chat-format validation split

## Generation Path

```bash
python scripts/conversion/generate_synthetic_dakota.py \
  --extracted-dir data/extracted \
  --pairs-per-language 8 \
  --output-file data/bilingual_training_set.jsonl

python scripts/conversion/convert_extracted_to_chat.py \
  --input-file data/bilingual_training_set.jsonl \
  --output-dir OpenAIFineTune
```

## Readiness Check

This is non-billing and validates the local files plus the configured base model. It now also emits token estimates for the current OpenAI chat-format splits:

```bash
python scripts/rl/dakota_openai_finetune.py --check-only
```

Current baseline target:

- default base model: `gpt-4.1-2025-04-14`
- train split after full original Dakota conversion: `1,956` examples
- validation split after full original Dakota conversion: `489` examples
- estimated trained tokens are emitted by `--check-only`

## Prelaunch Gate

Before submitting a paid job:

- rerun `--check-only` and record the token estimate
- confirm the active fine-tunable model in the official OpenAI docs
- confirm current OpenAI pricing before launch
- keep the run under the current project budget cap

Point-in-time model check, updated `2026-06-10`:

- OpenAI's supervised fine-tuning guide points to model docs for supported models and uses the GPT-4.1 family in examples.
- Current GPT-5 family model pages mark fine-tuning as not supported.
- `o4-mini-2025-04-16` is fine-tuning-capable for reinforcement fine-tuning, but this baseline is supervised fine-tuning on chat examples.
- For the strongest current SFT baseline, use `gpt-4.1-2025-04-14`.
- Run `3` epochs for the fixed baseline. If validation behavior becomes too narrow or memorized, rerun at `2`; if the model under-follows Dakota forms, rerun at `4`.

Recheck this before launch. Model support and pricing are both time-sensitive.

## Remote Launch

This submits a paid remote OpenAI fine-tuning job. Set `OPENAI_API_KEY` first. Optional controls:

- `OPENAI_FINETUNE_MODEL` — fine-tunable base model override
- `OPENAI_FINETUNE_EPOCHS` — epoch count override
- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN_NAME` — experiment tracking
- `HUGGINGFACE_TOKEN`, `HUGGINGFACE_DATASET_REPO`, `HUGGINGFACE_DATASET_PRIVATE` — dataset publishing

```bash
python scripts/rl/dakota_openai_finetune.py
```

The launcher uploads the train/validation files, creates the fine-tuning job, and monitors it to completion.

For an AutoScientist baseline launch where the remote job should continue after
the local command exits, use:

```bash
python scripts/rl/dakota_openai_finetune.py --launch-only \
  --ledger OpenAIFineTune/runs/dakota-openai-sft-baseline.json
```

The ledger records the base model, split counts, token estimates, uploaded file
IDs, and OpenAI fine-tuning job ID without storing secrets.

If file upload succeeds but job creation is blocked by quota or billing, reuse
the uploaded file IDs from the ledger:

```bash
python scripts/rl/dakota_openai_finetune.py --launch-only \
  --ledger OpenAIFineTune/runs/dakota-openai-sft-baseline.json \
  --training-file-id file-... \
  --validation-file-id file-...
```
