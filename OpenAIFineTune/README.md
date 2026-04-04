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

This is non-billing and validates the local files plus the configured base model:

```bash
python scripts/rl/dakota_openai_finetune.py --check-only
```

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
