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

Current step-0 readiness output in this repo:

- default base model: `gpt-4.1-mini-2025-04-14`
- train split: `980` examples, about `220,983` tokens
- validation split: `245` examples, about `54,956` tokens
- estimated trained tokens at `3` epochs: about `662,949`

## Prelaunch Gate

Before submitting a paid job:

- rerun `--check-only` and record the token estimate
- confirm the active fine-tunable model in the official OpenAI docs
- confirm current OpenAI pricing before launch
- keep the run under the current project budget cap

Point-in-time step-0 estimate, dated `2026-04-04`:

- official OpenAI model-optimization docs list supervised fine-tuning support for `gpt-4.1-2025-04-14`, `gpt-4.1-mini-2025-04-14`, and `gpt-4.1-nano-2025-04-14`
- the official OpenAI pricing surface currently lists `gpt-4.1 mini` training at about `$5.00 / 1M tokens`
- with an estimated `662,949` trained tokens at `3` epochs, the Dakota baseline is about `$3.31` in training spend before any later inference usage

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
