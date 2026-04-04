# Setup

These instructions are the maintained step-0 setup path for reproducing the Dakota core workflow in a sandbox.

## Python

- Python `3.10+`
- Consumer-hardware target for local checks: single GPU when available, CPU-only fallback for dataset and environment validation

## Install

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_hf_inference.txt
python -m pip install -e environments/dakota_grammar_translation
```

Notes:

- `requirements.txt` now pins `huggingface-hub<1.0` because the HF inference path depends on a `transformers`-compatible hub version.
- `requirements_hf_inference.txt` now includes `peft` for adapter loading.

## Environment Variables

Create `.env` with the keys you actually plan to use:

```bash
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...
WANDB_API_KEY=...
TINKER_API_KEY=...
```

## Cost Triggers

- Anthropic grammar extraction: the current grammar extraction script estimates about `$0.25/page`
- Anthropic dictionary extraction: cost is not encoded in the script; treat as paid API usage
- Gemini synthetic QA: paid API usage, proportional to prompt volume
- OpenAI baseline submission: paid API usage if you create a fine-tune job
- Tinker / PrimeIntellect RL: remote compute cost depends on chosen platform and hardware

## Offline Smoke Checks

```bash
python -m pytest -q
python dakota_rl_training/train.py --check-only
python scripts/rl/dakota_openai_finetune.py --check-only
```

## Minimal Live Smokes

Run these only if the API keys are valid and funded:

```bash
python scripts/extraction/extract_grammar_pages.py --test
python scripts/extraction/extract_dakota_dictionary_v2.py --test

python scripts/conversion/generate_synthetic_dakota.py \
  --extracted-dir data/extracted \
  --pairs-per-language 1 \
  --context-size 2 \
  --output-file data/bilingual_training_set_smoke.jsonl

python scripts/conversion/convert_extracted_to_chat.py \
  --input-file data/bilingual_training_set_smoke.jsonl \
  --output-dir OpenAIFineTune/smoke
```

## Known Constraints

- `scripts/extraction/extract_grammar_pages.py` now exits nonzero when page extraction fails.
- Local inference still requires a runtime whose installed `huggingface-hub` version matches the pinned requirements; the current sandbox may need a reinstall before `run_inference.py` succeeds.
- HF remote inference also depends on token permissions for Inference Providers.
