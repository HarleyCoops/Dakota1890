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

## Windows / PowerShell

For Windows audit checks, use a Windows venv or the known-good system Python explicitly. The repo ignores `.venv_win/` and `.venv_linux_broken/` so local environment folders do not pollute `git status`.

```powershell
py -3.12 -m venv .venv_win
.\.venv_win\Scripts\python.exe -m pip install --upgrade pip
.\.venv_win\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv_win\Scripts\python.exe -m pip install -r requirements_hf_inference.txt
.\.venv_win\Scripts\python.exe -m pip install -e .\environments\dakota_grammar_translation

.\scripts\check_windows_tooling.ps1 -Full
```

If `.venv_win` is incomplete but the system Python already has the project dependencies, run:

```powershell
.\scripts\check_windows_tooling.ps1 -UseSystemPython -Full
```

Known Windows host issue: in some Codex desktop sessions `rg` resolves to the bundled `WindowsApps\OpenAI.Codex...\rg.exe` and fails with `Access is denied`. That is a host tooling issue, not a Dakota pipeline issue. Install a normal ripgrep earlier in `PATH`, for example:

```powershell
winget install BurntSushi.ripgrep.MSVC
```

Long-running RL training remains WSL/Linux-first because Tinker/PrimeIntellect launch scripts and GPU runtimes are Linux-oriented. Windows/PowerShell is supported for repo audit checks, data validation, OpenAI readiness checks, and non-billing verifier tests.

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

The OpenAI readiness check now reports:

- resolved train/validation asset paths
- the configured fine-tune base model
- estimated train, validation, and total trained tokens for the current epoch count

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

python scripts/rl/dakota_openai_finetune.py --check-only
```

## Known Constraints

- `scripts/extraction/extract_grammar_pages.py` now exits nonzero when page extraction fails.
- Local inference still requires a runtime whose installed `huggingface-hub` version matches the pinned requirements; the current sandbox may need a reinstall before `run_inference.py` succeeds.
- HF remote inference also depends on token permissions for Inference Providers.
- The OpenAI SFT launcher can submit a paid remote fine-tuning job via `python scripts/rl/dakota_openai_finetune.py`, but step-0 validation only runs the readiness check by default.
- As of the current step-0 audit, the launcher defaults to `gpt-4.1-mini-2025-04-14` and should be treated as budget-gated remote work.
