# Dakota1890 Pipeline Audit

**Date:** 2026-05-01
**Auditor:** Claude (read-only static audit; live execution was deliberately stopped)
**Repo state:** clone present at `C:\Users\chris\Dakota1890`, branch `main`, HEAD `38b05881`

This audit replaces the prior PIPELINE_AUDIT.md with a fresh end-to-end inspection of the repository as it sits today. No pipeline stages were actually executed during this audit — see "Execution status" at the bottom for why.

---

## TL;DR

- All seven pipeline stages exist in code and have artifacts on disk. None were executed during this audit; the assessment is purely static.
- The single largest blocker to a clean run is **environment**: the on-disk `.venv/` was a Linux Python 3.10 venv (`home = /usr/bin`) on a Windows host, so every entry point shebang was broken. It was renamed to `.venv_linux_broken/` and a fresh `.venv_win/` was created on `C:\Python312\python.exe`. Core deps (`anthropic`, `google-generativeai`, `Pillow`, etc.) were installed; the heavy RL stack (`verifiers`, `prime-rl`, `tinker`, `peft`, `torch`) was NOT installed.
- The pipeline is **multi-command**, not single-entrypoint. PIPELINE.md and CLAUDE.md confirm this and list the canonical commands.
- Datasets are **already materialized** at every stage, so the RL environment, training, and eval stages can in principle run without re-running extraction or SFT generation.
- Several scripts hard-fail without paid API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TINKER_API_KEY`). A `.env` exists in-repo (4361 bytes) but its contents were not inspected.

---

## Stage 0 — Environment

**What worked**

- Repo already cloned at `C:\Users\chris\Dakota1890` (no clone needed).
- README.md (46 KB), CLAUDE.md (15 KB), PIPELINE.md, SETUP.md, REPO_MAP.md, MODEL_CARD.md, GRANT_TECHNICAL_SUMMARY.md, VALIDATION_REPORT.md all present and informative.
- Public-domain source PDF (`grammardictionar00riggrich.pdf`, 30 MB) and 440 JP2 page images present under `Dictionary/grammardictionar00riggrich_jp2/`.

**What broke**

- `.venv/pyvenv.cfg` had `home = /usr/bin` and `version = 3.10.12`. `.venv/bin/python` was a broken symlink (`python -> python3`). Every Windows launch attempt errored with `No Python at '"/usr/bin\python.exe'`. The venv was created in WSL/Linux and does not function on the Windows host.
- `Qwen3-30B-ThinkingMachines-Dakota1890` submodule shows "modified content, untracked content" in `git status`.

**What's needed to fix**

- Recreate the venv on the Windows host (done: `.venv_win/` from `C:\Python312\python.exe`), or run on the WSL/Linux side where the original venv lives.
- Install `requirements.txt`, `requirements_hf_inference.txt`, then `pip install -e environments/dakota_grammar_translation`. The RL stack additionally needs `pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git` and `pip install git+https://github.com/PrimeIntellect-ai/verifiers.git` per CLAUDE.md.

---

## Stage 1 — Source Acquisition

**Status:** Complete on disk, no action required.

- `grammardictionar00riggrich.pdf` (30,232,737 bytes) at repo root.
- `Dictionary/grammardictionar00riggrich_jp2/grammardictionar00riggrich_*.jp2` — **440 JP2 page images** present.
- Public-domain status discussed in prior audit (1890 Riggs work, US PD).

---

## Stage 2 — OCR / VLM Extraction

**Entry points (all real, all surfaced via `--help`):**

- `scripts/extraction/convert_all_images.py` — JP2 -> JPEG conversion via `dakota_extraction.tools.image_converter`.
- `scripts/extraction/extract_grammar_pages.py` — grammar pages 1–88. Args: `--pages`, `--test`, `--yes`. Default test mode hits page 10 only.
- `scripts/extraction/extract_dakota_dictionary_v2.py` — dictionary pages.
- `python -m dakota_extraction.run_extraction` — canonical entry, args: `--input`, `--processed`, `--extracted`, `--datasets`, `--start-page`, `--end-page`, `--thinking-budget`, `--skip-conversion`, `--skip-extraction`, `--only-datasets`. Module imports cleanly.

**Materialized output:**

- `data/extracted/*.json` — **239 page JSON files** (page_095.json onward).
- `data/grammar_extracted/`, `data/processed_images/`, `data/reasoning_traces*/` — many timestamped / smoketest variants present (8+ `*_smoketest*` dirs from prior Qwen3-VL experiments).

**What would break a fresh run**

- Both grammar and dictionary extractors call the Anthropic API (`ANTHROPIC_API_KEY` required). At ~$0.25/page (per SETUP.md) a full grammar extraction ≈ $20, full dictionary extraction is more.
- `dakota_extraction/tinker_qwen3vl/` is the Qwen3-VL alternative path; CLAUDE.md / prior audit call it "not the maintained path."
- `extract_grammar_pages.py` exits nonzero on any per-page failure (per SETUP.md).

**Live verification:** not run. `--test` mode would burn a real API call.

---

## Stage 3 — SFT Dataset Generation (secondary path)

**Entry points:**

- `scripts/conversion/generate_synthetic_dakota.py` — Gemini-driven synthetic Q/A from `data/extracted/*.json` -> `data/bilingual_training_set.jsonl`. Requires `GOOGLE_API_KEY`. Note: `google-generativeai` package is **deprecated** (FutureWarning emitted during import); the script still imports it.
- `scripts/conversion/convert_extracted_to_chat.py` — JSONL -> OpenAI chat format -> `OpenAIFineTune/dakota_train.jsonl` and `dakota_valid.jsonl`. UTF-8 BOM tolerant per audit notes.
- `scripts/rl/dakota_openai_finetune.py` — readiness check (`--check-only`) and live OpenAI fine-tune submission. Imports `openai`, `tiktoken` (optional), `huggingface_hub` (optional), `dotenv`. SETUP.md notes it defaults to `gpt-4.1-mini-2025-04-14`.

**Materialized output:**

- `data/bilingual_training_set.jsonl` exists.
- `OpenAIFineTune/dakota_train.jsonl` (980), `dakota_valid.jsonl` (245).

**What would break a fresh run**

- `generate_synthetic_dakota.py`: needs `GOOGLE_API_KEY`; deprecated SDK may stop working at any time.
- `dakota_openai_finetune.py` (live): needs `OPENAI_API_KEY` with billing enabled. `--check-only` should run offline if `tiktoken` is installed (it isn't in the fresh `.venv_win`).

---

## Stage 4 — RL Task Generation + Environment

**RL rule pipeline:**

- `scripts/rl/organize_grammar_for_rl.py` — produces `data/rl_training_rules/`. On disk:
  - `rules_particles.json` — 1,176 lines
  - `rules_phonology.json` — 2,269 lines
  - `rules_syntax.json` — 7,222 lines
  - `rules_translation.json` — 10,158 lines
- `scripts/conversion/convert_rules_to_primeintellect.py` — fans rules into curriculum-bucketed JSONL.

**Materialized RL tasks:**

- `dakota_rl_training/datasets/grammar_tasks_complete.jsonl` — **10,576** tasks.
- `grammar_tasks_easy.jsonl` (1,973), `grammar_tasks_medium.jsonl` (5,294), `grammar_tasks_hard.jsonl` (1,172).
- Sample row schema: `{prompt, answer, info: {task_type, rule_id, rule_type, pattern, difficulty, source_pages, confidence}}` — confirmed by reading first row of `grammar_tasks_complete.jsonl`.

**Packaged environment:** `environments/dakota_grammar_translation/`

- `pyproject.toml`: `dakota1890 v0.1.17`, requires `verifiers>=0.1.7.post0`, `datasets>=2.18`, Python ≥3.10. Dual package layout (`dakota_grammar_translation/` + legacy `dakota1890/`).
- `dakota_grammar_translation/environment.py` defines a `SingleTurnEnv`-based env, `DEFAULT_SYSTEM_PROMPT`, `_char_f1`, dataset auto-discovery (package data -> repo data -> GitHub URL fallback at `https://raw.githubusercontent.com/HarleyCoops/Dakota1890/main/dakota_rl_training/datasets/grammar_tasks_complete.jsonl`).
- `dakota_grammar_translation.egg-info/` and `dakota1890.egg-info/` exist -> package was previously installed editable into the broken Linux venv.

**What would break a fresh run**

- `pip install -e environments/dakota_grammar_translation` will pull `verifiers` (large; PrimeIntellect git package). Not installed in `.venv_win`.
- `from dakota_grammar_translation import load_environment` cannot be tested without `verifiers`.

---

## Stage 5 — RL Training

**Local (PrimeIntellect) entry point:** `dakota_rl_training/train.py`

- Wraps `prime_rl.rl.RLConfig`, `prime_rl.rl.rl`, `prime_rl.utils.pydantic_config.parse_argv`. Try/except sets `PRIME_RL_AVAILABLE = False` if missing — script will *not* hard-crash on import but the actual training call needs `prime_rl`.
- `sys.path` munging removes script dir from path so the legacy local `verifiers/` folder doesn't shadow the pip-installed one — clever but brittle.
- Documented invocation: `python dakota_rl_training/train.py --check-only`.

**Remote (Tinker) entry point:** `dakota_rl_training/tinker_train.py`

- `tinker_integration/` package: `env.py`, `dataset.py`, `types.py`, `publish.py`.
- Requires `TINKER_API_KEY`. Documented invocation against `Qwen/Qwen3-30B-A3B-Instruct-2507`.

**Adjacent training assets (lots of them, somewhat noisy):**

- `dakota_rl_training/` contains **30+ Markdown launch/debug logs** (`LAUNCH_*.md`, `FIX_*.md`, `DEBUG_*.md`, `SSH_*.md`). Reads as a captain's log of past runs rather than maintained docs.
- `launch_*.ps1`, `launch_*.sh`, `launch_with_ledger.py`, `launch_primeintellect.py`, `validate_setup.sh` — multiple parallel launch surfaces.
- `verifiers/` (legacy local fork) and `tinker_integration/` (Tinker-specific) coexist.
- `outputs/` directory present (not enumerated; likely contains prior run logs and `metrics.jsonl` per CLAUDE.md).

**What would break a fresh run**

- `prime_rl` and `verifiers` not installed in `.venv_win`. `train.py --check-only` would import-fall-through but the actual training path requires `prime_rl`.
- `tinker_train.py` requires the `tinker` package + valid `TINKER_API_KEY`.
- A real local training run requires GPU + vLLM, which Windows-on-ARM (this host: `MINGW64_NT-10.0-26200-ARM64`) is unlikely to support.

---

## Stage 6 — Inference / Deployment

**Entry points:**

- `run_inference.py` — local inference shim (1908 bytes).
- `hf_inference_standalone.py` (15 KB) — HF Inference Providers path.
- `huggingface_space/` — Spaces deployment.
- `Qwen3-30B-ThinkingMachines-Dakota1890/` — git submodule (HF repo of the published adapter; `git status` reports it has uncommitted modifications).

**Published model card** (per CLAUDE.md / MODEL_CARD.md):

- `HarleyCooper/Qwen3-30B-ThinkingMachines-Dakota1890`, base `Qwen/Qwen3-30B-A3B-Instruct-2507`, LoRA rank 32, context 8192.

**What would break a fresh run**

- `run_inference.py` needs `transformers` + `peft` + a compatible `huggingface-hub` (requirements pin `<1.0`). None installed in `.venv_win`.
- HF inference needs `HF_TOKEN` with Inference Providers permission.

---

## Stage 7 — Evaluation / Benchmarks

**Entry points:**

- `eval/run_eval.py` — small-subset extraction eval. Args: `--pred`, `--truth`, `--out`. Computes token accuracy, char distance, diacritic preservation; writes Markdown report. Imports `score_extraction` (sibling module).
- `eval/score_extraction.py` — scoring primitives.
- `eval/benchmarks/`, `eval/fixtures/`, `eval/results/` directories present.
- `eval/report.md`, `eval/README.md` present.

**Test suite (`tests/`):**

- `test_inference_configuration.py`
- `test_offline_eval.py`
- `test_openai_finetune_readiness.py`
- `test_sft_baseline.py`
- `test_training_dataset_builder.py`
- `test_verifier_integration.py`

`pytest.ini` and `conftest.py` are configured. SETUP.md says `python -m pytest -q` is the smoke check.

**Verifiers-native eval:** CLAUDE.md mentions `vf-eval dakota1890 -n 10` once `verifiers` is installed.

**Reward ledger / W&B analysis:**

- `wandb_analysis/`, `wandb_visualizations/`, `wandb/` (raw W&B run dirs).
- `scripts/create_tinker_visualizations.py`, `scripts/analyze_wandb_run.py`, `scripts/export_ledger_now.py`.

**What would break a fresh run**

- `pytest` itself is installable, but `test_verifier_integration.py` requires `verifiers`; `test_sft_baseline.py` and `test_openai_finetune_readiness.py` likely need `openai` + `tiktoken`.
- `eval/run_eval.py` needs prediction + truth pair files and `score_extraction` deps; not yet exercised here.

---

## Cross-Cutting Concerns

### Repo cleanliness

- 30+ Markdown launch/debug logs in `dakota_rl_training/` are stale-feeling; the previous PIPELINE_AUDIT.md acknowledged multiple parallel launch surfaces. CLEANUP_PLAN.md (3.2 KB) and RERUN_PLAN.md (empty file, 0 bytes) suggest cleanup is an in-flight intent.
- `data/` has 8+ `extracted_qwen3vl_tinker_smoketest*/` directories from earlier experiments — likely safe to archive.
- `archive/` directory exists for older artifacts.
- Root has `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` checked-in-ish (probably gitignored but locally persisted).

### Hidden coupling / footguns

- `dakota_rl_training/train.py` aggressively reorders `sys.path` to push site-packages ahead of the script dir — required because there is a sibling `dakota_rl_training/verifiers/` folder that name-clashes with the `verifiers` PyPI package. Any tool that imports `dakota_rl_training.verifiers` directly will likely break.
- `environments/dakota_grammar_translation/` ships **two package names** in one wheel (`dakota_grammar_translation` and `dakota1890`). Both have their own `.egg-info`. This dual-name layout is a known maintenance burden.
- The packaged environment falls back to a **GitHub raw URL** if the dataset isn't found locally — works in CI but means an offline run on a clean checkout could silently pull a different dataset version than the local one.
- `dakota_extraction/tinker_qwen3vl/` exists but is documented as "not maintained" — it is not gated behind an opt-in flag, so a casual contributor could wire up the wrong path.

### API / billing risks

- Any "live" smoke (extraction, synthetic QA, OpenAI SFT submit, Tinker train) costs real money. Only `--check-only` and `--test` flags are safe.
- `google-generativeai` is deprecated (FutureWarning observed during import). Synthetic QA path is on borrowed time.

### Submodule drift

- `Qwen3-30B-ThinkingMachines-Dakota1890` submodule is dirty. Either commit/push the submodule changes or `git submodule update --init --remote` to reset.

---

## What it would take to run end-to-end on this host

Roughly, in order:

1. **Recreate venv on Windows** (done partially): `C:\Python312\python.exe -m venv .venv_win`, then:
   ```
   .venv_win\Scripts\pip install -r requirements.txt
   .venv_win\Scripts\pip install -r requirements_hf_inference.txt
   .venv_win\Scripts\pip install -e environments/dakota_grammar_translation
   .venv_win\Scripts\pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git
   .venv_win\Scripts\pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
   .venv_win\Scripts\pip install pytest tiktoken openai peft transformers torch
   ```
2. **Confirm `.env`** has the keys you actually intend to use (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TINKER_API_KEY`, `HF_TOKEN`, `WANDB_API_KEY`, optionally `GOOGLE_API_KEY`).
3. **Smoke without spending money**:
   - `python -m pytest -q`
   - `python dakota_rl_training/train.py --check-only`
   - `python scripts/rl/dakota_openai_finetune.py --check-only`
   - `python -c "from dakota_grammar_translation import load_environment; e = load_environment(); print(len(e.dataset))"`
4. **Live extraction smoke** (paid): `python scripts/extraction/extract_grammar_pages.py --test`
5. **RL train check**: `python dakota_rl_training/train.py --model Qwen/Qwen3-0.6B --max-steps 5` (needs GPU; on Windows-ARM this almost certainly won't work — use a Linux GPU host).
6. **Eval**: `python eval/run_eval.py --pred ... --truth ... --out eval/report.md` once you have prediction/truth pairs; or `vf-eval dakota1890 -n 10` for verifiers-native eval.

---

## Execution status

Per user instruction mid-audit ("Stop trying to install or run anything. Just report what you've learned"), no pipeline stage was actually executed. Findings are based on:

- Direct reading of README.md, CLAUDE.md, PIPELINE.md, SETUP.md, prior PIPELINE_AUDIT.md, and selected source files (`train.py`, `environment.py`, `pyproject.toml`, `run_eval.py`, `dakota_openai_finetune.py`).
- File-system enumeration of `dakota_extraction/`, `scripts/extraction/`, `scripts/conversion/`, `scripts/rl/`, `dakota_rl_training/`, `environments/`, `data/`, `eval/`, `tests/`.
- `--help` output for `extract_grammar_pages.py` and `dakota_extraction.run_extraction` (these confirmed those entry points import cleanly with just core deps).
- Line counts on all major JSONL/JSON datasets to verify materialized artifact sizes match the claims in PIPELINE.md (organized rules in 4 files totaling ~20K lines pre-bucketing; 10,576 RL tasks confirmed; 980/245 OpenAI splits confirmed).

Live execution of any stage would require completing the env setup in §"What it would take" above, which the user explicitly de-scoped.
