# Cleanup Plan

This is the implemented step-0 cleanup ledger. The plan is conservative by design: remove dead ends from the active surface, preserve the story, and avoid rewriting historical research artifacts unless they block the Dakota path.

## Implemented

### Canonical Dakota surfaces

- Kept the Dakota core path centered on:
  - `dakota_extraction/`
  - `data/rl_training_rules/`
  - `dakota_rl_training/datasets/`
  - `environments/dakota_grammar_translation/`
  - `dakota_rl_training/`
  - `run_inference.py`
  - `hf_inference_standalone.py`
- Updated canonical docs and model references to the current observed counts:
  - `1,497` rules
  - `10,576` RL tasks
  - `980 / 245` OpenAI baseline splits
- Aligned the published adapter lineage to `Qwen/Qwen3-30B-A3B-Instruct-2507`

### Harness hardening

- Restricted `pytest` to `tests/`
- Excluded `.venv`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, and archived directories from recursion
- Replaced the legacy verifier integration script with packaged-environment integration tests
- Added:
  - `tests/test_sft_baseline.py`
  - `tests/test_inference_configuration.py`

### Script cleanup

- Made `run_inference.py` a real CLI with lazy imports and aligned constants
- Added `--check-only` readiness behavior to the OpenAI baseline path
- Fixed `extract_grammar_pages.py` so it runs from the repo root and exits nonzero on extraction failure
- Fixed `dakota_rl_training/train.py --check-only` so it can import the packaged environment from the repo checkout
- Made `convert_extracted_to_chat.py` tolerant of UTF-8 BOM input on Windows

### Archival moves

Moved into `archive/step0_legacy/`:

- stale setup and launch guides
- duplicate extraction scripts
- manual and obsolete tests
- tracked root-level debug/error logs
- duplicate `tmp_publish/` bundle

## Preserved Intentionally

- `README.md`
- `Public/`, `media/`, `wandb*`, model cards, and visual assets
- `Qwen3-30B-ThinkingMachines-Dakota1890/`
- reference projects:
  - `Akkadian/`
  - `RailroadEngineer1959/`
  - `Qwen3-RailroadEngineer1959-RL/`
  - `Baguettotron-Dakota1890/`

## Deferred

These remain in place for now and should be discussed before any destructive move:

- `downloaded_model_step_400/`
- `model_step_400.tar.gz`
- `benchmark_results.jsonl`
- `test_model_inference.py`
- `test_space_inference_local.py`
- older `docs/status/*` historical records that still quote early-run counts

## Rationale

- Archive, do not erase, when a file captures an earlier research branch that may still matter pedagogically.
- Patch only the canonical Dakota path in step 0.
- Leave broad refactors and single-entrypoint orchestration for a later step once the maintainer has reloaded project context.
