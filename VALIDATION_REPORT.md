# Validation Report

Date: `2026-04-04`

This report captures both the pre-cleanup baseline issues discovered during the audit and the post-cleanup validation results from the current sandbox.

## Pre-Cleanup Baseline

- `python -m pytest -q`
  - failed because `pytest` was recursing into `.venv` and third-party packages
- `tests/test_verifier_integration.py`
  - failed collection because it imported the obsolete local `verifiers.grammar_env`
- `run_inference.py`
  - pointed at a `Qwen3-4B` base model while the saved adapter metadata referenced the `30B` base
- root docs and model card
  - contained stale counts (`1,036` rules / `5,657` tasks), OpenRouter references, and mismatched model lineage

## Post-Cleanup Results

| Stage | Command | Result | Notes |
| --- | --- | --- | --- |
| Environment build | `python -m pip install --dry-run -r requirements.txt` | PASS | dependency resolution succeeded with the new `huggingface-hub<1.0` pin |
| HF inference deps | `python -m pip install --dry-run -r requirements_hf_inference.txt` | PASS | dry-run resolved `huggingface_hub-0.36.2`; `peft` now listed explicitly |
| Repo harness | `python -m pytest -q` | PASS | `9` tests passed |
| Packaged environment | inline `load_environment(max_examples=3, eval_fraction=0)` | PASS | loaded `3` train examples |
| Dataset integrity | inline count script | PASS | `1497` rules, `10576` tasks, `980/245` OpenAI splits |
| SFT baseline conversion | `python scripts/conversion/convert_extracted_to_chat.py --input-file data/step0_sft_smoke.jsonl --output-dir OpenAIFineTune/step0_smoke` | PASS | wrote `4` train and `1` validation example |
| OpenAI readiness | `python scripts/rl/dakota_openai_finetune.py --check-only` | PASS | files found; `OPENAI_API_KEY` present |
| OpenAI fine-tune API smoke | inline `OpenAI().fine_tuning.jobs.list(limit=1)` | PASS | fine-tuning endpoint reachable without submitting a paid job |
| RL launch readiness | `python dakota_rl_training/train.py --check-only` | PASS | local prerequisite check now succeeds from repo checkout |
| Reward ledger smoke | inline `env.rubric.score(...)` | PASS | returned `0.84` with a populated ledger |
| Anthropic grammar smoke | `python scripts/extraction/extract_grammar_pages.py --test` | FAIL | Anthropic API returned low-credit error |
| Anthropic dictionary smoke | `python scripts/extraction/extract_dakota_dictionary_v2.py --test` | FAIL | Anthropic API returned low-credit error |
| Gemini smoke | inline two-entry `DakotaQAGenerator.generate_qa_pairs(...)` | FAIL | Google API key is expired |
| Local inference smoke | `python run_inference.py --prompt "Translate 'my elder brother' to Dakota."` | FAIL | current sandbox has incompatible `huggingface-hub==1.2.4` installed |
| HF remote inference smoke | `python hf_inference_standalone.py --prompt "Translate 'my elder brother' to Dakota." --json` and direct `InferenceClient.chat_completion(...)` probe | FAIL | local runtime still hits the HF package skew; direct chat call returned `403` insufficient token permissions |
| Model-output round-trip | blocked by inference failures | FAIL | could not score model-generated output because neither local nor HF inference completed |

## Exact Failure Text

### Anthropic grammar smoke

`Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}}`

### Anthropic dictionary smoke

`anthropic.BadRequestError: Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits.'}}`

### Gemini smoke

`400 API key expired. Please renew the API key. [reason: "API_KEY_INVALID"]`

### Local inference smoke

`ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, but found huggingface-hub==1.2.4.`

### HF chat-completions probe

`403 Forbidden: This authentication method does not have sufficient permissions to call Inference Providers on behalf of user HarleyCooper.`

## Supplemental Outputs

- Reward-ledger smoke:
  - `reward = 0.84`
  - ledger keys present: `exact_match_raw`, `char_overlap_raw`, `pattern_raw`, `affix_raw`, `difficulty_multiplier`, `reward_scalar`
- Difficulty distribution in the RL task set:
  - `easy`: `1,973`
  - `medium`: `5,294`
  - `hard`: `1,172`
  - `advanced`: `2,137`

## Readiness Summary

- Offline Dakota pipeline surfaces: `GREEN`
- Live extraction / synthetic API surfaces: `RED` because the current Anthropic and Gemini credentials are not usable
- Local inference surface: `RED` until the installed HF packages match the pinned requirements
- Remote HF inference surface: `RED` until the token has the required provider permissions
