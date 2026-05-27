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

The RL verifier now preserves reward metadata from the packaged dataset's nested `info` object. The packaged dataset stores `difficulty`, `task_type`, `rule_id`, and pattern data under `entry["info"]`; the environment had been reading top-level fields such as `entry["verification_pattern"]` and silently dropping `info.pattern`. That was the root cause for the dead `pattern_reward` channel in the old public runs.

The verifier also treats bracketed grammar placeholders as literal patterns when regex matching does not match. The previous Tinker runs predate the metadata fix; their metrics show `env/all/ledger/pattern_raw = 0.0`. Do not use those runs as evidence that the pattern reward component worked.

Regression coverage:

```bash
python -m pytest tests/test_verifier_integration.py -q
```

Reward-channel smoke:

```bash
python scripts/rl/check_reward_channels.py
```

Expected result before any paid rerun:

- `checked` is greater than zero.
- `gold_pattern_nonzero` is greater than zero; current local result is `5` out of `5`.
- The verbose probe keeps `pattern_raw = 1.0` while `exact_match_raw = 0.0`, which confirms exact match is intentionally strict and will not reward explanatory completions.

## Exact-Match Diagnosis

The public 0.6B orchestrator run `29hn8w98` has 1,000 rows where both `metrics/exact_match_reward` and `metrics/pattern_reward` are exactly `0.0`. The same run still learned through other channels: `metrics/char_overlap_reward` rose from `0.0384` to `0.5349`, `metrics/affix_reward` rose from `0.9531` to `0.9792`, and `reward/mean` rose from `0.1202` to `0.3486`.

Treat `pattern_reward = 0` as a repaired plumbing bug. Treat `exact_match_reward = 0` as an experiment-design issue:

- local Tinker ledgers have nonzero `env/all/ledger/exact_match_raw`, so the exact-match reward function is not universally dead;
- the dataset median answer length is about 4 words;
- the old public run used long completions, with `completion_len/mean` starting near `445` tokens and a max of `512`;
- strict normalized equality only fires for answer-only completions, not explanations that contain the right answer.

## Runtime Readiness

Treat `requirements.txt` as unreliable for launch decisions. It is only a convenience install list, not a lock file and not proof that the active Python runtime can talk to Tinker, Gemini, or W&B. Before spending GPU/API budget, verify the live environment directly:

```bash
python scripts/rl/check_runtime_readiness.py \
  --tinker-model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --gemini-model gemini-3.5-flash \
  --gemini-generate-smoke \
  --check-wandb-api
```

Current known-good RL runtime target:

- `tinker==0.22.1`
- `tinker-cookbook==0.4.1`
- `protobuf>=6.31.1,<7`
- `google-genai>=2.6.0`

Do not reinstall the full root requirements immediately before a paid Tinker run unless the readiness check is repeated afterwards. Current Tinker requires protobuf 6.x; the deprecated `google-generativeai` package path is not compatible with that assumption and should not be used by active extraction or Q/A scripts.

For dictionary Q/A generation, use Gemini 3.x thinking deliberately. The runtime smoke uses a minimal thinking level to keep JSON output cheap and deterministic, but the full Q/A generation pass should use `--thinking-level medium` with `gemini-3.5-flash` unless a smaller pilot shows worse JSON compliance or unacceptable cost:

```bash
python scripts/conversion/generate_synthetic_dakota.py \
  --model gemini-3.5-flash \
  --thinking-level medium \
  --max-output-tokens 4096
```

## Tinker Model Choice

For the small reward-channel pilot, keep `Qwen/Qwen3-30B-A3B-Instruct-2507` if the goal is direct comparison with the old Dakota run and the pilot is launched before its retirement window closes. For a longer-lived full rerun, test a non-retiring successor first. The current best medium-cost candidate is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`: it has a 64K context window and only slightly higher training price than the retiring Qwen 30B path. Treat `Qwen/Qwen3.6-35B-A3B` as a more expensive exploratory option, not the default full-run target.

## Rerun Gates

Do not launch the full GPU run until these local gates pass:

```bash
python scripts/rl/check_runtime_readiness.py --gemini-generate-smoke --check-wandb-api
python -m pytest -q
python scripts/rl/check_reward_channels.py
```

Before a full mixed-task rerun, run a small exact-sensitive pilot. Keep it limited to tasks where exact match is meaningful and force answer-only behavior:

```bash
python dakota_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --wandb-project dakota-rl-grammar \
  --wandb-name qwen3-30b-dakota-reward-channel-pilot \
  --log-path dakota_rl_training/outputs/tinker_qwen30b_reward_channel_pilot \
  --task-filter identify_pattern word_translation reverse_translation \
  --max-examples 512 \
  --batch-size 16 \
  --group-size 8 \
  --max-tokens 64 \
  --temperature 0.3 \
  --learning-rate 4e-5 \
  --lora-rank 32 \
  --eval-every 10 \
  --save-every 50 \
  --ledger-csv wandb_analysis/reward_ledger_tinker_reward_channel_pilot.csv \
  --sync-metrics-to-wandb
```

Immediately audit the local metrics file before trusting the W&B surface:

```bash
python scripts/rl/audit_tinker_metrics.py \
  --metrics dakota_rl_training/outputs/tinker_qwen30b_reward_channel_pilot/metrics.jsonl \
  --require-nonzero pattern_raw exact_match_raw
```

The audit requires the full composite reward ledger: component weights, raw component scores, normalized component scores, weighted contributions, length multiplier, difficulty multiplier, reconstructed composites, final `reward_scalar`, `composite_diff`, and `parse_success`.

Proceed to the full rerun only if the pilot logs nonzero pattern values under the Tinker ledger namespace and nonzero exact-match values on exact-sensitive task families. If exact remains zero, tighten the system prompt further, lower `max_tokens`, or gate exact-match weight by task type before scaling. If local metrics pass but W&B is missing `env/all/ledger/*`, rerun with `--sync-metrics-to-wandb` or replay `metrics.jsonl` before treating the run as publishable.

## Completed Reward-Channel Pilot

The first successful post-fix Tinker pilot completed on May 27, 2026:

- W&B run: `christian-cooper-us/dakota-rl-grammar/runs/d44bra91`
- W&B name: `qwen3-30b-dakota-reward-channel-pilot-20260527_133821`
- local output: `dakota_rl_training/outputs/tinker_qwen30b_reward_channel_pilot_20260527_133821`
- final state path: `tinker://e8838941-d80a-5225-b3a9-391a03e2dd37:train:0/weights/final`
- final sampler path: `tinker://e8838941-d80a-5225-b3a9-391a03e2dd37:train:0/sampler_weights/final`
- reward ledger CSV: `wandb_analysis/reward_ledger_tinker_reward_channel_pilot_20260527_133821.csv`

Audit result:

```bash
python scripts/rl/audit_tinker_metrics.py \
  --metrics dakota_rl_training/outputs/tinker_qwen30b_reward_channel_pilot_20260527_133821/metrics.jsonl \
  --require-nonzero pattern_raw exact_match_raw
```

The audit passed with 29 metric rows and 29 ledger rows. `pattern_raw` was nonzero in 153 scanned ledger points, and `exact_match_raw` was nonzero in 47 scanned ledger points. W&B API verification confirmed remote visibility for `env/all/ledger/pattern_raw`, `test/env/all/ledger/pattern_raw`, and `test/env/all/ledger/exact_match_raw`.

## W&B MCP Monitoring

W&B MCP is an agent-side analysis interface, not a replacement for the training logger. The Dakota full run still needs to log normally to W&B. For Tinker runs, include `--sync-metrics-to-wandb` when launching from this repo so local `metrics.jsonl` values, including `env/all/ledger/*`, are replayed into the matching W&B run after training. Without that sync, an MCP-connected agent may see the public run shell but miss the local reward-ledger channels that matter for the reward fix audit.

Recommended hosted MCP setup for a Codex/Hermes-style agent:

```bash
export WANDB_API_KEY="<wandb-api-key>"
codex mcp add wandb \
  --url https://mcp.withwandb.com/mcp \
  --bearer-token-env-var WANDB_API_KEY
```

Verification prompt after installation:

```text
List my W&B entities, then probe christian-cooper-us/dakota-rl-grammar and report the metric keys related to reward, exact_match, pattern, and ledger.
```

Hermes agent instruction:

```text
You are working in C:\Users\chris\Dakota1890. Install/configure the official W&B MCP server for your agent using the hosted endpoint https://mcp.withwandb.com/mcp and the existing WANDB_API_KEY environment variable. Do not print or persist the API key. Verify access by listing W&B entities and probing christian-cooper-us/dakota-rl-grammar. For the Dakota rerun, confirm that the launch command logs to --wandb-project dakota-rl-grammar, uses a unique --wandb-name, and includes --sync-metrics-to-wandb for dakota_rl_training/tinker_train.py so reward ledger metrics become queryable through W&B MCP. After the run appears, compare the new run against 29hn8w98 and report whether pattern and exact-match channels are nonzero.
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
  --max-tokens 128 \
  --temperature 0.5 \
  --learning-rate 4e-5 \
  --lora-rank 32 \
  --eval-every 20 \
  --save-every 20 \
  --sync-metrics-to-wandb
```

## Audit Fix Order

1. Rerun Dakota1890 RL after the verifier fix and archive metrics/checkpoint IDs.
2. Build `run_pipeline.py` as the single audited entry point with `--check-only`, `--from-existing-extraction`, `--skip-ocr`, and `--stage` options.
3. Create a small human-validated held-out Dakota test set with provenance and reviewer fields.
4. Use `scripts/check_windows_tooling.ps1` to keep Windows/PowerShell audit checks separate from WSL/Linux-first RL training.
5. Add language config YAML only after the Dakota path is stable.
6. Add community-in-the-loop review fields for register, temporal drift, and approval status.
