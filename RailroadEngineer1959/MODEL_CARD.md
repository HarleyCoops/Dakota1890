---
language: en
license: other
base_model: Qwen/Qwen3-4B-Instruct-2507
pipeline_tag: text-generation
tags:
  - lora
  - grpo
  - reinforcement-learning
  - tinker
  - qwen
  - qwen3
  - rail-safety
  - operations
  - historical-rules
  - deterministic-rubric
  - safety-critical
---

# Qwen3-4B-RailRoadEngineer1959 (Tinker RL LoRA)

A reinforcement learning LoRA that teaches Qwen3-4B-Instruct-2507 to reason through 1959 railroad operating rules. Trained end-to-end on deterministic rewards derived from the PrimeIntellect `railroad_1959` gym: no external LLMs in the loop, just rule-grounded scoring. This is a zero-to-one example of taking a historical ruleset → building a verifiable gym → producing a deployable policy on modest hardware.

## Why this matters (zero-to-one)
- **Rules to behavior**: Starts from a scanned rulebook, converts to structured tasks, and uses a rubric that encodes safety, procedure, and terminology. The model learns operational judgment, not just text mimicry.
- **Deterministic scoring**: Rewards come from exact-match + token-F1 checks (no Anthropic or other LLM dependency), so the signal is stable and repeatable.
- **Small model, real skill**: A 4B base acquires role-specific procedures (meets, inspections, forms, restricted speed) with a single Tinker run.
- **Generalizable pattern**: Swap the dataset and rubric, keep the loop—this is the same pipeline pattern used in Dakota1890, now applied to railroad safety.

## Training recipe
- **Base model**: `Qwen/Qwen3-4B-Instruct-2507`
- **Checkpoint (sampler)**: `tinker://e9732434-7315-5c9c-8de3-6e7d937b8ec5:train:0/sampler_weights/final`
- **Optimizer**: Tinker GRPO, LoRA rank 32
- **Tokens**: max 256, temperature 0.9 during training; inference defaults to 0.7
- **Dataset**: `RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json` (2,708 tasks)
- **Reward/Rubric**: Deterministic composite:
  - Safety (50%) = max(exact match, token-F1)
  - Procedure (30%) = token-F1
  - Terminology (20%) = token-F1
  - Parse success logged to ensure clean outputs
- **Batches**: 77 (batch_size=32, group_size=8); eval every 20 batches

## Files in this repo
- `railroad_inference.py` — local inference against the Tinker sampler checkpoint
- `RailroadEngineer1959/railroad_rl_training/tinker_train.py` — full Tinker training launcher
- `RailroadEngineer1959/outputs/tinker_railroad_run/` — metrics, checkpoints, HTML logs (local run)
- `RailroadEngineer1959/wandb_analysis/railroad_reward_ledger_tinker.csv` — exported reward ledger

## Usage (Tinker sampler checkpoint)
Set your Tinker key and run a task from the dataset:
```bash
export TINKER_API_KEY=...
python RailroadEngineer1959/railroad_inference.py --task-id 713A-001
```
Or pass a custom prompt:
```bash
python RailroadEngineer1959/railroad_inference.py --prompt "Explain Form E time order meet procedure."
```

## Hugging Face upload
The sampler checkpoint was unpacked from Tinker into:
```
tmp_publish/railroad_sampler/
  adapter_model.safetensors
  adapter_config.json
```
Upload with:
```bash
python scripts/conversion/upload_model_to_hf.py \
  --weights-dir tmp_publish/railroad_sampler \
  --repo-id HarleyCooper/Qwen3-4B-RailRoadEngineer1959
```
If you prefer the raw archive, use `scripts/conversion/upload_tinker_checkpoint.py` and point `--file` at `final_sampler.ckpt`.

## Expected behavior
- Correct side inspections during meets (e.g., brakeman to far side, conductor to near side).
- Safe defaulting: hold position, await authority, monitor for defects.
- Procedure-aware phrasing and consistent terminology from the 1959 code.

## Limitations and cautions
- Deterministic rubric emphasizes literal alignment; creative paraphrase may lower scores.
- No vision inputs; assumes text scenarios only.
- Domain-specific: not a general assistant; keep prompts constrained to railroad operations.
- Always keep a human in the loop for real-world safety decisions.

## Reproducing training
```bash
export TINKER_API_KEY=...
python RailroadEngineer1959/railroad_rl_training/tinker_train.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --log-path RailroadEngineer1959/outputs/tinker_railroad_run \
  --dataset-path RailroadEngineer1959/data/railroad_extracted/safety_tasks_complete.json \
  --batch-size 32 --group-size 8 --max-tokens 256 --temperature 0.9
```

## Acknowledgments
- Base model: Qwen team
- Infra: Thinking Machines Tinker
- Dataset/rubric: RailroadEngineer1959 project (PrimeIntellect gym converted to Tinker)
