# Baguettotron-Dakota1890

PrimeIntellect runbook for fine-tuning **PleIAs/Baguettotron** on the Dakota RL gym with GRPO, using an orchestrator + trainer + inference setup. Targets 4×A100, keeps TOPLOC verification enabled, and mirrors the existing Dakota pipeline while swapping in the Baguettotron base.

## What changes vs. Qwen runs
- Base model: `PleIAs/Baguettotron` (321M, Llama-like, 4k context, instruction + `<think>` traces).
- LoRA scaled down for a small backbone (rank 16, alpha 32, dropout 0.05).
- Shorter sequence lengths and rollouts to match the 4k context window.
- Trainer configured for 4 GPUs; inference can run on CPU or a spare GPU.

## Files here
- `configs/baguettotron_train.toml` — trainer config (GRPO, LoRA, curriculum, TOPLOC).
- `configs/baguettotron_orch.toml` — orchestrator config (rollouts, env ID, W&B).
- `configs/baguettotron_infer.toml` — inference server config (vLLM-style, 4k context).
- `scripts/launch_prime_intellect.sh` — example launch script tying the three configs together.
- `RUNBOOK_PRIME_INTELLECT.md` — step-by-step checklist for PrimeIntellect.

## Quick start
1. Ensure `PI_API_KEY` is set and `uv` + `prime-rl` stack are installed on the PrimeIntellect box.
2. Start inference, trainer, orchestrator with the helper script (see runbook for GPU pinning).
3. Monitor W&B and TOPLOC checkpoints; adjust batch sizes if you observe instability.

