# PrimeIntellect Runbook — Baguettotron Dakota RL

Goal: fine-tune `PleIAs/Baguettotron` on the Dakota grammar env with GRPO, using orchestrator + trainer + inference, TOPLOC on, 4×A100 for the trainer.

## Prereqs
- `PI_API_KEY` exported (or in `.env`), network egress allowed to Hugging Face.
- `uv` installed; `prime-rl`, `verifiers`, and the Dakota env installed (see `dakota_rl_training/README.md`).
- Data: `dakota_rl_training/datasets/*.jsonl` present on the box.
- Optional: `HF_TOKEN` if HF throttles pulls.

## GPU layout options
- Preferred: keep all 4 A100s for the trainer; run inference on CPU (model is 321M) to avoid contention.  
- Alternative: reserve 1 GPU for inference and give the trainer the other 3 GPUs (set `CUDA_VISIBLE_DEVICES` per process).

## Launch (component mode)
Run from repo root.

Terminal 1 — inference (CPU):
```bash
cd Baguettotron-Dakota1890
CUDA_VISIBLE_DEVICES= uv run inference @ configs/baguettotron_infer.toml
```

Terminal 2 — trainer (4 GPUs):
```bash
cd Baguettotron-Dakota1890
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run trainer @ configs/baguettotron_train.toml
```

Terminal 3 — orchestrator:
```bash
cd Baguettotron-Dakota1890
uv run orchestrator @ configs/baguettotron_orch.toml
```

One-shot combined launch (uses the same configs):
```bash
cd Baguettotron-Dakota1890
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run rl \
  --trainer @ configs/baguettotron_train.toml \
  --orchestrator @ configs/baguettotron_orch.toml \
  --inference @ configs/baguettotron_infer.toml
```

## Knobs to watch
- **LoRA rank/alpha**: tuned down (16/32). If underfitting, raise rank to 32. If instability, lower LR to 3e-6.
- **Rollouts per example** (`rollouts_per_example`): defaults to 4; raise only if you have headroom.
- **max_steps / ckpt.interval**: start small (<=1k) to sanity check; increase after stability is confirmed.
- **TOPLOC**: stays enabled; keep `checkpoint_frequency` conservative (100) to catch regressions early.

## Validation checklist
- Orchestrator log shows env connected (`harleycooper/dakota1890`) and TOPLOC on.
- Trainer sees 4 devices and LoRA modules loaded.
- Inference answers use the `<|im_start|>...<|im_end|>` template with `<think>` prelude (sanity-check with a single request).
- W&B streams both trainer and orchestrator runs; metrics include reward and character/affix accuracies.

