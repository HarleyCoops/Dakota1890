# PrimeIntellect Runbook — Baguettotron Dakota RL

Goal: fine-tune `PleIAs/Baguettotron` on the Dakota grammar env with GRPO, using orchestrator + trainer + inference, TOPLOC on, 4×A100 for the trainer. This follows the real PrimeIntellect flow (provision → SSH → install → copy configs → launch), not a single local script.

## Prereqs (before renting)
- PrimeIntellect account with quota to rent GPUs; SSH key uploaded in the PI console.
- Tokens/keys ready: `PI_API_KEY`, optional `HF_TOKEN`, `WANDB_API_KEY`.
- Know the repo commit to run (e.g., latest `main` where Baguettotron configs live).

## Step 1: Rent/Start the instance in PrimeIntellect
1) Go to https://app.primeintellect.ai → Compute/Instances/Workbench → rent/provision GPU.  
2) Pick 4×A100 (or equivalent) + recent CUDA image.  
3) Wait for “running” state; note public IP/hostname.  
4) Ensure your SSH key is attached; grab the provided SSH command.

## Step 2: SSH in and bootstrap
```bash
ssh <user>@<pi-host>
sudo apt-get update && sudo apt-get install -y git wget
nvidia-smi  # verify GPUs
```

Install uv (if not preinstalled):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

Clone repos (keep Dakota repo alongside prime-rl):
```bash
cd ~
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
git clone https://github.com/HarleyCoops/Dakota1890.git
cd Dakota1890
git pull  # ensure latest
```

Install Python deps (inside Dakota repo for env package):
```bash
uv pip install -e environments/dakota_grammar_translation
uv pip install git+https://github.com/PrimeIntellect-ai/verifiers.git
```

Install prime-rl deps (from prime-rl repo):
```bash
cd ~/prime-rl
uv sync  # pulls torch/triton/vLLM and prime-rl extras
```

## Step 3: Place configs/data
- Configs: `~/Dakota1890/Baguettotron-Dakota1890/configs/*.toml`
- Datasets: `~/Dakota1890/dakota_rl_training/datasets/*.jsonl`
- Confirm paths in the configs match these locations (they reference `../dakota_rl_training/...`).

## Step 4: Set environment variables
```bash
export PI_API_KEY=<your_pi_key>
export HF_TOKEN=<optional_for_hf_pull>
export WANDB_API_KEY=<your_wandb_key>
export WANDB_PROJECT=dakota-rl-grammar
```

## Step 5: Launch (from prime-rl repo, GPU plan: 4×A100 trainer, CPU inference)
Recommended: inference on CPU (model is 321M) to leave all 4 GPUs for trainer.

Terminal A — inference (CPU):
```bash
cd ~/prime-rl
CUDA_VISIBLE_DEVICES= uv run inference @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_infer.toml
```

Terminal B — trainer (4 GPUs):
```bash
cd ~/prime-rl
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run trainer @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_train.toml
```

Terminal C — orchestrator:
```bash
cd ~/prime-rl
uv run orchestrator @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_orch.toml
```

One-shot combined (if you want orchestrator/trainer/inference under one `rl` command):
```bash
cd ~/prime-rl
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run rl \
  --trainer @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_train.toml \
  --orchestrator @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_orch.toml \
  --inference @ ~/Dakota1890/Baguettotron-Dakota1890/configs/baguettotron_infer.toml
```

## Knobs to watch
- **LoRA rank/alpha**: 16/32 for stability on 321M. If underfitting, try rank 32; if unstable, lower LR to 3e-6.  
- **Rollouts per example**: 4 in orchestrator; raise only if throughput allows.  
- **max_steps/ckpt**: start at 1k/200; increase after sanity checks.  
- **TOPLOC**: stay at `checkpoint_frequency=100` until stable.

## Validation checklist
- Orchestrator log: env `harleycooper/dakota1890` connected; TOPLOC on.  
- Trainer log: sees 4 CUDA devices; LoRA targets loaded; no OOM.  
- Inference sanity: single prompt uses `<|im_start|>...<|im_end|>` + `<think>` prelude.  
- W&B: both trainer/orchestrator streaming reward + char/affix metrics.

## Troubleshooting (PrimeIntellect reality)
- No “launch job” in UI: you must SSH and run the CLI above (see `dakota_rl_training/REAL_LAUNCH_METHOD.md`).  
- HF pull throttled: set `HF_TOKEN` and retry.  
- OOM: cut `per_device_train_batch_size` to 4 and/or reduce `seq_len` in `baguettotron_orch.toml`.  
- Missing datasets: `ls ~/Dakota1890/dakota_rl_training/datasets` and fix paths in configs.
