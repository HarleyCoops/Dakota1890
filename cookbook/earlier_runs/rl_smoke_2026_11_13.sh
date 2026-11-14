#!/bin/bash
# Archived: the exact job script of the 13 November 2026 RL smoke
# (job HarleyCooper/6af7b68523985fb9cd80f25d was its third attempt; the first
# two died on a bash quoting bug and on W&B auth, see the README's Stage 2
# section). Kept as the provenance record of the smoke's numbers; the live
# launcher is cookbook/launch_full_rl_pass.py.
#
# RL smoke: 2 steps of lexgym GRPO on Tinker, Qwen3.5-4B from the SFT checkpoint,
# with the new 800-token answer limit. Mirrors cookbook/earlier_runs/launch_rl_job.py
# (thinking off, no task filter, smoke-sized), pinned to commit c9a068ab08a9291149c74962d92add3b1bb36c8d.
set -e
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q tinker==0.39.0 tinker-cookbook==0.6.1 wandb python-dotenv huggingface_hub datasets
# These HF env secrets don't include a valid wandb key, so never init W&B:
# tinker_cookbook errors out at logging setup if wandb_project is set and auth fails.
export WANDB_MODE=offline
mkdir repo && cd repo && git init -q
GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 https://github.com/HarleyCoops/Dakota1890 c9a068ab08a9291149c74962d92add3b1bb36c8d
GIT_LFS_SKIP_SMUDGE=1 git checkout -q FETCH_HEAD
python -c 'from lexgym.recall_curve import fetch_tasks; fetch_tasks("HarleyCooper/riggs1890-lexicon-graph", "data/lexgym_tasks")'
# Upload the run folder when the job ends for any reason, so checkpoint paths and traces are kept.
upload_run() {
  python -c 'from pathlib import Path; from lexgym.recall_curve import upload; upload("HarleyCooper/riggs1890-lexgym-traces", Path("lexgym_runs/rl-smoke-2026-11-13-qwen35-4b"), "rl-smoke-2026-11-13/qwen-3.5-4b/rl-smoke-800tok")'
}
trap upload_run EXIT
python -m lexgym.train_rl \
  --model-name Qwen/Qwen3.5-4B \
  --load-checkpoint-path "tinker://536d3512-c611-5499-9dda-869a1d74ad09:train:0/weights/final" \
  --task-dir data/lexgym_tasks \
  --log-path lexgym_runs/rl-smoke-2026-11-13-qwen35-4b \
  --batch-size 32 \
  --group-size 8 \
  --learning-rate 2e-5 \
  --eval-every 100 \
  --save-every 100 \
  --eval-limit-per-task 5 \
  --max-tokens 800 \
  --stage rl-smoke-800tok \
  --wandb-name rl-smoke-2026-11-13-qwen35-4b \
  --max-steps 2
