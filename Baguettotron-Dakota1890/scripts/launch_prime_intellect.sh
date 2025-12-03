#!/usr/bin/env bash
# Launch Baguettotron Dakota RL on PrimeIntellect (orchestrator + trainer + inference)
# Assumes uv + prime-rl stack installed and PI_API_KEY exported.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Launching Baguettotron Dakota RL with GRPO"
echo "Base model: PleIAs/Baguettotron"
echo "Trainer GPUs: 4 (set via CUDA_VISIBLE_DEVICES)"
echo "TOPLOC: enabled"
echo

if [[ -z "${PI_API_KEY:-}" ]]; then
  echo "[ERROR] PI_API_KEY is not set. Export it before launching."
  exit 1
fi

echo "[INFO] Using configs from $ROOT_DIR/configs"

echo
echo "[STEP 1] Start inference (CPU to leave GPUs for trainer)."
echo "  CUDA_VISIBLE_DEVICES= uv run inference @ configs/baguettotron_infer.toml &"

echo
echo "[STEP 2] Start trainer on 4 GPUs."
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run trainer @ configs/baguettotron_train.toml &"

echo
echo "[STEP 3] Start orchestrator (connects to env + TOPLOC)."
echo "  uv run orchestrator @ configs/baguettotron_orch.toml"

echo
echo "Tip: if you need a one-liner instead, run:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run rl \\"
echo "    --trainer @ configs/baguettotron_train.toml \\"
echo "    --orchestrator @ configs/baguettotron_orch.toml \\"
echo "    --inference @ configs/baguettotron_infer.toml"

