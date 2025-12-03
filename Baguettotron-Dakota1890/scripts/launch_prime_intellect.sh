#!/usr/bin/env bash
# PrimeIntellect launch helper for Baguettotron Dakota RL.
# Use this ON the rented PrimeIntellect/Linux instance (after SSH, repo clone, deps install).

set -euo pipefail

DAKOTA_ROOT="${DAKOTA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PRIME_RL_ROOT="${PRIME_RL_ROOT:-$HOME/prime-rl}"
CONFIG_DIR="$DAKOTA_ROOT/configs"
DATA_DIR="$DAKOTA_ROOT/../dakota_rl_training/datasets"

echo "=== Baguettotron Dakota RL (PrimeIntellect) ==="
echo "Assumes: uv installed, prime-rl repo at $PRIME_RL_ROOT, datasets present, run on a GPU node."
echo

# Pre-flight checks
if [[ -z "${PI_API_KEY:-}" ]]; then
  echo "[ERROR] PI_API_KEY is not set. Export it before launching." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found. Install via https://astral.sh/uv/install.sh" >&2
  exit 1
fi
if [[ ! -d "$PRIME_RL_ROOT" ]]; then
  echo "[ERROR] prime-rl repo not found at $PRIME_RL_ROOT. Clone https://github.com/PrimeIntellect-ai/prime-rl.git" >&2
  exit 1
fi
for f in baguettotron_train.toml baguettotron_orch.toml baguettotron_infer.toml; do
  [[ -f "$CONFIG_DIR/$f" ]] || { echo "[ERROR] Missing $CONFIG_DIR/$f"; exit 1; }
done
for f in grammar_tasks_easy.jsonl grammar_tasks_medium.jsonl grammar_tasks_hard.jsonl; do
  [[ -f "$DATA_DIR/$f" ]] || { echo "[ERROR] Missing dataset $DATA_DIR/$f"; exit 1; }
done

echo "[OK] Pre-flight checks passed."
echo "  Config dir: $CONFIG_DIR"
echo "  Data dir:   $DATA_DIR"
echo "  prime-rl:   $PRIME_RL_ROOT"
echo
echo "Recommended layout: keep inference on CPU, trainer on 4×A100. If you prefer 3+1 split, set GPU IDs accordingly."
echo

echo "[Run these in separate terminals/tmux panes on the PI instance]:"
echo "  # Inference (CPU) from prime-rl repo"
echo "  cd $PRIME_RL_ROOT"
echo "  CUDA_VISIBLE_DEVICES= uv run inference @ $CONFIG_DIR/baguettotron_infer.toml"
echo
echo "  # Trainer (4 GPUs)"
echo "  cd $PRIME_RL_ROOT"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run trainer @ $CONFIG_DIR/baguettotron_train.toml"
echo
echo "  # Orchestrator"
echo "  cd $PRIME_RL_ROOT"
echo "  uv run orchestrator @ $CONFIG_DIR/baguettotron_orch.toml"
echo
echo "One-shot (all three under rl, still on the PI instance):"
echo "  cd $PRIME_RL_ROOT"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3 uv run rl \\"
echo "    --trainer @ $CONFIG_DIR/baguettotron_train.toml \\"
echo "    --orchestrator @ $CONFIG_DIR/baguettotron_orch.toml \\"
echo "    --inference @ $CONFIG_DIR/baguettotron_infer.toml"
