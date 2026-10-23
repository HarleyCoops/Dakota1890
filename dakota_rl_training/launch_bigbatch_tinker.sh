#!/bin/bash
# Launch Dakota RL Training on Tinker - LARGE BATCH strategy
#
# Strategy ("large-batch is the RL cheat code"):
#   - Many GRPO groups per step averages out a noisy compositional reward,
#     so the per-step gradient points the right way even when any single
#     reward (char-recall + affix-regex + substring-semantic) is unreliable.
#   - Streaming minibatches act as gradient accumulation so the big batch
#     fits memory (process the batch in chunks instead of all at once).
#   - KL penalty stays at 0 so the policy can move; eval runs often because
#     a substring-based reward can be gamed and train-reward != learning.
#
# All knobs are overridable via environment variables, e.g.:
#   BATCH_SIZE=256 STREAM_NUM_MINIBATCHES=8 ./launch_bigbatch_tinker.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
LOG_PATH="${LOG_PATH:-dakota_rl_training/outputs/tinker_bigbatch}"

# Large-batch core: more GROUPS (not bigger groups) is what reduces
# advantage-estimate noise under GRPO.
BATCH_SIZE="${BATCH_SIZE:-192}"          # number of env groups per batch (was 32)
GROUP_SIZE="${GROUP_SIZE:-16}"           # rollouts per GRPO group

# Gradient-accumulation analog: stream the big batch in chunks so it fits
# memory. Keep each minibatch ~32 groups (192 / 6 = 32).
STREAM_GROUPS_PER_BATCH="${STREAM_GROUPS_PER_BATCH:-$BATCH_SIZE}"
STREAM_NUM_MINIBATCHES="${STREAM_NUM_MINIBATCHES:-6}"

# Cleaner gradients tolerate a slightly higher LR; LoRA rank bumped for headroom.
LEARNING_RATE="${LEARNING_RATE:-6e-5}"
LORA_RANK="${LORA_RANK:-64}"

# Higher temperature keeps groups varied so fewer get dropped by
# remove_constant_reward_groups; big batch absorbs the extra variance.
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"

# No KL leash; let the policy move under the cleaner gradient signal.
KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.0}"

# Watch EVAL, not train: a substring reward can be gamed, so check often.
EVAL_EVERY="${EVAL_EVERY:-10}"
SAVE_EVERY="${SAVE_EVERY:-20}"

WANDB_PROJECT="${WANDB_PROJECT:-thinking-machines-qwen3-30b}"
WANDB_NAME="${WANDB_NAME:-dakota-bigbatch-v1}"

# ---------------------------------------------------------------------------
# Resolve paths and load environment
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

echo "======================================================================"
echo "DAKOTA RL TRAINING - LARGE BATCH (TINKER)"
echo "======================================================================"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "[ERROR] TINKER_API_KEY not set (export it or add to .env)."
    exit 1
fi

if ! python -c "import tinker_cookbook" 2>/dev/null; then
    echo "[ERROR] tinker_cookbook not importable. Install Tinker deps first:"
    echo "        pip install tinker tinker-cookbook"
    exit 1
fi

EFFECTIVE_ROLLOUTS=$(( BATCH_SIZE * GROUP_SIZE ))
GROUPS_PER_MINIBATCH=$(( STREAM_GROUPS_PER_BATCH / STREAM_NUM_MINIBATCHES ))

echo ""
echo "Configuration:"
echo "  Model:                 ${MODEL_NAME}"
echo "  Batch size (groups):   ${BATCH_SIZE}"
echo "  Group size:            ${GROUP_SIZE}"
echo "  Effective rollouts:    ${EFFECTIVE_ROLLOUTS} per step"
echo "  Stream minibatches:    ${STREAM_NUM_MINIBATCHES} (~${GROUPS_PER_MINIBATCH} groups each)"
echo "  Learning rate:         ${LEARNING_RATE}"
echo "  LoRA rank:             ${LORA_RANK}"
echo "  Temperature:           ${TEMPERATURE}"
echo "  KL penalty:            ${KL_PENALTY_COEF}"
echo "  Eval / Save every:     ${EVAL_EVERY} / ${SAVE_EVERY}"
echo "  Log path:              ${LOG_PATH}"
echo "  W&B:                   ${WANDB_PROJECT} / ${WANDB_NAME}"
echo ""

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
exec python dakota_rl_training/tinker_train.py \
    --model-name "${MODEL_NAME}" \
    --log-path "${LOG_PATH}" \
    --batch-size "${BATCH_SIZE}" \
    --group-size "${GROUP_SIZE}" \
    --stream-groups-per-batch "${STREAM_GROUPS_PER_BATCH}" \
    --stream-num-minibatches "${STREAM_NUM_MINIBATCHES}" \
    --learning-rate "${LEARNING_RATE}" \
    --lora-rank "${LORA_RANK}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}" \
    --kl-penalty-coef "${KL_PENALTY_COEF}" \
    --eval-every "${EVAL_EVERY}" \
    --save-every "${SAVE_EVERY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --ledger-csv "wandb_analysis/reward_ledger_tinker_bigbatch.csv" \
    --sync-metrics-to-wandb
