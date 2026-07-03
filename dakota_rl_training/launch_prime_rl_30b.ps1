# Launch Prime-RL Training for Qwen 30B-A3B
# Model: qwen/qwen3-30b-a3b-instruct-2507
# Environment: harleycooper/dakota1890
# 
# GPU Allocation (8 A100s):
#   - Inference: GPUs 0-3 (4 GPUs, dp=2, tp=2)
#   - Trainer: GPUs 4-7 (4 GPUs)
#
# Usage:
#   cd prime-rl-framework
#   .\launch_prime_rl_30b.ps1
#
# Or from Dakota1890 root:
#   cd prime-rl-framework
#   uv run rl `
#     --trainer @ ../dakota_rl_training/configs/train_30b.toml `
#     --orchestrator @ ../dakota_rl_training/configs/orch_30b.toml `
#     --inference @ ../dakota_rl_training/configs/infer_30b.toml `
#     --trainer-gpu-ids 4,5,6,7 `
#     --inference-gpu-ids 0,1,2,3 `
#     --output-dir ../dakota_rl_training/outputs/grpo_30b

# Change to prime-rl-framework directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptPath
$primeRlDir = Join-Path $repoRoot "prime-rl-framework"

if (-not (Test-Path $primeRlDir)) {
    Write-Error "prime-rl-framework directory not found at: $primeRlDir"
    exit 1
}

Set-Location $primeRlDir

Write-Host "Launching Prime-RL training from: $primeRlDir" -ForegroundColor Green
Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Model: qwen/qwen3-30b-a3b-instruct-2507"
Write-Host "  Environment: harleycooper/dakota1890"
Write-Host "  Inference GPUs: 0,1,2,3"
Write-Host "  Trainer GPUs: 4,5,6,7"
Write-Host "  Output Directory: ../dakota_rl_training/outputs/grpo_30b"
Write-Host ""

# Require W&B credentials from the caller's environment.
if ([string]::IsNullOrWhiteSpace($env:WANDB_API_KEY)) {
    Write-Error "WANDB_API_KEY is not set. Set it in your environment before launching Prime-RL."
    exit 1
}

# Launch Prime-RL training
uv run rl `
    --trainer "@" ../dakota_rl_training/configs/train_30b.toml `
    --orchestrator "@" ../dakota_rl_training/configs/orch_30b.toml `
    --inference "@" ../dakota_rl_training/configs/infer_30b.toml `
    --trainer-gpu-ids 4,5,6,7 `
    --inference-gpu-ids 0,1,2,3 `
    --output-dir ../dakota_rl_training/outputs/grpo_30b

# Expected:
#   - Training time: ~1.5 hours (90 minutes)
#   - Dataset: 10,576 examples
#   - Max steps: 500
#   - Checkpoints: Every 100 steps

