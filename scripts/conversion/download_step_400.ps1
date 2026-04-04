# Quick script to download step_400 model files
# Run this from PowerShell

$instanceIP = "65.109.75.43"  # Update if your instance IP is different
$sshKey = "$env:USERPROFILE\.ssh\DakotaRL3"
$remotePath = "~/dakota_rl_training/outputs/ledger_test_400/weights/step_400"
$localPath = "archive\\step0_legacy\\root_artifacts\\downloaded_model_step_400"

Write-Host "Downloading model files from step_400..." -ForegroundColor Green
Write-Host "Remote: root@${instanceIP}:${remotePath}" -ForegroundColor Yellow
Write-Host "Local: ${localPath}" -ForegroundColor Yellow
Write-Host ""

python scripts/conversion/download_model_from_instance.py `
    --instance-ip $instanceIP `
    --remote-path $remotePath `
    --local-path $localPath `
    --ssh-key $sshKey `
    --ssh-port 1234 `
    --user root

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host " Download complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next step: Upload to Hugging Face" -ForegroundColor Cyan
    Write-Host "  python scripts/conversion/upload_model_to_hf.py --model-dir `"${localPath}`"" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host " Download failed. Check the error above." -ForegroundColor Red
}

