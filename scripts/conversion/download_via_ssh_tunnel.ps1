# Alternative download method using SSH tunnel
# This works if you're already connected via SSH

# Step 1: First, create the archive on the remote server
# Run this in your SSH session:
#   cd ~/dakota_rl_training/outputs/ledger_test_400/weights
#   tar -czf ~/model_step_400.tar.gz step_400/

# Step 2: Then download from PowerShell (new window)
# You'll need the public IP from Prime Intellect dashboard

Write-Host "Downloading model archive..." -ForegroundColor Green
Write-Host ""

$localArchive = "archive\\step0_legacy\\root_artifacts\\model_step_400.tar.gz"
$localExtractDir = "archive\\step0_legacy\\root_artifacts"
New-Item -ItemType Directory -Force -Path $localExtractDir | Out-Null
Write-Host "First, create the archive on the server:" -ForegroundColor Yellow
Write-Host "  cd ~/dakota_rl_training/outputs/ledger_test_400/weights" -ForegroundColor White
Write-Host "  tar -czf ~/model_step_400.tar.gz step_400/" -ForegroundColor White
Write-Host ""
Write-Host "Then run this script with the correct IP:" -ForegroundColor Yellow
Write-Host ""

$instanceIP = Read-Host "Enter instance IP (check Prime Intellect dashboard)"
$sshKey = "$env:USERPROFILE\.ssh\prime_rl_key"

if (-not (Test-Path $sshKey)) {
    Write-Host "SSH key not found at: $sshKey" -ForegroundColor Red
    Write-Host "Trying without explicit key..." -ForegroundColor Yellow
    $sshKey = $null
}

$scpCmd = "scp"
if ($sshKey) {
    $scpCmd += " -i `"$sshKey`""
}
$scpCmd += " root@${instanceIP}:~/model_step_400.tar.gz `"$localArchive`""

Write-Host "Running: $scpCmd" -ForegroundColor Cyan
Invoke-Expression $scpCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host " Download complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Extracting archive..." -ForegroundColor Yellow
    tar -xzf $localArchive -C $localExtractDir
    if ($LASTEXITCODE -eq 0) {
        Write-Host " Extracted under: ${localExtractDir}" -ForegroundColor Green
        Write-Host ""
        Write-Host "Next: Upload to Hugging Face" -ForegroundColor Cyan
        Write-Host "  python scripts/conversion/upload_model_to_hf.py --model-dir `"${localExtractDir}\\step_400`"" -ForegroundColor White
    }
} else {
    Write-Host ""
    Write-Host " Download failed. Try:" -ForegroundColor Red
    Write-Host "  1. Check Prime Intellect dashboard for public IP" -ForegroundColor Yellow
    Write-Host "  2. Use web console file download if available" -ForegroundColor Yellow
    Write-Host "  3. Or use SSH port forwarding" -ForegroundColor Yellow
}

