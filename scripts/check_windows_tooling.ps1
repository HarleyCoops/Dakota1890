param(
    [switch]$Install,
    [switch]$UseSystemPython,
    [switch]$Full,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$script:Failed = $false

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

function Write-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )

    $status = if ($Ok) { "OK" } else { "FAIL" }
    Write-Host ("[{0}] {1}: {2}" -f $status, $Name, $Detail)
    if (-not $Ok) {
        $script:Failed = $true
    }
}

function Invoke-Check {
    param(
        [string]$Name,
        [scriptblock]$Body
    )

    try {
        $global:LASTEXITCODE = 0
        $detail = & $Body
        if ($global:LASTEXITCODE -ne 0) {
            throw "native command exited with code $global:LASTEXITCODE"
        }
        if (-not $detail) {
            $detail = "passed"
        }
        Write-Check -Name $Name -Ok $true -Detail ($detail -join " ")
    }
    catch {
        Write-Check -Name $Name -Ok $false -Detail $_.Exception.Message
    }
}

function Get-ProjectPython {
    $venvPython = Join-Path $RepoRoot ".venv_win\Scripts\python.exe"
    if ((-not $UseSystemPython) -and (Test-Path $venvPython)) {
        return $venvPython
    }

    $pythonCommand = Get-Command python -ErrorAction Stop
    return $pythonCommand.Source
}

$venvPython = Join-Path $RepoRoot ".venv_win\Scripts\python.exe"
if ($Install) {
    if (-not (Test-Path $venvPython)) {
        & python -m venv ".venv_win"
    }

    & $venvPython -m pip install --upgrade pip
    & $venvPython -m pip install -r "requirements-dev.txt"
    & $venvPython -m pip install -r "requirements_hf_inference.txt"
    & $venvPython -m pip install -e "environments\dakota_grammar_translation"
}

$Python = Get-ProjectPython
Write-Host ("Repository: {0}" -f $RepoRoot)
Write-Host ("Python: {0}" -f $Python)

Invoke-Check "python version" {
    & $Python --version
}

try {
    $rgCommand = Get-Command rg -ErrorAction Stop
    try {
        $rgVersion = & $rgCommand.Source --version 2>$null | Select-Object -First 1
        Write-Host ("[OK] rg: {0} ({1})" -f $rgVersion, $rgCommand.Source)
    }
    catch {
        Write-Host ("[WARN] rg: found at {0}, but execution failed: {1}" -f $rgCommand.Source, $_.Exception.Message)
        Write-Host "[WARN] rg: install a normal Windows ripgrep earlier in PATH, for example with winget install BurntSushi.ripgrep.MSVC"
    }
}
catch {
    Write-Host "[WARN] rg: not found. Install ripgrep or use git grep / Select-String on Windows."
}

$importCheck = @"
import importlib.util
mods = ["pytest", "openai", "dotenv", "datasets", "verifiers"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit("missing modules: " + ", ".join(missing))
print("imports ok")
"@

Invoke-Check "python imports" {
    $tempFile = New-TemporaryFile
    try {
        Set-Content -LiteralPath $tempFile -Value $importCheck -Encoding UTF8
        & $Python $tempFile
    }
    finally {
        Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
    }
}

if (-not $SkipTests) {
    Invoke-Check "focused pytest suite" {
        & $Python -m pytest tests/test_verifier_integration.py tests/test_inference_configuration.py tests/test_training_dataset_builder.py tests/test_offline_eval.py tests/test_openai_finetune_readiness.py tests/test_sft_baseline.py -q
    }
}

if ($Full) {
    Invoke-Check "PrimeIntellect local readiness" {
        & $Python "dakota_rl_training\train.py" --check-only
    }

    Invoke-Check "OpenAI SFT readiness" {
        & $Python "scripts\rl\dakota_openai_finetune.py" --check-only
    }
}

if ($script:Failed) {
    exit 1
}
