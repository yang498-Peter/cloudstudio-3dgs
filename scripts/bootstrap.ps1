[CmdletBinding()]
param(
    [switch]$Training
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$uvExe = Join-Path $venvDir "Scripts\uv.exe"
$lockPath = Join-Path $repoRoot "upstream\gsplat.lock.json"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Command $($Arguments -join ' ')"
    }
}

$pythonVersion = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.12") {
    throw "Python 3.12 is required; detected '$pythonVersion'."
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    Invoke-Checked python -m venv $venvDir
}

Invoke-Checked $venvPython -m pip install "uv>=0.12,<0.13" --disable-pip-version-check
Invoke-Checked $uvExe sync --frozen --active --project $repoRoot

if (-not $Training) {
    Write-Host "CPU environment ready. Run scripts\doctor.ps1 for diagnostics."
    exit 0
}

$lock = Get-Content -LiteralPath $lockPath -Encoding UTF8 | ConvertFrom-Json
$patchPath = Join-Path $repoRoot ($lock.patch -replace "/", "\")
$actualPatchHash = (Get-FileHash -LiteralPath $patchPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualPatchHash -ne $lock.patch_sha256) {
    throw "Patch SHA256 mismatch. Expected $($lock.patch_sha256), got $actualPatchHash."
}

$externalRoot = Join-Path $repoRoot "external"
$gsplatDir = Join-Path $externalRoot "gsplat"
if (-not (Test-Path -LiteralPath $gsplatDir)) {
    New-Item -ItemType Directory -Force -Path $externalRoot | Out-Null
    Invoke-Checked git clone --filter=blob:none --no-checkout $lock.repo $gsplatDir
}
if (-not (Test-Path -LiteralPath (Join-Path $gsplatDir ".git"))) {
    throw "$gsplatDir exists but is not a Git checkout."
}

$dirty = & git -C $gsplatDir status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "Unable to inspect gsplat worktree."
}
$patchAlreadyApplied = $false
& git -C $gsplatDir apply --reverse --check $patchPath 2>$null
if ($LASTEXITCODE -eq 0) {
    $patchAlreadyApplied = $true
}
if ($dirty -and -not $patchAlreadyApplied) {
    throw "gsplat has unrelated local changes; refusing to overwrite them."
}

$head = (& git -C $gsplatDir rev-parse HEAD 2>$null)
if ($LASTEXITCODE -ne 0 -or $head.Trim() -ne $lock.commit) {
    if ($dirty) {
        throw "gsplat is patched but HEAD is not the locked commit; refusing to switch revisions."
    }
    Invoke-Checked git -C $gsplatDir fetch --depth 1 origin $lock.commit
    Invoke-Checked git -C $gsplatDir checkout --detach $lock.commit
}

if (-not $patchAlreadyApplied) {
    Invoke-Checked git -C $gsplatDir apply --check $patchPath
    Invoke-Checked git -C $gsplatDir apply $patchPath
}
Invoke-Checked git -C $gsplatDir submodule update --init --depth 1 gsplat/cuda/csrc/third_party/glm

$setupScript = Join-Path $repoRoot "train\setup_new_machine.cmd"
Invoke-Checked $setupScript
Write-Host "Training environment ready at locked gsplat commit $($lock.commit)."
