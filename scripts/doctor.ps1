[CmdletBinding()]
param(
    [switch]$RequireTrainingReady
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$lockPath = Join-Path $repoRoot "upstream\gsplat.lock.json"
$lock = Get-Content -LiteralPath $lockPath -Encoding UTF8 | ConvertFrom-Json
$issues = [System.Collections.Generic.List[string]]::new()

function Get-CommandVersion {
    param([string]$Name, [string[]]$Arguments)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if (-not $command) {
        return $null
    }
    $value = & $command.Source @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return ($value | Select-Object -First 1).ToString().Trim()
}

$python = Get-CommandVersion "python" @("--version")
$git = Get-CommandVersion "git" @("--version")
$nvcc = Get-CommandVersion "nvcc" @("--version")
$driver = Get-CommandVersion "nvidia-smi" @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
$torch = $null
$cudaAvailable = $false
$cudaArch = $null
$gsplat = $null

if ($python) {
    $runtime = & python -c "import json,sys; print(json.dumps({'python':sys.version.split()[0]}))"
    if ($LASTEXITCODE -eq 0) {
        $python = ($runtime | ConvertFrom-Json).python
    }
    $torchResult = & python -c "import json;`ntry:`n import torch; print(json.dumps({'version':torch.__version__,'cuda':torch.cuda.is_available(),'arch':torch.cuda.get_device_capability() if torch.cuda.is_available() else None}))`nexcept Exception as e: print(json.dumps({'error':str(e)}))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $torchResult) {
        $torchInfo = $torchResult | ConvertFrom-Json
        if (-not $torchInfo.error) {
            $torch = $torchInfo.version
            $cudaAvailable = [bool]$torchInfo.cuda
            if ($torchInfo.arch) {
                $cudaArch = "$($torchInfo.arch[0]).$($torchInfo.arch[1])"
            }
        }
    }
    $gsplatResult = & python -c "import json;`ntry:`n import gsplat; print(json.dumps({'version':gsplat.__version__}))`nexcept Exception as e: print(json.dumps({'error':str(e)}))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $gsplatResult) {
        $gsplatInfo = $gsplatResult | ConvertFrom-Json
        if (-not $gsplatInfo.error) {
            $gsplat = $gsplatInfo.version
        }
    }
}

$gsplatDir = Join-Path $repoRoot "external\gsplat"
$gsplatCommit = $null
$patchApplied = $false
if (Test-Path -LiteralPath (Join-Path $gsplatDir ".git")) {
    $gsplatCommit = (& git -C $gsplatDir rev-parse HEAD).Trim()
    $patchPath = Join-Path $repoRoot ($lock.patch -replace "/", "\")
    & git -C $gsplatDir apply --reverse --check $patchPath 2>$null
    $patchApplied = $LASTEXITCODE -eq 0
}

if (-not $python -or -not $python.StartsWith("3.12.")) { $issues.Add("Python 3.12 is required.") }
if (-not $git) { $issues.Add("Git is required.") }
if ($RequireTrainingReady) {
    if (-not $nvcc) { $issues.Add("nvcc is required for training.") }
    if (-not $driver) { $issues.Add("nvidia-smi did not report a GPU.") }
    if (-not $torch) { $issues.Add("PyTorch is not importable.") }
    if (-not $cudaAvailable) { $issues.Add("PyTorch cannot access CUDA.") }
    if (-not $gsplat) { $issues.Add("gsplat is not importable.") }
    if ($gsplatCommit -ne $lock.commit) { $issues.Add("gsplat HEAD does not match the lock.") }
    if (-not $patchApplied) { $issues.Add("The locked gsplat patch is not applied.") }
}

$report = [ordered]@{
    python = $python
    git = $git
    gpu = $driver
    nvcc = $nvcc
    torch = $torch
    torch_cuda_available = $cudaAvailable
    detected_cuda_arch = $cudaArch
    configured_cuda_arch = $lock.cuda_arch_list
    gsplat = $gsplat
    gsplat_commit = $gsplatCommit
    expected_gsplat_commit = $lock.commit
    patch_applied = $patchApplied
    status = if ($issues.Count -eq 0) { "PASS" } else { "FAIL" }
    issues = @($issues)
}
$report | ConvertTo-Json -Depth 4
if ($issues.Count -gt 0) {
    exit 1
}
