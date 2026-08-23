#requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Output,

    [string]$TrainerConfig,

    [switch]$ProbeOnly,

    [switch]$LinkExistingObjects,

    [switch]$ErrorWeightedSampling
)

$ErrorActionPreference = "Stop"
$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$python = Join-Path $repoRoot "external\.venv-gate1-local\Scripts\python.exe"
$cudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Training Python not found: $python"
}
if (-not (Test-Path -LiteralPath (Join-Path $cudaHome "bin\nvcc.exe") -PathType Leaf)) {
    throw "CUDA nvcc not found under $cudaHome"
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = $null
if (Test-Path -LiteralPath $vswhere -PathType Leaf) {
    $vsPath = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
}
if (-not $vsPath -and (Test-Path -LiteralPath "D:\Visual Studio\VC\Auxiliary\Build\vcvars64.bat" -PathType Leaf)) {
    $vsPath = "D:\Visual Studio"
}
if (-not $vsPath) {
    throw "Visual Studio C++ Build Tools not found"
}
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
$ninjaDir = Join-Path $vsPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if (-not (Test-Path -LiteralPath (Join-Path $ninjaDir "ninja.exe") -PathType Leaf)) {
    throw "Visual Studio Ninja not found under $ninjaDir"
}

# A PowerShell-hosted process can inherit both `Path` and `PATH`. cmd.exe uses
# them case-insensitively, while Python may serialize the wrong duplicate into
# nvcc's child environment. Capture vcvars64, normalize keys, then give every
# child exactly one canonical Path entry.
$vcvarsCommand = "call `"$vcvars`" && set"
$vcvarsLines = & $env:ComSpec /d /c $vcvarsCommand
if ($LASTEXITCODE -ne 0) {
    throw "vcvars64 failed with exit code $LASTEXITCODE"
}

$environment = [Collections.Generic.Dictionary[string, string]]::new(
    [StringComparer]::OrdinalIgnoreCase
)
$pathCandidates = [Collections.Generic.List[string]]::new()
foreach ($line in $vcvarsLines) {
    if ($line -notmatch "^([^=]+)=(.*)$") {
        continue
    }
    $name = $Matches[1]
    $value = $Matches[2]
    if ($name.Equals("Path", [StringComparison]::OrdinalIgnoreCase)) {
        $pathCandidates.Add($value)
        continue
    }
    $environment[$name] = $value
}
$compilerPath = $pathCandidates |
    Where-Object { $_ -like "*\VC\Tools\MSVC\*\bin\Hostx64\x64*" } |
    Select-Object -First 1
if (-not $compilerPath) {
    throw "vcvars64 output did not contain an x64 MSVC compiler path"
}

$venvScripts = Split-Path -Parent $python
$environment["Path"] = "$cudaHome\bin;$ninjaDir;$venvScripts;$compilerPath"
$environment["CUDA_HOME"] = $cudaHome
$environment["CUDA_PATH"] = $cudaHome
$environment["VSLANG"] = "1033"
$environment["DISTUTILS_USE_SDK"] = "1"
$environment["MSSdk"] = "1"
$environment["PYTHONPATH"] = Join-Path $repoRoot "train\build_compat"
$environment["TORCH_CUDA_ARCH_LIST"] = "12.0"
$environment["MAX_JOBS"] = "2"
$environment["TORCH_EXTENSIONS_DIR"] = Join-Path $repoRoot "external\.torch-ext-gate1-local"
$environment["NVCC_FLAGS"] = "-Xcompiler /FI$repoRoot\train\build_compat\msvc_clzll.h"
$environment["LINK"] = "/FORCE:UNRESOLVED"
$include = $environment["INCLUDE"]
$environment["INCLUDE"] = "$include;$cudaHome\include\targets\x64"
$runtimeLogRoot = Join-Path $repoRoot "external\runtime-logs"
[IO.Directory]::CreateDirectory($runtimeLogRoot) | Out-Null
$script:processIndex = 0

function Invoke-ConfiguredProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [string]$WorkingDirectory = $repoRoot
    )

    $start = [Diagnostics.ProcessStartInfo]::new()
    $start.FileName = $FilePath
    $start.WorkingDirectory = $WorkingDirectory
    $start.UseShellExecute = $false
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $start.Environment.Clear()
    foreach ($entry in $environment.GetEnumerator()) {
        $start.Environment[$entry.Key] = $entry.Value
    }
    foreach ($argument in $Arguments) {
        [void]$start.ArgumentList.Add($argument)
    }
    $process = [Diagnostics.Process]::Start($start)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $script:processIndex += 1
    $stamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $logPath = Join-Path $runtimeLogRoot ("gate2-process-{0}-{1}.log" -f $stamp, $script:processIndex)
    $payload = "STDOUT`n$stdout`nSTDERR`n$stderr"
    [IO.File]::WriteAllText($logPath, $payload, [Text.UTF8Encoding]::new($false))
    Write-Host "PROCESS_LOG=$logPath"
    Write-Host "PROCESS_EXIT_CODE=$($process.ExitCode)"
    $lines = $payload -split "`r?`n"
    $lines | Select-Object -Last 60 | ForEach-Object { Write-Host $_ }
    return $process.ExitCode
}

function Link-ExistingGsplatObjects {
    $buildDir = Join-Path $repoRoot "external\.torch-ext-gate1-local\gsplat_cuda"
    $ninjaFile = Join-Path $buildDir "build.ninja"
    if (-not (Test-Path -LiteralPath $ninjaFile -PathType Leaf)) {
        throw "gsplat build.ninja is missing: $ninjaFile"
    }
    $linkLine = Get-Content -LiteralPath $ninjaFile -Encoding UTF8 |
        Where-Object { $_ -like "build gsplat_cuda.pyd: link *" } |
        Select-Object -First 1
    if (-not $linkLine) {
        throw "gsplat link rule is missing from $ninjaFile"
    }
    $objectNames = ($linkLine -replace "^build gsplat_cuda\.pyd: link ", "") -split " +"
    $missing = @($objectNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $buildDir $_) -PathType Leaf) })
    if ($missing.Count -gt 0) {
        throw "cannot direct-link gsplat; missing objects: $($missing -join ', ')"
    }
    if ($objectNames.Count -ne 42) {
        throw "expected 42 gsplat objects, found $($objectNames.Count)"
    }

    $compilerBin = $compilerPath -split ";" |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "link.exe") -PathType Leaf } |
        Select-Object -First 1
    if (-not $compilerBin) {
        throw "MSVC link.exe is missing from the normalized compiler Path"
    }
    $linkExe = Join-Path $compilerBin "link.exe"
    $torchLib = (& $python -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent / 'lib')" | Select-Object -Last 1).Trim()
    $pythonBase = (& $python -c "import sys; print(sys.base_prefix)" | Select-Object -Last 1).Trim()
    if (-not (Test-Path -LiteralPath $torchLib -PathType Container)) {
        throw "PyTorch library directory is missing: $torchLib"
    }
    $linkArguments = [Collections.Generic.List[string]]::new()
    foreach ($objectName in $objectNames) {
        $linkArguments.Add($objectName)
    }
    foreach ($argument in @(
        "/nologo",
        "/DLL",
        "c10.lib",
        "c10_cuda.lib",
        "torch_cpu.lib",
        "torch_cuda.lib",
        "/INCLUDE:?warp_size@cuda@at@@YAHXZ",
        "torch.lib",
        "/LIBPATH:$torchLib",
        "torch_python.lib",
        "/LIBPATH:$pythonBase\libs",
        "/LIBPATH:$cudaHome\lib\x64",
        "cudart.lib",
        "/FORCE:UNRESOLVED",
        "/out:gsplat_cuda.pyd"
    )) {
        $linkArguments.Add($argument)
    }
    $linkExit = Invoke-ConfiguredProcess -FilePath $linkExe -Arguments $linkArguments.ToArray() -WorkingDirectory $buildDir
    if ($linkExit -ne 0) {
        exit $linkExit
    }
    $extension = Join-Path $buildDir "gsplat_cuda.pyd"
    if (-not (Test-Path -LiteralPath $extension -PathType Leaf)) {
        throw "link.exe reported success but extension is missing: $extension"
    }
    return $extension
}

$probe = @'
import os, shutil
for name in ("cl", "ninja", "nvcc"):
    path = shutil.which(name)
    print(f"{name.upper()}={path}")
    assert path, f"{name} is missing from child Path"
path_keys = [key for key in os.environ if key.lower() == "path"]
print(f"PATH_KEYS={path_keys}")
assert len(path_keys) == 1, path_keys
'@
$exitCode = Invoke-ConfiguredProcess -FilePath $python -Arguments @("-c", $probe)
if ($exitCode -ne 0) {
    exit $exitCode
}
if ($ProbeOnly) {
    exit 0
}
if ($Output -and $TrainerConfig) {
    throw "Output and TrainerConfig are mutually exclusive"
}
if ($TrainerConfig -and $ErrorWeightedSampling) {
    throw "ErrorWeightedSampling is only valid for the synthetic acceptance path"
}
if (-not $Output -and -not $TrainerConfig) {
    throw "Output or TrainerConfig is required unless -ProbeOnly is used"
}
$outputPath = $null
$trainerConfigPath = $null
if ($TrainerConfig) {
    $trainerConfigPath = if ([IO.Path]::IsPathRooted($TrainerConfig)) {
        [IO.Path]::GetFullPath($TrainerConfig)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $TrainerConfig))
    }
    if (-not (Test-Path -LiteralPath $trainerConfigPath -PathType Leaf)) {
        throw "Trainer config is missing: $trainerConfigPath"
    }
} else {
    $outputPath = if ([IO.Path]::IsPathRooted($Output)) {
        [IO.Path]::GetFullPath($Output)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $Output))
    }
    if (Test-Path -LiteralPath $outputPath) {
        throw "Output already exists: $outputPath"
    }
}

$extension = Join-Path $repoRoot "external\.torch-ext-gate1-local\gsplat_cuda\gsplat_cuda.pyd"
if ($LinkExistingObjects) {
    $extension = Link-ExistingGsplatObjects
}
if (-not (Test-Path -LiteralPath $extension -PathType Leaf)) {
    throw "prebuilt gsplat extension is missing; use -LinkExistingObjects after all objects compile"
}
$bootstrap = Join-Path $repoRoot "tools\run_with_prebuilt_gsplat.py"
$exitCode = Invoke-ConfiguredProcess -FilePath $python -Arguments @(
    $bootstrap,
    "--extension", $extension,
    "--probe"
)
if ($exitCode -ne 0) {
    exit $exitCode
}

if ($trainerConfigPath) {
    $exitCode = Invoke-ConfiguredProcess -FilePath $python -Arguments @(
        $bootstrap,
        "--extension", $extension,
        "tools\train_gsplat.py",
        "--config", $trainerConfigPath
    )
    exit $exitCode
}

$acceptanceArguments = @(
    $bootstrap,
    "--extension", $extension,
    "tools\run_synthetic_training_acceptance.py",
    "--output", $outputPath,
    "--gsplat-lock", "upstream\cloudstudio_trainer.lock.json",
    "--steps", "80",
    "--full-mcmc",
    "--resume-equivalence"
)
if ($ErrorWeightedSampling) {
    $acceptanceArguments += "--error-weighted-sampling"
}
$exitCode = Invoke-ConfiguredProcess -FilePath $python -Arguments $acceptanceArguments
exit $exitCode
