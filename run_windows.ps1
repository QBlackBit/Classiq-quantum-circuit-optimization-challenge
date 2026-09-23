# Native Windows run, no Docker.
# Needs: NVIDIA driver + CUDA Toolkit (nvcc), Visual Studio 2019/2022 or its Build Tools with
# "Desktop development with C++" (nvcc on Windows can only use Microsoft's cl.exe; MSYS2 gcc
# cannot be used for CUDA), and Python 3.8+ (python.org build).
# Usage (PowerShell, in this folder):   .\run_windows.ps1            [extra worker options]
param([Parameter(ValueFromRemainingArguments = $true)] $WorkerArgs)
$ErrorActionPreference = "Stop"
if (-not $WorkerArgs) { $WorkerArgs = @() }
Set-Location -Path $PSScriptRoot

function Fail([string]$msg) {
    Write-Host ""
    Write-Host "ERROR: $msg" -ForegroundColor Red
    exit 1
}

# ---- 1. Python -------------------------------------------------------------------------
$py = $null; $pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py"; $pyArgs = @("-3") }
elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $py = "python3" }
if (-not $py) { Fail "Python 3 not found. Install it from https://www.python.org/downloads/ and tick 'Add python.exe to PATH'." }
& $py @pyArgs -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if ($LASTEXITCODE -ne 0) { Fail "Python 3.8 or newer is required ($py)." }
Write-Host "python: $py $pyArgs"

# ---- 2. nvcc (CUDA Toolkit) ------------------------------------------------------------
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    if ($env:CUDA_PATH -and (Test-Path (Join-Path $env:CUDA_PATH "bin\nvcc.exe"))) {
        $env:PATH = (Join-Path $env:CUDA_PATH "bin") + ";" + $env:PATH
    }
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    Fail "nvcc not found. Install the CUDA Toolkit (https://developer.nvidia.com/cuda-downloads) or add its bin folder to PATH."
}
Write-Host ("nvcc: " + (Get-Command nvcc).Source)

# ---- 3. MSVC (cl.exe), loaded from Visual Studio's vcvars64.bat --------------------------
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        Fail "Visual Studio not found. Install 'Build Tools for Visual Studio 2022' with 'Desktop development with C++' (https://visualstudio.microsoft.com/downloads/)."
    }
    $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { Fail "Visual Studio has no C++ compiler. Add the 'Desktop development with C++' workload." }
    $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path $vcvars)) { Fail "vcvars64.bat not found under $vs" }
    $bat = Join-Path $env:TEMP "qbb_vcvars_dump.bat"
    Set-Content -Path $bat -Encoding ASCII -Value "@call `"$vcvars`" >nul 2>&1`r`n@if errorlevel 1 exit /b 1`r`n@set"
    $dump = & cmd.exe /c $bat
    if ($LASTEXITCODE -ne 0) { Fail "vcvars64.bat failed." }
    foreach ($line in $dump) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path ("env:" + $matches[1]) -Value $matches[2] }
    }
    Remove-Item $bat -ErrorAction SilentlyContinue
}
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) { Fail "cl.exe still not available after loading vcvars64.bat." }
Write-Host ("cl: " + (Get-Command cl).Source)

# ---- 4. build both engines ---------------------------------------------------------------
Write-Host "building the CUDA engine (anneal_gpu.exe) ..."
& nvcc -O3 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -arch=native -o anneal_gpu.exe anneal_gpu.cu
if ($LASTEXITCODE -ne 0) {
    Write-Host "nvcc -arch=native failed (older CUDA?); retrying with an explicit architecture list ..."
    & nvcc -O3 -std=c++17 -D_CRT_SECURE_NO_WARNINGS -o anneal_gpu.exe anneal_gpu.cu `
        -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75 `
        -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 `
        -gencode arch=compute_89,code=compute_89
    if ($LASTEXITCODE -ne 0) { Fail "CUDA build failed; please send the messages above." }
}
Write-Host "building the CPU engine (core_test.exe) ..."
& cl /nologo /O2 /TP /EHsc /D_CRT_SECURE_NO_WARNINGS core_test.c /Fe:core_test.exe
if ($LASTEXITCODE -ne 0) { Fail "CPU engine build failed; please send the messages above." }
Remove-Item core_test.obj -ErrorAction SilentlyContinue

# ---- 5. run --------------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path out | Out-Null
Write-Host "starting the worker (log: out\worker.log) ..."
& $py @pyArgs worker.py --out out @WorkerArgs
if ($LASTEXITCODE -ne 0) { Fail "worker stopped with exit code $LASTEXITCODE -- please send out\worker.log" }
