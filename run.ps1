# Windows (PowerShell), needs Docker Desktop (WSL2 backend) and a recent NVIDIA driver.
# Usage:  .\run.ps1            (extra worker options may follow, e.g. .\run.ps1 --hours 10)
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
docker build -t qbb-worker .
if ($LASTEXITCODE -ne 0) { throw "docker build failed (see messages above)" }
New-Item -ItemType Directory -Force -Path out | Out-Null
docker run --rm --gpus all -v "${PSScriptRoot}\out:/out" qbb-worker @args
if ($LASTEXITCODE -ne 0) { throw "worker failed with exit code $LASTEXITCODE -- please send out\worker.log" }
