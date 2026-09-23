#!/usr/bin/env bash
# Linux / WSL2 with Docker and the NVIDIA Container Toolkit.  Usage: ./run.sh [worker options]
set -euo pipefail
cd "$(dirname "$0")"
docker build -t qbb-worker .
mkdir -p out
docker run --rm --gpus all -v "$PWD/out:/out" qbb-worker "$@"
