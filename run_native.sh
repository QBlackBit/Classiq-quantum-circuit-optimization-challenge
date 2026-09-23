#!/usr/bin/env bash
# Without Docker (Linux or WSL2 Ubuntu).  Needs python3 and gcc; for the GPU also the CUDA
# toolkit (nvcc).  Without nvcc the worker runs on all CPU cores.  Usage: ./run_native.sh [options]
set -euo pipefail
cd "$(dirname "$0")"
gcc -D_POSIX_C_SOURCE=200809L -O3 -std=c11 -Wall -Wextra -Werror -o core_test core_test.c -lm
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -std=c++17 -arch=native -o anneal_gpu anneal_gpu.cu || { echo "nvcc build failed; running on CPU only"; rm -f anneal_gpu; }
else
  echo "nvcc not found: running on CPU only"
fi
mkdir -p out
python3 worker.py --out out "$@"
