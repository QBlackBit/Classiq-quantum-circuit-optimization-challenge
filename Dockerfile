# QBlackBit side-circuit search worker (CUDA).  Build and run with run.ps1 / run.sh.
# Default base supports NVIDIA drivers >= 550.  GPU code is compiled for Pascal ... Hopper
# (sm_61 .. sm_90) plus PTX, which newer GPUs (e.g. RTX 50 series) JIT-compile.
ARG CUDA_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04
FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core.h io.h anneal_gpu.cu core_test.c verify.py selftest.py worker.py problems.json problems_twosplit.json ./
RUN nvcc -O3 -std=c++17 -o anneal_gpu anneal_gpu.cu \
      -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 \
      -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
      -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
      -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 \
 && gcc -D_POSIX_C_SOURCE=200809L -O3 -std=c11 -Wall -Wextra -Werror -o core_test core_test.c -lm
ENTRYPOINT ["python3", "worker.py", "--out", "/out"]
