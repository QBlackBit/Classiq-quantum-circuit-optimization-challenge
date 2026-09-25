# QBlackBit GPU search worker

This worker helps the QBlackBit entry in the Classiq *Build a Better Phase Oracle*
challenge. It searches on an NVIDIA GPU for small reversible circuits that our oracle
designs need. Tens of thousands of simulated-annealing chains run at once. Each chain
first finds a *correct* circuit, then anneals it towards the *smallest depth*, using an
exact model of the final u3/cx depth.

**Current mission (default): the single-comparator oracle.** The logo equals one
comparison `[l(y) + mu(x,u) >= 6]`. The oracle then needs only two small circuits: a
3-bit level code of the rows (64 points) and one of the columns (128 points). If both
reach a depth of about 33, the whole oracle has a depth of about 120. The worker searches
the codes round-robin and keeps the shallowest verified circuits per code.

Two machines can share the work:

| machine | command | searches |
|---|---|---|
| stronger GPU | `.\run_windows.ps1 --only cols` (or `.\run.ps1 --only cols`) | column codes (harder, 128 points) |
| other GPU | `.\run.ps1 --only rows` | row codes |

The previous mission (the 4+4 two-round search) is still available with
`--problems problems_twosplit.json`. Stop it (Ctrl+C), `git pull`, and restart with the
command above; the `out` folder can stay.

## What you need

* Windows 10/11 with an NVIDIA GPU and a recent driver (R550 or newer),
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) using the WSL2
  backend (the default). GPU support is built in.

## Run it natively on Windows (no Docker)

Needs the **CUDA Toolkit** (`nvcc`), **Visual Studio 2019/2022 or its Build Tools with
"Desktop development with C++"** and **Python 3.8+** (python.org). On Windows `nvcc`
can only use Microsoft's compiler (`cl.exe`), so MSYS2/gcc is *not* used; the script
finds Visual Studio by itself and loads its environment. In PowerShell, in this folder:

```powershell
Set-ExecutionPolicy -Scope Process Bypass     # only if scripts are blocked
.\run_windows.ps1
```

It checks every requirement and stops with an exact message if one is missing. It then
builds `anneal_gpu.exe` (CUDA) and `core_test.exe` (CPU) and starts the worker.
Options can be appended, e.g. `.\run_windows.ps1 --hours 10`.

## Run it with Docker (Windows)

Open PowerShell in this folder and run:

```powershell
.\run.ps1
```

The first run builds the image, which takes a few minutes. After that the worker:

1. **self-tests** the GPU engine: GPU results must match an independent Python
   implementation on 300 random circuits, and it must recover known ("planted")
   solutions. If the GPU check fails it says so loudly and falls back to the CPU.
   If everything fails it stops;
2. works through `problems.json` in priority order, writing progress to the console
   and to `out\worker.log`;
3. saves every verified circuit to `out\results\` and its progress to `out\state.json`.

You can stop it at any time with `Ctrl+C`. Running `.\run.ps1` again **resumes** where
it stopped. To bound the run time: `.\run.ps1 --hours 10`.

If PowerShell refuses to run the script, allow local scripts for this session first:
`Set-ExecutionPolicy -Scope Process Bypass`.

## What to send back

Zip the whole `out` folder and send it, at the end or whenever the log shows new
records:

```
*** NEW BEST col_5-31-20: depth 38 ***
```

Sending it every few hours is ideal: we assemble and verify the full oracle from the
best row and column circuits found so far.

If anything goes wrong, send `out\worker.log`. Errors are never swallowed: they are
written there with a full traceback.

## Linux / WSL2 without Docker

In an Ubuntu shell under WSL2 (or on Linux) with `python3` and `gcc`:

```bash
./run_native.sh
```

If the CUDA toolkit (`nvcc`) is installed it builds and uses the GPU. Otherwise it
runs on all CPU cores, which is still useful but much slower.

## Correctness design

* `core.h` holds all search logic. The CPU harness (`core_test.c`) and the GPU kernel
  (`anneal_gpu.cu`) run this same code, and the CPU path is unit-tested.
* The engines only *propose* circuits. `verify.py` is independent pure-Python code
  that re-checks each proposal (legal steps, exact span membership by GF(2)
  elimination) before anything is saved. An invalid proposal stops the run.
* The GPU host code re-checks every solution a second time before printing it.
* Results are checked a third time by the team on all 4096 inputs of the full
  oracle, including a statevector simulation.

## Files

| file | role |
|---|---|
| `core.h` | search core (model, fitness, moves, annealing), shared by CPU and GPU |
| `io.h` | input parsing and output, strict (malformed input aborts) |
| `anneal_gpu.cu` | CUDA engine: many chains, independent restarts, sliced launches |
| `core_test.c` | CPU engine / test harness using the same core |
| `verify.py` | independent verifier |
| `selftest.py` | engine self-tests (fitness equivalence, planted recovery) |
| `worker.py` | orchestration, verification, resumable state, logging |
| `run_windows.ps1` | native Windows build + run (CUDA Toolkit + Visual Studio) |
| `run.ps1` / `run.sh` | Docker build + run (Windows / Linux) |
| `run_native.sh` | Linux / WSL2 build + run without Docker |
| `problems.json` | the search problems (generated by `make_problems.py`) |
