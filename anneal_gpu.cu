/*
 * anneal_gpu.cu -- CUDA worker: tens of thousands of independent annealing chains
 * (core.h: init_chain / run_slice, the same code the CPU harness runs).  A chain that
 * exhausts its iteration budget restarts from a fresh random circuit, so the GPU
 * performs massive numbers of independent restarts.  Work is issued in short slices
 * (kernel launches of S moves per chain) so no launch trips the Windows/WSL watchdog.
 *
 * Same stdin protocol as core_test (io.h).  Output: SOL lines, "DONE nsol best".
 * Every CUDA call is checked; any error aborts with exit code 4 (never silent).
 */
#include <cuda_runtime.h>
#include <time.h>
#include "io.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "FATAL CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
    fflush(stderr); exit(4); } } while (0)

__constant__ problem_t dP;

__device__ uint64_t seed_of(unsigned long long seed, int c, unsigned round) {
    uint64_t r = (seed + 1ULL) * 0x9E3779B97F4A7C15ULL + (uint64_t)(c + 1) * 0xBF58476D1CE4E5B9ULL
               + (uint64_t)round * 0x94D049BB133111EBULL;
    return r ? r : 1ULL;
}

__global__ void k_init(uint32_t *st, int *f, uint64_t *rs, uint32_t *it, int chains, unsigned long long seed) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chains) return;
    uint64_t r = seed_of(seed, c, 0);
    uint32_t s[MAXK]; int ff; uint32_t i;
    init_chain(&dP, s, &ff, &r, &i);
    for (int k = 0; k < dP.K; k++) st[(size_t)c * MAXK + k] = s[k];
    f[c] = ff; rs[c] = r; it[c] = i;
}

__global__ void k_slice(uint32_t *st, int *f, uint64_t *rs, uint32_t *it, int chains, uint32_t total, int S,
                        float T0, float T1, uint32_t *solbuf, int *solcount, int maxsol, int *gbest) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= chains) return;
    uint32_t s[MAXK];
    for (int k = 0; k < dP.K; k++) s[k] = st[(size_t)c * MAXK + k];
    int ff = f[c]; uint64_t r = rs[c]; uint32_t i = it[c];
    int got = run_slice(&dP, s, &ff, &r, &i, total, S, T0, T1);
    if (got) {
        int idx = atomicAdd(solcount, 1);
        if (idx < maxsol)
            for (int k = 0; k < dP.K; k++) solbuf[(size_t)idx * MAXK + k] = s[k];
    }
    atomicMin(gbest, ff);
    if (got || i >= total) init_chain(&dP, s, &ff, &r, &i);     /* independent restart */
    for (int k = 0; k < dP.K; k++) st[(size_t)c * MAXK + k] = s[k];
    f[c] = ff; rs[c] = r; it[c] = i;
}

__global__ void k_check(const uint32_t *circ, int *out, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    uint64_t w[MAXW];
    replay(&dP, circ + (size_t)c * MAXK, w);
    out[c] = fitness(&dP, w);
}

static double wall_s(void) { return (double)time(NULL); }   /* portable: Linux and MSVC */

int main(void) {
    problem_t P; char mode[16];
    read_problem(&P, mode);
    int dev = 0; cudaDeviceProp prop;
    CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    fprintf(stderr, "GPU %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    CK(cudaMemcpyToSymbol(dP, &P, sizeof P));
    const int TPB = 128;

    if (!strcmp(mode, "check")) {
        int n; if (scanf("%d", &n) != 1 || n < 1) die("bad circuit count");
        uint32_t *h = (uint32_t *)calloc((size_t)n * MAXK, sizeof(uint32_t)); int *ho = (int *)malloc(sizeof(int) * n);
        if (!h || !ho) die("out of host memory");
        for (int c = 0; c < n; c++) read_circuit(&P, h + (size_t)c * MAXK);
        uint32_t *d; int *dout;
        CK(cudaMalloc(&d, (size_t)n * MAXK * sizeof(uint32_t))); CK(cudaMalloc(&dout, sizeof(int) * n));
        CK(cudaMemcpy(d, h, (size_t)n * MAXK * sizeof(uint32_t), cudaMemcpyHostToDevice));
        k_check<<<(n + TPB - 1) / TPB, TPB>>>(d, dout, n);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ho, dout, sizeof(int) * n, cudaMemcpyDeviceToHost));
        for (int c = 0; c < n; c++) printf("FIT %d\n", ho[c]);
        return 0;
    }
    if (strcmp(mode, "search")) die("mode must be search or check");
    int chains, S, maxsol; unsigned long long seed; unsigned iters; float T0, T1; double tl;
    if (scanf("%d %u %d %llu %d %f %f %lf", &chains, &iters, &S, &seed, &maxsol, &T0, &T1, &tl) != 8) die("bad search params");
    if (chains < 1 || iters < 1 || S < 1 || maxsol < 1 || T0 <= 0 || T1 <= 0) die("search params out of range");

    uint32_t *d_st, *d_it, *d_sol; int *d_f, *d_cnt, *d_best; uint64_t *d_rs;
    CK(cudaMalloc(&d_st, (size_t)chains * MAXK * sizeof(uint32_t)));
    CK(cudaMalloc(&d_it, (size_t)chains * sizeof(uint32_t)));
    CK(cudaMalloc(&d_f, (size_t)chains * sizeof(int)));
    CK(cudaMalloc(&d_rs, (size_t)chains * sizeof(uint64_t)));
    CK(cudaMalloc(&d_sol, (size_t)maxsol * MAXK * sizeof(uint32_t)));
    CK(cudaMalloc(&d_cnt, sizeof(int))); CK(cudaMalloc(&d_best, sizeof(int)));
    int zero = 0, big = 1 << 30;
    CK(cudaMemcpy(d_cnt, &zero, sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_best, &big, sizeof(int), cudaMemcpyHostToDevice));
    int blocks = (chains + TPB - 1) / TPB;
    k_init<<<blocks, TPB>>>(d_st, d_f, d_rs, d_it, chains, seed);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    double t0 = wall_s(), last = t0; int printed = 0, cnt = 0, best = big;
    uint32_t *h_sol = (uint32_t *)malloc((size_t)maxsol * MAXK * sizeof(uint32_t));
    if (!h_sol) die("out of host memory");
    unsigned long long moves = 0;
    while (1) {
        k_slice<<<blocks, TPB>>>(d_st, d_f, d_rs, d_it, chains, iters, S, T0, T1, d_sol, d_cnt, maxsol, d_best);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        moves += (unsigned long long)chains * (unsigned long long)S;
        CK(cudaMemcpy(&cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&best, d_best, sizeof(int), cudaMemcpyDeviceToHost));
        if (cnt > maxsol) cnt = maxsol;
        if (cnt > printed) {
            CK(cudaMemcpy(h_sol, d_sol, (size_t)cnt * MAXK * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (int q = printed; q < cnt; q++) {
                const uint32_t *st = h_sol + (size_t)q * MAXK;
                uint64_t w[MAXW];
                for (int k = 0; k < P.K; k++) if (!step_ok(&P, st[k])) die("illegal step in a GPU solution");
                replay(&P, st, w);
                if (fitness(&P, w) != 0) die("GPU solution fails the host fitness re-check");
                print_solution(&P, st);
            }
            printed = cnt;
        }
        double t = wall_s();
        if (t - last > 15.0) {
            fprintf(stderr, "progress %.0fs: %.3g moves, %d solutions, best fitness %d\n", t - t0, (double)moves, printed, best);
            fflush(stderr); last = t;
        }
        if (printed >= maxsol || t - t0 > tl) break;
    }
    printf("DONE %d %d\n", printed, best);
    return 0;
}
