/*
 * core_test.c -- CPU harness running the exact chain logic of the GPU worker
 * (core.h: init_chain / run_slice), sequentially over `chains` chains.
 * Used to unit-test the core on machines without CUDA and as a slow fallback.
 */
#include <math.h>
#include <time.h>

static double wall_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#include "io.h"

int main(void) {
    problem_t P; char mode[16];
    read_problem(&P, mode);
    if (!strcmp(mode, "check")) {
        int n; if (scanf("%d", &n) != 1 || n < 1) die("bad circuit count");
        uint32_t st[MAXK]; uint64_t w[MAXW];
        for (int c = 0; c < n; c++) {
            read_circuit(&P, st);
            replay(&P, st, w);
            printf("FIT %d\n", fitness(&P, w));
        }
        return 0;
    }
    if (strcmp(mode, "search")) die("mode must be search or check");
    int chains, slice, maxsol; unsigned long long seed; unsigned iters; float T0, T1; double tl;
    if (scanf("%d %u %d %llu %d %f %f %lf", &chains, &iters, &slice, &seed, &maxsol, &T0, &T1, &tl) != 8) die("bad search params");
    if (chains < 1 || iters < 1 || slice < 1 || T0 <= 0 || T1 <= 0) die("search params out of range");
    int nsol = 0, best = 1 << 30; double t0 = wall_s();
    for (int c = 0; c < chains && nsol < maxsol; c++) {
        if (wall_s() - t0 > tl) break;                                  /* wall-clock limit across chains */
        uint64_t rs = (seed + 1) * 0x9E3779B97F4A7C15ULL + (uint64_t)(c + 1) * 0xBF58476D1CE4E5B9ULL; if (!rs) rs = 1;
        uint32_t st[MAXK], it; int f;
        init_chain(&P, st, &f, &rs, &it);
        while (it < iters) {
            if (run_slice(&P, st, &f, &rs, &it, iters, slice, T0, T1)) {
                for (int k = 0; k < P.K; k++) if (!step_ok(&P, st[k])) die("illegal step in a solution (core bug)");
                print_solution(&P, st); nsol++;
                break;
            }
            if (wall_s() - t0 > tl) break;
        }
        if (f < best) best = f;
    }
    printf("DONE %d %d\n", nsol, best);
    return 0;
}
