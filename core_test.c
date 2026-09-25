/*
 * core_test.c -- CPU harness running the exact chain logic of the GPU worker
 * (core.h: init_chain / run_slice), sequentially over `chains` chains.
 * Used to unit-test the core on machines without CUDA and as a slow fallback.
 */
#include <math.h>
#include <time.h>

static double wall_s(void) { return (double)time(NULL); }   /* portable (1 s resolution is enough) */
#include "io.h"

int main(void) {
    problem_t P; char mode[16];
    read_problem(&P, mode);
    if (!strcmp(mode, "dcheck")) {                      /* depth model on given circuits */
        int n; if (scanf("%d", &n) != 1 || n < 1) die("bad circuit count");
        uint32_t st[MAXK];
        for (int c = 0; c < n; c++) { read_circuit(&P, st); printf("DEPTH %d\n", depth_est(&P, st)); }
        return 0;
    }
    if (!strcmp(mode, "depth")) {
        int chains, maxsol, rep; unsigned long long seed; double tl; dparams_t D; int K1;
        if (scanf("%d %u %u %d %llu %d %f %f %d %d %d %lf %d %d", &chains, &D.it1, &D.it2, &D.S, &seed, &maxsol,
                  &D.T0, &D.T1, &D.W, &K1, &rep, &tl, &D.K1a, &D.NA) != 14) die("bad depth params");
        if (chains < 1 || D.it1 < 1 || D.it2 < 1 || D.S < 1 || D.W < 1 || K1 < 1 || K1 > P.K || D.T0 <= 0 || D.T1 <= 0
            || D.K1a < 1 || D.K1a > K1 || D.NA < 1 || D.NA > P.ntarg) die("depth params out of range");
        P.K1 = K1;
        int nsol = 0, best = 1 << 30; double t0 = wall_s();
        for (int c = 0; c < chains && nsol < maxsol && wall_s() - t0 <= tl; c++) {
            uint64_t rs = (seed + 1) * 0x9E3779B97F4A7C15ULL + (uint64_t)(c + 1) * 0xBF58476D1CE4E5B9ULL; if (!rs) rs = 1;
            uint32_t st[MAXK], bst[MAXK], out[MAXK], it; int f, ph, bd, od;
            d_restart(&P, &D, st, &f, &rs, &it, &ph, &bd);
            int done = 0;
            while (wall_s() - t0 <= tl) {                   /* one chain until its first depth stage ends */
                int ph_before = ph;
                if (chain_step(&P, &D, st, &f, &rs, &it, &ph, bst, &bd, out, &od)) {
                    if (od < best) best = od;
                    if (od <= rep) { print_depth_solution(&P, out, od); nsol++; }
                    done = 1;
                    break;
                }
                if (ph == 0 && ph_before != 0) break;          /* correctness stage gave up: next chain */
            }
            if (!done && ph == 2) {                            /* time is up mid depth stage: keep its best */
                if (bd < best) best = bd;
                if (bd <= rep) { print_depth_solution(&P, bst, bd); nsol++; }
            }
        }
        printf("DONE %d %d\n", nsol, best);
        return 0;
    }
    if (!strcmp(mode, "check")) {
        int n; if (scanf("%d", &n) != 1 || n < 1) die("bad circuit count");
        uint32_t st[MAXK]; tt_t w[MAXW];
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
        init_chain(&P, st, &f, &rs, &it, P.K1, P.ntarg);
        while (it < iters) {
            if (run_slice(&P, st, &f, &rs, &it, iters, slice, T0, T1, 0, P.K1, P.ntarg)) {
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
