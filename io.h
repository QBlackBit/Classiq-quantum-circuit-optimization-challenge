/*
 * io.h -- host-side problem parsing and result printing, shared by the CPU harness
 * and the CUDA worker.  Text protocol on stdin:
 *   line 1:  MODE npts m K maxform ntarg          (MODE = search | check | depth | dcheck)
 *   m lines: ro hex                               (read-only flag, truth table)
 *   ntarg lines: hex                              (targets)
 *   search:  chains iters_per_chain slice seed maxsol T0 T1 time_limit_s
 *   check / dcheck: n, then n circuits of K lines "t am ac bm bc"
 *   depth:   chains iters1 iters2 slice seed maxsol T0 T1 W K1 report_depth time_limit_s
 * Any malformed input aborts with exit code 3 (never silently continues).
 */
#ifndef QBB_IO_H
#define QBB_IO_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core.h"

static void die(const char *msg) { fprintf(stderr, "FATAL: %s\n", msg); fflush(stderr); exit(3); }

/* one hex token (1..32 digits, no prefix) -> 128-bit truth table; aborts on anything else */
static tt_t read_hex128(const char *what) {
    char buf[64]; tt_t r; r.lo = 0; r.hi = 0;
    if (scanf("%63s", buf) != 1) die(what);
    size_t n = strlen(buf);
    if (n < 1 || n > 32) die(what);
    for (size_t i = 0; i < n; i++) {
        char c = buf[i]; int v = -1;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        if (v < 0) die(what);
        r.hi = (r.hi << 4) | (r.lo >> 60);
        r.lo = (r.lo << 4) | (uint64_t)v;
    }
    return r;
}

static void read_problem(problem_t *P, char *mode) {
    memset(P, 0, sizeof *P);
    if (scanf("%15s %d %d %d %d %d", mode, &P->npts, &P->m, &P->K, &P->maxform, &P->ntarg) != 6) die("bad header");
    if (P->npts < 1 || P->npts > 128) die("npts must be 1..128");
    if (P->m < 2 || P->m > MAXW) die("m out of range");
    if (P->K < 1 || P->K > MAXK) die("K out of range");
    P->K1 = P->K;                               /* depth mode may lower it */
    if (P->ntarg < 1 || P->ntarg > MAXT) die("ntarg out of range");
    if (P->maxform < 1 || P->maxform > P->m) die("maxform out of range");
    P->full.lo = P->npts >= 64 ? ~0ULL : ((1ULL << P->npts) - 1);
    P->full.hi = P->npts <= 64 ? 0ULL : (P->npts == 128 ? ~0ULL : ((1ULL << (P->npts - 64)) - 1));
    for (int i = 0; i < P->m; i++) {
        if (scanf("%d", &P->ro[i]) != 1 || (P->ro[i] | 1) != 1) die("bad wire line");
        tt_t v = read_hex128("bad wire table");
        if ((v.lo & ~P->full.lo) || (v.hi & ~P->full.hi)) die("wire table has bits beyond npts");
        P->init[i] = v;
        if (!P->ro[i]) P->freew[P->nfree++] = i;
    }
    if (P->nfree < 1) die("no writable wire");
    for (int t = 0; t < P->ntarg; t++) {
        tt_t v = read_hex128("bad target table");
        if ((v.lo & ~P->full.lo) || (v.hi & ~P->full.hi)) die("target has bits beyond npts");
        P->targ[t] = v;
    }
}

static void print_steps(const problem_t *P, const uint32_t *st) {
    for (int k = 0; k < P->K; k++)
        printf(" %d,%d,%d,%d,%d", st_t(st[k]), st_am(st[k]), st_ac(st[k]), st_bm(st[k]), st_bc(st[k]));
    printf("\n");
    fflush(stdout);
}

/* depth-mode result: re-checks legality and zero span error on the host before printing */
static void print_depth_solution(const problem_t *P, const uint32_t *st, int d) {
    tt_t w[MAXW];
    for (int k = 0; k < P->K; k++) if (!step_ok(P, st[k])) die("illegal step in a depth-stage solution");
    replay(P, st, w);
    if (fitness(P, w) != 0) die("depth-stage solution fails the span re-check");
    if (depth_est(P, st) != d) die("depth-stage solution: reported depth differs from the model");
    printf("SOLD %d", d);
    print_steps(P, st);
}

static void print_solution(const problem_t *P, const uint32_t *st) {
    printf("SOL");
    for (int k = 0; k < P->K; k++)
        printf(" %d,%d,%d,%d,%d", st_t(st[k]), st_am(st[k]), st_ac(st[k]), st_bm(st[k]), st_bc(st[k]));
    printf("\n");
    fflush(stdout);
}

static void read_circuit(const problem_t *P, uint32_t *st) {
    for (int k = 0; k < P->K; k++) {
        int t, am, ac, bm, bc;
        if (scanf("%d %d %d %d %d", &t, &am, &ac, &bm, &bc) != 5) die("bad circuit line");
        if (t < 0 || t >= P->m || am < 0 || am >= (1 << P->m) || bm < 0 || bm >= (1 << P->m) || (ac | bc) >> 1)
            die("circuit step out of range");
        st[k] = st_pack(t, am, ac, bm, bc);
    }
}
#endif
