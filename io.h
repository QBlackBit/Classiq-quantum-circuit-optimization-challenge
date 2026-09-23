/*
 * io.h -- host-side problem parsing and result printing, shared by the CPU harness
 * and the CUDA worker.  Text protocol on stdin:
 *   line 1:  MODE npts m K maxform ntarg          (MODE = search | check)
 *   m lines: ro hex                               (read-only flag, truth table)
 *   ntarg lines: hex                              (targets)
 *   search:  chains iters_per_chain slice seed maxsol T0 T1 time_limit_s
 *   check:   n, then n circuits of K lines "t am ac bm bc"
 * Any malformed input aborts with exit code 3 (never silently continues).
 */
#ifndef QBB_IO_H
#define QBB_IO_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core.h"

static void die(const char *msg) { fprintf(stderr, "FATAL: %s\n", msg); fflush(stderr); exit(3); }

static void read_problem(problem_t *P, char *mode) {
    memset(P, 0, sizeof *P);
    if (scanf("%15s %d %d %d %d %d", mode, &P->npts, &P->m, &P->K, &P->maxform, &P->ntarg) != 6) die("bad header");
    if (P->npts < 1 || P->npts > 64) die("npts must be 1..64");
    if (P->m < 2 || P->m > MAXW) die("m out of range");
    if (P->K < 1 || P->K > MAXK) die("K out of range");
    if (P->ntarg < 1 || P->ntarg > MAXT) die("ntarg out of range");
    if (P->maxform < 1 || P->maxform > P->m) die("maxform out of range");
    P->full = P->npts == 64 ? ~0ULL : ((1ULL << P->npts) - 1);
    for (int i = 0; i < P->m; i++) {
        unsigned long long v;
        if (scanf("%d %llx", &P->ro[i], &v) != 2) die("bad wire line");
        if (v & ~P->full) die("wire table has bits beyond npts");
        P->init[i] = v;
        if (!P->ro[i]) P->freew[P->nfree++] = i;
    }
    if (P->nfree < 1) die("no writable wire");
    for (int t = 0; t < P->ntarg; t++) {
        unsigned long long v;
        if (scanf("%llx", &v) != 1) die("bad target line");
        if (v & ~P->full) die("target has bits beyond npts");
        P->targ[t] = v;
    }
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
