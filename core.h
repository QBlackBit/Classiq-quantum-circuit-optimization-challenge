/*
 * core.h -- search core shared by the CUDA worker (anneal_gpu.cu) and the CPU
 * harness (core_test.c).  Everything the search does lives here, so the same code
 * is unit-tested on the CPU and run on the GPU.
 *
 * Model: m wires hold Boolean functions of the inputs (truth tables over npts <= 128
 * points, two uint64 words each: tt_t).  A step writes
 *     w[t] ^= (XOR_{i in A} w[i] ^ ac) & (XOR_{i in B} w[i] ^ bc)
 * (t not in A or B, t not read-only).  A == B with ac != bc is a no-op slot.
 * Fitness = sum over targets of the Hamming distance to the nearest element of the
 * affine span of the final wires; 0 means every target is an XOR of wires (+const).
 */
#ifndef QBB_CORE_H
#define QBB_CORE_H
#include <math.h>
#include <stdint.h>
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#include <intrin.h>
#endif

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

#define MAXW 12
#define MAXK 48
#define MAXT 8

typedef struct { uint64_t lo, hi; } tt_t;          /* truth table: points 0-63 in lo, 64-127 in hi */

typedef struct {
    int npts, m, K, maxform, ntarg, nfree;
    int K1;                 /* correctness stage uses steps [0, K1); the depth stage uses all K */
    tt_t full;
    tt_t init[MAXW];
    tt_t targ[MAXT];
    int ro[MAXW];
    int freew[MAXW];
} problem_t;

/* step packing: t 4 bits | A mask 12 bits | ac 1 | B mask 12 bits | bc 1 */
HD uint32_t st_pack(int t, int am, int ac, int bm, int bc) {
    return (uint32_t)t | ((uint32_t)am << 4) | ((uint32_t)ac << 16) | ((uint32_t)bm << 17) | ((uint32_t)bc << 29);
}
HD int st_t(uint32_t s) { return (int)(s & 15u); }
HD int st_am(uint32_t s) { return (int)((s >> 4) & 0xFFFu); }
HD int st_ac(uint32_t s) { return (int)((s >> 16) & 1u); }
HD int st_bm(uint32_t s) { return (int)((s >> 17) & 0xFFFu); }
HD int st_bc(uint32_t s) { return (int)((s >> 29) & 1u); }

HD int pop64(uint64_t x) {
#if defined(__CUDA_ARCH__)
    return __popcll(x);
#elif defined(_MSC_VER)
    return (int)__popcnt64(x);
#else
    return __builtin_popcountll(x);
#endif
}
HD int pop32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __popc(x);
#elif defined(_MSC_VER)
    return (int)__popcnt(x);
#else
    return __builtin_popcount(x);
#endif
}
HD int ctz32(uint32_t x) {
#if defined(__CUDA_ARCH__)
    return __ffs((int)x) - 1;
#elif defined(_MSC_VER)
    unsigned long idx; _BitScanForward(&idx, x); return (int)idx;
#else
    return __builtin_ctz(x);
#endif
}

HD uint64_t rnd64(uint64_t *s) {            /* xorshift64 (state never 0) */
    uint64_t x = *s;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *s = x;
    return x;
}
HD float urand(uint64_t *s) { return (float)(rnd64(s) >> 40) * (1.0f / 16777216.0f); }

HD tt_t tt_xor(tt_t a, tt_t b) { tt_t r; r.lo = a.lo ^ b.lo; r.hi = a.hi ^ b.hi; return r; }
HD tt_t tt_and(tt_t a, tt_t b) { tt_t r; r.lo = a.lo & b.lo; r.hi = a.hi & b.hi; return r; }
HD int tt_pop(tt_t a) { return pop64(a.lo) + pop64(a.hi); }
HD tt_t tt_zero(void) { tt_t r; r.lo = 0; r.hi = 0; return r; }

HD tt_t form(const tt_t *w, int m, int mask, int cst, tt_t full) {
    tt_t r = tt_zero();
    for (int i = 0; i < m; i++) if ((mask >> i) & 1) r = tt_xor(r, w[i]);
    return cst ? tt_xor(r, full) : r;
}

HD void replay(const problem_t *P, const uint32_t *st, tt_t *w) {
    for (int i = 0; i < P->m; i++) w[i] = P->init[i];
    for (int k = 0; k < P->K; k++) {
        uint32_t s = st[k];
        int am = st_am(s), bm = st_bm(s), ac = st_ac(s), bc = st_bc(s);
        if (am == bm && ac != bc) continue;                  /* no-op slot */
        tt_t a = form(w, P->m, am, ac, P->full), b = form(w, P->m, bm, bc, P->full);
        w[st_t(s)] = tt_xor(w[st_t(s)], tt_and(a, b));
    }
}

/* min Hamming distance of the first nt targets to the affine span of the wires (Gray-code walk) */
HD int fitness_n(const problem_t *P, const tt_t *w, int nt) {
    int best[MAXT];
    for (int t = 0; t < nt; t++) best[t] = tt_pop(P->targ[t]);
    tt_t e = tt_zero();
    uint32_t n = 1u << (P->m + 1);
    for (uint32_t i = 1; i < n; i++) {
        int g = ctz32(i);
        e = tt_xor(e, (g < P->m) ? w[g] : P->full);
        for (int t = 0; t < nt; t++) {
            int d = tt_pop(tt_xor(e, P->targ[t]));
            if (d < best[t]) best[t] = d;
        }
    }
    int tot = 0;
    for (int t = 0; t < nt; t++) tot += best[t];
    return tot;
}
HD int fitness(const problem_t *P, const tt_t *w) { return fitness_n(P, w, P->ntarg); }

HD int random_form(const problem_t *P, uint64_t *rs, int t, int *cst) {
    int msk;
    do {
        msk = 0;
        int k = 1 + (int)(rnd64(rs) % (uint64_t)P->maxform);
        for (int i = 0; i < k; i++) msk |= 1 << (int)(rnd64(rs) % (uint64_t)P->m);
        msk &= ~(1 << t);
    } while (!msk);
    *cst = (int)(rnd64(rs) & 1u);
    return msk;
}
HD int random_target(const problem_t *P, uint64_t *rs) { return P->freew[rnd64(rs) % (uint64_t)P->nfree]; }
HD uint32_t random_step(const problem_t *P, uint64_t *rs) {
    int t = random_target(P, rs), ac, bc;
    int am = random_form(P, rs, t, &ac), bm = random_form(P, rs, t, &bc);
    return st_pack(t, am, ac, bm, bc);
}
HD uint32_t mutate(const problem_t *P, uint64_t *rs, uint32_t s) {
    int t = st_t(s), am = st_am(s), ac = st_ac(s), bm = st_bm(s), bc = st_bc(s);
    int r = (int)(rnd64(rs) % 6u);
    if (r == 0) {                                   /* new target, keep forms legal */
        t = random_target(P, rs);
        am &= ~(1 << t); bm &= ~(1 << t);
        if (!am) am = random_form(P, rs, t, &ac);
        if (!bm) bm = random_form(P, rs, t, &bc);
    } else if (r == 1 || r == 2) {                  /* flip one wire in a form */
        int bit = (int)(rnd64(rs) % (uint64_t)P->m);
        if (bit != t) {
            int nm = (r == 1 ? am : bm) ^ (1 << bit);
            if (nm && pop32((uint32_t)nm) <= P->maxform) { if (r == 1) am = nm; else bm = nm; }
        }
    } else if (r == 3) { ac ^= 1; }
    else if (r == 4) { bc ^= 1; }
    else { return random_step(P, rs); }
    return st_pack(t, am, ac, bm, bc);
}

/* a step is legal: target writable, target not in its forms, forms non-empty and <= maxform */
HD int step_ok(const problem_t *P, uint32_t s) {
    int t = st_t(s), am = st_am(s), bm = st_bm(s);
    if (t >= P->m || P->ro[t]) return 0;
    if ((am >> t) & 1 || (bm >> t) & 1) return 0;
    if (!am || !bm || am >> P->m || bm >> P->m) return 0;
    if (pop32((uint32_t)am) > P->maxform || pop32((uint32_t)bm) > P->maxform) return 0;
    return 1;
}

/* random circuit in slots [0, active); every later slot starts as a no-op */
HD void init_chain(const problem_t *P, uint32_t *st, int *f, uint64_t *rs, uint32_t *it, int active, int nt) {
    tt_t w[MAXW];
    for (int k = 0; k < P->K; k++) st[k] = random_step(P, rs);
    for (int k = active; k < P->K; k++) st[k] = st_pack(st_t(st[k]), st_am(st[k]), 0, st_am(st[k]), 1);  /* no-op slots */
    replay(P, st, w);
    *f = fitness_n(P, w, nt);
    *it = 0;
}

/* up to S Metropolis moves of one chain, mutating only steps [lo, hi) and scoring the first
 * nt targets; returns 1 as soon as that fitness reaches 0.  (lo, hi, nt) = (0, K1, ntarg) is
 * the one-shot search; staging runs (0, K1a, 2) and then (K1a, K1, ntarg). */
HD int run_slice(const problem_t *P, uint32_t *st, int *f, uint64_t *rs, uint32_t *it, uint32_t total,
                 int S, float T0, float T1, int lo, int hi, int nt) {
    tt_t w[MAXW];
    for (int s = 0; s < S && *it < total; s++) {
        float frac = (float)(*it) / (float)total;
        float T = T0 * powf(T1 / T0, frac);
        int j = lo + (int)(rnd64(rs) % (uint64_t)(hi - lo));
        uint32_t old = st[j];
        st[j] = mutate(P, rs, old);
        replay(P, st, w);
        int nf = fitness_n(P, w, nt);
        (*it)++;
        if (nf <= *f || urand(rs) < expf((float)(*f - nf) / T)) {
            *f = nf;
            if (nf == 0) return 1;
        } else {
            st[j] = old;
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------------------
 * Depth model: gate-level ASAP of the standard emission, identical to depth_est in
 * tools/revdepth.c (pivot glue CXs; the Margolus relative-phase Toffoli lowered as
 * u3 cx(b) u3 cx(a) u3 cx(b) u3 on the target; glue undone in reverse; consecutive u3s
 * on a wire fuse; negated controls are free).  Matches the real emitted depth within one
 * layer.
 * ---------------------------------------------------------------------------------- */
HD int st_noop(uint32_t s) { return st_am(s) == st_bm(s) && st_ac(s) != st_bc(s); }
HD uint32_t noop_of(uint32_t s) { return st_pack(st_t(s), st_am(s), 0, st_am(s), 1); }

HD void dm_u3(int *lv, int *lu, int w) { if (!lu[w]) { lv[w]++; lu[w] = 1; } }
HD void dm_cx(int *lv, int *lu, int c, int t) {
    int L = (lv[c] > lv[t] ? lv[c] : lv[t]) + 1;
    lv[c] = L; lv[t] = L; lu[c] = 0; lu[t] = 0;
}

HD int depth_est(const problem_t *P, const uint32_t *st) {
    int lv[MAXW], lu[MAXW], m = P->m;
    for (int i = 0; i < m; i++) { lv[i] = 0; lu[i] = 0; }
    for (int k = 0; k < P->K; k++) {
        uint32_t s = st[k];
        if (st_noop(s)) continue;
        int t = st_t(s), am = st_am(s), bm = st_bm(s);
        if (am == bm) {                                   /* CX-type: t ^= A (^ const) */
            for (int i = 0; i < m; i++) if ((am >> i) & 1) dm_cx(lv, lu, i, t);
            if (st_ac(s)) dm_u3(lv, lu, t);
            continue;
        }
        int pa = -1, pb = -1;
        for (int i = 0; i < m && pa < 0; i++) if (((am >> i) & 1) && !((bm >> i) & 1)) pa = i;
        for (int i = 0; i < m && pa < 0; i++) if ((am >> i) & 1) pa = i;
        int bsrc = bm;
        if ((bsrc >> pa) & 1) { bsrc = (bsrc & ~(1 << pa)) ^ (am & ~(1 << pa)); bsrc |= 1 << pa; }
        for (int i = 0; i < m && pb < 0; i++) if (((bsrc >> i) & 1) && i != pa) pb = i;
        if (pb < 0) continue;                             /* degenerate */
        int ga[MAXW], na = 0, gb[MAXW], nb = 0;
        for (int i = 0; i < m; i++) if (((am >> i) & 1) && i != pa) { dm_cx(lv, lu, i, pa); ga[na++] = i; }
        for (int i = 0; i < m; i++) if (((bsrc >> i) & 1) && i != pb) { dm_cx(lv, lu, i, pb); gb[nb++] = i; }
        dm_u3(lv, lu, t); dm_cx(lv, lu, pb, t); dm_u3(lv, lu, t); dm_cx(lv, lu, pa, t);
        dm_u3(lv, lu, t); dm_cx(lv, lu, pb, t); dm_u3(lv, lu, t);
        for (int j = nb - 1; j >= 0; j--) dm_cx(lv, lu, gb[j], pb);
        for (int j = na - 1; j >= 0; j--) dm_cx(lv, lu, ga[j], pa);
    }
    int d = 0;
    for (int i = 0; i < m; i++) if (lv[i] > d) d = lv[i];
    return d;
}

/* Depth stage: Metropolis on f = W * span_errors + depth with revdepth's moves (swap
 * neighbours, delete = turn a step into a no-op slot, mutate).  Keeps the shallowest
 * error-free circuit this chain has seen in best[] / *bestd. */
HD void run_slice_depth(const problem_t *P, uint32_t *st, int *f, uint64_t *rs, uint32_t *it, uint32_t total,
                        int S, float T0, float T1, int W, uint32_t *best, int *bestd) {
    tt_t w[MAXW];
    for (int s = 0; s < S && *it < total; s++) {
        float T = T0 * powf(T1 / T0, (float)(*it) / (float)total);
        int j = (int)(rnd64(rs) % (uint64_t)P->K), r = (int)(rnd64(rs) % 10u);
        uint32_t oj = st[j], oj1 = (j + 1 < P->K) ? st[j + 1] : 0u;
        int swapped = 0;
        if (r == 0 && j + 1 < P->K) { st[j] = oj1; st[j + 1] = oj; swapped = 1; }
        else if (r == 1) { st[j] = noop_of(oj); }
        else { st[j] = mutate(P, rs, oj); }
        replay(P, st, w);
        int e = fitness(P, w), d = depth_est(P, st), nf = W * e + d;
        (*it)++;
        if (nf <= *f || urand(rs) < expf((float)(*f - nf) / T)) {
            *f = nf;
            if (e == 0 && d < *bestd) { *bestd = d; for (int k = 0; k < P->K; k++) best[k] = st[k]; }
        } else {
            st[j] = oj;
            if (swapped) st[j + 1] = oj1;
        }
    }
}
/* ------------------------------------------------------------------------------------
 * Depth-mode chain state machine, shared by the CPU harness and the GPU kernel:
 *   phase 0: correctness, stage A  (slots [0,K1a), first NA targets)  -- or one-shot
 *   phase 1: correctness, stage B  (slots [K1a,K1), all targets; stage A frozen)
 *   phase 2: depth annealing over all K slots from the correct circuit
 * One call = one slice.  Returns 1 when a depth stage has just finished: its shallowest
 * error-free circuit is in out[] / *outd and the chain has already restarted.
 * ---------------------------------------------------------------------------------- */
typedef struct {
    int K1a, NA, W, S;
    uint32_t it1, it2;
    float T0, T1;
} dparams_t;

HD int d_oneshot(const problem_t *P, const dparams_t *D) { return D->NA >= P->ntarg || D->K1a >= P->K1; }

HD void d_restart(const problem_t *P, const dparams_t *D, uint32_t *st, int *f, uint64_t *rs, uint32_t *it,
                  int *ph, int *bestd) {
    int one = d_oneshot(P, D);
    init_chain(P, st, f, rs, it, one ? P->K1 : D->K1a, one ? P->ntarg : D->NA);
    *ph = 0; *bestd = 1 << 30;
}

HD void d_enter_depth(const problem_t *P, uint32_t *st, int *f, uint32_t *it, int *ph, uint32_t *best, int *bestd) {
    *ph = 2; *it = 0;
    *bestd = depth_est(P, st); *f = *bestd;
    for (int k = 0; k < P->K; k++) best[k] = st[k];
}

HD int chain_step(const problem_t *P, const dparams_t *D, uint32_t *st, int *f, uint64_t *rs, uint32_t *it,
                  int *ph, uint32_t *best, int *bestd, uint32_t *out, int *outd) {
    int one = d_oneshot(P, D);
    if (*ph == 0) {
        int hi = one ? P->K1 : D->K1a, nt = one ? P->ntarg : D->NA;
        if (run_slice(P, st, f, rs, it, D->it1, D->S, D->T0, D->T1, 0, hi, nt)) {
            if (one) d_enter_depth(P, st, f, it, ph, best, bestd);
            else {
                tt_t w[MAXW];
                replay(P, st, w);
                *f = fitness(P, w); *it = 0; *ph = 1;
            }
        } else if (*it >= D->it1) d_restart(P, D, st, f, rs, it, ph, bestd);
    } else if (*ph == 1) {
        if (run_slice(P, st, f, rs, it, D->it1, D->S, D->T0, D->T1, D->K1a, P->K1, P->ntarg))
            d_enter_depth(P, st, f, it, ph, best, bestd);
        else if (*it >= D->it1) d_restart(P, D, st, f, rs, it, ph, bestd);
    } else {
        run_slice_depth(P, st, f, rs, it, D->it2, D->S, D->T0, D->T1, D->W, best, bestd);
        if (*it >= D->it2) {
            for (int k = 0; k < P->K; k++) out[k] = best[k];
            *outd = *bestd;
            d_restart(P, D, st, f, rs, it, ph, bestd);
            return 1;
        }
    }
    return 0;
}
#endif
