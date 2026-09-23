/*
 * core.h -- search core shared by the CUDA worker (anneal_gpu.cu) and the CPU
 * harness (core_test.c).  Everything the search does lives here, so the same code
 * is unit-tested on the CPU and run on the GPU.
 *
 * Model: m wires hold Boolean functions of the inputs (truth tables over npts <= 64
 * points, one uint64 each).  A step writes
 *     w[t] ^= (XOR_{i in A} w[i] ^ ac) & (XOR_{i in B} w[i] ^ bc)
 * (t not in A or B, t not read-only).  A == B with ac != bc is a no-op slot.
 * Fitness = sum over targets of the Hamming distance to the nearest element of the
 * affine span of the final wires; 0 means every target is an XOR of wires (+const).
 */
#ifndef QBB_CORE_H
#define QBB_CORE_H
#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD static inline
#endif

#define MAXW 12
#define MAXK 32
#define MAXT 8

typedef struct {
    int npts, m, K, maxform, ntarg, nfree;
    uint64_t full;
    uint64_t init[MAXW];
    uint64_t targ[MAXT];
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
#ifdef __CUDA_ARCH__
    return __popcll(x);
#else
    return __builtin_popcountll(x);
#endif
}
HD int pop32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __popc(x);
#else
    return __builtin_popcount(x);
#endif
}
HD int ctz32(uint32_t x) {
#ifdef __CUDA_ARCH__
    return __ffs((int)x) - 1;
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

HD uint64_t form(const uint64_t *w, int m, int mask, int cst, uint64_t full) {
    uint64_t r = 0;
    for (int i = 0; i < m; i++) if ((mask >> i) & 1) r ^= w[i];
    return cst ? (r ^ full) : r;
}

HD void replay(const problem_t *P, const uint32_t *st, uint64_t *w) {
    for (int i = 0; i < P->m; i++) w[i] = P->init[i];
    for (int k = 0; k < P->K; k++) {
        uint32_t s = st[k];
        int am = st_am(s), bm = st_bm(s), ac = st_ac(s), bc = st_bc(s);
        if (am == bm && ac != bc) continue;                  /* no-op slot */
        uint64_t a = form(w, P->m, am, ac, P->full), b = form(w, P->m, bm, bc, P->full);
        w[st_t(s)] ^= (a & b);
    }
}

/* min Hamming distance of every target to the affine span of the wires (Gray-code walk) */
HD int fitness(const problem_t *P, const uint64_t *w) {
    int best[MAXT];
    for (int t = 0; t < P->ntarg; t++) best[t] = pop64(P->targ[t]);
    uint64_t e = 0;
    uint32_t n = 1u << (P->m + 1);
    for (uint32_t i = 1; i < n; i++) {
        int g = ctz32(i);
        e ^= (g < P->m) ? w[g] : P->full;
        for (int t = 0; t < P->ntarg; t++) {
            int d = pop64(e ^ P->targ[t]);
            if (d < best[t]) best[t] = d;
        }
    }
    int tot = 0;
    for (int t = 0; t < P->ntarg; t++) tot += best[t];
    return tot;
}

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

HD void init_chain(const problem_t *P, uint32_t *st, int *f, uint64_t *rs, uint32_t *it) {
    uint64_t w[MAXW];
    for (int k = 0; k < P->K; k++) st[k] = random_step(P, rs);
    replay(P, st, w);
    *f = fitness(P, w);
    *it = 0;
}

/* up to S Metropolis moves of one chain; returns 1 as soon as fitness 0 is reached */
HD int run_slice(const problem_t *P, uint32_t *st, int *f, uint64_t *rs, uint32_t *it, uint32_t total,
                 int S, float T0, float T1) {
    uint64_t w[MAXW];
    for (int s = 0; s < S && *it < total; s++) {
        float frac = (float)(*it) / (float)total;
        float T = T0 * powf(T1 / T0, frac);
        int j = (int)(rnd64(rs) % (uint64_t)P->K);
        uint32_t old = st[j];
        st[j] = mutate(P, rs, old);
        replay(P, st, w);
        int nf = fitness(P, w);
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
#endif
