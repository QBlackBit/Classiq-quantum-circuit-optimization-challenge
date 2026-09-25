"""Independent verification (pure Python, shares no code with core.h).

A proposed circuit is accepted only if every step is legal (writable target, target
not among its own controls, non-empty forms of at most maxform wires) and every
canonical target is an XOR of final wires (plus optionally the constant), found by
exact GF(2) elimination.  `fitness_bruteforce` recomputes the search fitness from
scratch for the self-test that compares the C/CUDA fitness with this one."""


def replay(init, steps, full):
    w = list(init)
    for t, am, ac, bm, bc in steps:
        if am == bm and ac != bc:
            continue
        a = full if ac else 0
        b = full if bc else 0
        for i in range(len(w)):
            if am >> i & 1:
                a ^= w[i]
            if bm >> i & 1:
                b ^= w[i]
        w[t] ^= a & b
    return w


def express(target, wires, full):
    """(sorted wire list, const) with XOR of wires ^ const*full == target, or None."""
    basis = {}                                   # pivot bit -> (value, label set)
    for v, lab in [(full, {'c'})] + [(x, {i}) for i, x in enumerate(wires)]:
        lab = set(lab)
        for p, (bv, bl) in basis.items():
            if v >> p & 1:
                v ^= bv; lab ^= bl
        if v:
            p = v.bit_length() - 1
            for q in list(basis):
                bv, bl = basis[q]
                if bv >> p & 1:
                    basis[q] = (bv ^ v, bl ^ lab)
            basis[p] = (v, lab)
    v, lab = target, set()
    for p, (bv, bl) in basis.items():
        if v >> p & 1:
            v ^= bv; lab ^= bl
    if v:
        return None
    return sorted(x for x in lab if x != 'c'), (1 if 'c' in lab else 0)


def check_steps(steps, m, ro, maxform):
    for t, am, ac, bm, bc in steps:
        if not (0 <= t < m) or t in ro:
            return f'step targets wire {t} (read-only or out of range)'
        for mk in (am, bm):
            if mk <= 0 or mk >> m:
                return 'empty or out-of-range control form'
            if mk >> t & 1:
                return 'target appears in its own control form'
            if bin(mk).count('1') > maxform:
                return 'control form wider than maxform'
        if ac not in (0, 1) or bc not in (0, 1):
            return 'bad constant'
    return None


def verify(init, steps, targets, ro, maxform, npts):
    """-> (outs, None) if valid else (None, reason)."""
    full = (1 << npts) - 1
    err = check_steps(steps, len(init), set(ro), maxform)
    if err:
        return None, err
    fin = replay(init, steps, full)
    for i in ro:
        if fin[i] != init[i]:
            return None, f'read-only wire {i} changed'
    outs = []
    for t in targets:
        e = express(t, fin, full)
        if e is None:
            return None, 'a target is not in the span of the final wires'
        outs.append(e)
    return outs, None


def fitness_bruteforce(init, steps, targets, npts):
    full = (1 << npts) - 1
    fin = replay(init, steps, full)
    span = [0]
    for g in fin + [full]:
        span += [x ^ g for x in span]
    return sum(min(bin(t ^ s).count('1') for s in span) for t in targets)


def depth_model(steps, m):
    """Independent re-implementation of the gate-level ASAP depth model
    (tools/revdepth.c depth_est) for the self-test of the engines."""
    lv, lu = [0] * m, [False] * m

    def u3(w):
        if not lu[w]:
            lv[w] += 1; lu[w] = True

    def cx(c, t):
        L = max(lv[c], lv[t]) + 1
        lv[c] = lv[t] = L; lu[c] = lu[t] = False

    for t, am, ac, bm, bc in steps:
        if am == bm and ac != bc:
            continue
        if am == bm:
            for i in range(m):
                if am >> i & 1: cx(i, t)
            if ac: u3(t)
            continue
        A = [i for i in range(m) if am >> i & 1]
        pa = next((i for i in A if not bm >> i & 1), A[0])
        bsrc = bm
        if bsrc >> pa & 1:
            bsrc = (bsrc & ~(1 << pa)) ^ (am & ~(1 << pa)); bsrc |= 1 << pa
        pb = next((i for i in range(m) if bsrc >> i & 1 and i != pa), None)
        if pb is None:
            continue
        ga = [i for i in range(m) if am >> i & 1 and i != pa]
        gb = [i for i in range(m) if bsrc >> i & 1 and i != pb]
        for i in ga: cx(i, pa)
        for i in gb: cx(i, pb)
        u3(t); cx(pb, t); u3(t); cx(pa, t); u3(t); cx(pb, t); u3(t)
        for i in reversed(gb): cx(i, pb)
        for i in reversed(ga): cx(i, pa)
    return max(lv) if lv else 0
