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
