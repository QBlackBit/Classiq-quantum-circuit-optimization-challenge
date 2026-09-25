"""Self-tests of a search engine binary (CPU harness or CUDA worker).  Exit code 1 on ANY
failure; the worker refuses to search unless they pass.  Covered, at 64 and 128 points:
  1. fitness equivalence: engine fitness == independent Python fitness on random
     circuits (read-only wire, no-op slots, CX-type steps, mixed constants);
  2. depth-model equivalence: engine depth == independent Python depth model;
  3. search: planted solutions are recovered and every reported circuit verifies;
  4. depth mode: every reported circuit verifies and its depth equals the model."""
import random
import subprocess
import sys

from verify import depth_model, fitness_bruteforce, replay, verify


def problem_text(mode, npts, init, ro, targets, K, maxform):
    lines = [f'{mode} {npts} {len(init)} {K} {maxform} {len(targets)}']
    lines += [f'{1 if i in ro else 0} {v:x}' for i, v in enumerate(init)]
    lines += [f'{t:x}' for t in targets]
    return '\n'.join(lines) + '\n'


def base_problem(npts=64):
    """rows-like (64 points: 6 data wires + 3 ancillas, wire 5 read-only) or
    columns-like (128 points: 6 data + 3 ancillas + a read-only 7th input on wire 9)"""
    var = lambda i: sum(1 << p for p in range(npts) if p >> i & 1)
    if npts == 64:
        return 9, {5}, [var(i) for i in range(6)] + [0, 0, 0]
    return 10, {9}, [var(i) for i in range(6)] + [0, 0, 0] + [var(6)]


def rand_step(rng, m, ro, maxform):
    t = rng.choice([i for i in range(m) if i not in ro])

    def form():
        while True:
            k = rng.randint(1, maxform); mk = 0
            for _ in range(k):
                mk |= 1 << rng.randrange(m)
            mk &= ~(1 << t)
            if mk:
                return mk
    am, bm = form(), form()
    r = rng.random()
    if r < 0.1:
        return (t, am, 0, am, 1)                     # no-op slot
    if r < 0.2:
        c = rng.randint(0, 1)
        return (t, am, c, am, c)                     # CX-type step
    return (t, am, rng.randint(0, 1), bm, rng.randint(0, 1))


def run(binary, txt, timeout):
    p = subprocess.run(binary, input=txt, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        fail(f'{binary} exited {p.returncode}: {p.stderr.strip()[-300:]}')
    return p.stdout


def fail(msg):
    print('SELFTEST FAILED:', msg, flush=True)
    sys.exit(1)


def circuits(rng, m, ro, K, n):
    return [[rand_step(rng, m, ro, 3) for _ in range(K)] for _ in range(n)]


def body(circs):
    return f'{len(circs)}\n' + ''.join(f'{t} {am} {ac} {bm} {bc}\n' for c in circs for t, am, ac, bm, bc in c)


def test_fitness(binary, n=300, npts=64):
    rng = random.Random(1 + npts); m, ro, init = base_problem(npts); K = 10
    targets = [rng.getrandbits(npts) for _ in range(3)]
    cs = circuits(rng, m, ro, K, n)
    out = run(binary, problem_text('check', npts, init, ro, targets, K, 3) + body(cs), 900)
    got = [int(l.split()[1]) for l in out.splitlines() if l.startswith('FIT')]
    if len(got) != n:
        fail(f'check mode returned {len(got)} values for {n} circuits')
    exp = [fitness_bruteforce(init, c, targets, npts) for c in cs]
    bad = [i for i in range(n) if got[i] != exp[i]]
    if bad:
        fail(f'{npts}-point fitness mismatch on {len(bad)}/{n}, e.g. #{bad[0]}: engine {got[bad[0]]} vs python {exp[bad[0]]}')
    print(f'  fitness equivalence ({npts} points): {n}/{n} identical', flush=True)


def test_depth_model(binary, n=300, npts=64):
    """only for engines with a dcheck mode (CPU harness); the GPU engine's depth is
    covered by test_depth_mode, whose reported depths are checked against the model"""
    rng = random.Random(2 + npts); m, ro, init = base_problem(npts); K = 14
    cs = circuits(rng, m, ro, K, n)
    out = run(binary, problem_text('dcheck', npts, init, ro, [1], K, 3) + body(cs), 900)
    got = [int(l.split()[1]) for l in out.splitlines() if l.startswith('DEPTH')]
    if len(got) != n:
        fail(f'dcheck mode returned {len(got)} values for {n} circuits')
    bad = [i for i in range(n) if got[i] != depth_model(cs[i], m)]
    if bad:
        fail(f'{npts}-point depth-model mismatch on {len(bad)}/{n}')
    print(f'  depth-model equivalence ({npts} points): {n}/{n} identical', flush=True)


def planted(npts, seed, nsteps=6):
    rng = random.Random(seed); m, ro, init = base_problem(npts)
    steps = [rand_step(rng, m, ro, 2) for _ in range(nsteps)]
    full = (1 << npts) - 1
    fin = replay(init, steps, full)
    T = [fin[6] ^ fin[0], fin[7] ^ fin[1], fin[8] ^ fin[2]]
    if any(t in (0, full) for t in T):
        return planted(npts, seed + 1000, nsteps)
    return m, ro, init, T


def test_planted(binary, search_params, trials=3, npts=64):
    ok = 0
    for tr in range(trials):
        m, ro, init, T = planted(npts, 7 + tr)
        out = run(binary, problem_text('search', npts, init, ro, T, 6, 3) + search_params + '\n', 1200)
        sols = [l for l in out.splitlines() if l.startswith('SOL ')]
        for l in sols:
            st = [tuple(map(int, x.split(','))) for x in l.split()[1:]]
            _, err = verify(init, st, T, ro, 3, npts)
            if err:
                fail(f'engine reported an INVALID circuit: {err}')
        ok += bool(sols)
        print(f'  planted search ({npts} points) instance {tr}: {len(sols)} verified circuits', flush=True)
    if ok < trials:
        fail(f'recovered only {ok}/{trials} easy planted instances ({npts} points)')


def test_depth_mode(binary, depth_params, npts=64, label=''):
    m, ro, init, T = planted(npts, 71, nsteps=8)
    out = run(binary, problem_text('depth', npts, init, ro, T, 16, 3) + depth_params + '\n', 1200)
    sols = [l for l in out.splitlines() if l.startswith('SOLD ')]
    if not sols:
        fail(f'depth mode {label} found no circuit for an easy planted instance ({npts} points)')
    for l in sols:
        parts = l.split(); d = int(parts[1])
        st = [tuple(map(int, x.split(','))) for x in parts[2:]]
        _, err = verify(init, st, T, ro, 3, npts)
        if err:
            fail(f'depth mode reported an INVALID circuit: {err}')
        if depth_model(st, m) != d:
            fail(f'depth mode reported depth {d}, independent model says {depth_model(st, m)}')
    print(f'  depth mode {label} ({npts} points): {len(sols)} verified circuits, depths {sorted(int(l.split()[1]) for l in sols)[:6]}', flush=True)


def run_all(binary, kind):
    """the full self-test used by the worker; kind = 'cpu' or 'gpu'"""
    if kind == 'gpu':
        search = '4096 200000 64 11 4 3.0 0.05 600'
        depth = '4096 400000 400000 64 5 16 3.0 0.05 20 10 999 300'
    else:
        search = '256 200000 2000 11 4 3.0 0.05 600'
        depth = '40 400000 400000 2000 5 16 3.0 0.05 20 10 999 300'
    for npts in (64, 128):
        test_fitness(binary, npts=npts)
        if kind == 'cpu':
            test_depth_model(binary, npts=npts)
        test_planted(binary, search, npts=npts)
        test_depth_mode(binary, depth + ' 10 3', npts=npts, label='one-shot')     # K1a = K1, NA = all
        test_depth_mode(binary, depth + ' 5 2', npts=npts, label='staged')        # 2 targets in 5 slots first


if __name__ == '__main__':
    kind = sys.argv[1]
    binary = sys.argv[2:]
    print('self-test of', ' '.join(binary), flush=True)
    run_all(binary, kind)
    print('SELFTEST PASSED', flush=True)
