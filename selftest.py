"""Self-tests of a search binary (CPU harness or CUDA worker).  Exits non-zero on
ANY failure; the worker refuses to search unless these pass.
  1. fitness equivalence: binary fitness == independent Python fitness on random
     circuits (with a read-only wire, with no-op slots, mixed constants);
  2. search sanity: the binary recovers planted solutions (a random 6-step circuit's
     final wires as targets) and every reported solution passes verify()."""
import random, subprocess, sys
from verify import fitness_bruteforce, verify

def problem_text(mode, npts, init, ro, targets, K, maxform):
    lines = [f'{mode} {npts} {len(init)} {K} {maxform} {len(targets)}']
    lines += [f'{1 if i in ro else 0} {v:x}' for i, v in enumerate(init)]
    lines += [f'{t:x}' for t in targets]
    return '\n'.join(lines) + '\n'

def rand_step(rng, m, ro, maxform):
    t = rng.choice([i for i in range(m) if i not in ro])
    def form():
        while True:
            k = rng.randint(1, maxform); mk = 0
            for _ in range(k): mk |= 1 << rng.randrange(m)
            mk &= ~(1 << t)
            if mk: return mk
    am, bm = form(), form()
    if rng.random() < 0.1: bm, ac, bc = am, 0, 1       # no-op slot
    else: ac, bc = rng.randint(0, 1), rng.randint(0, 1)
    return (t, am, ac, bm, bc)

def base_problem(npts=64):
    m, ro = 9, {5}
    init = [sum(1 << p for p in range(npts) if p >> i & 1) for i in range(6)] + [0, 0, 0]
    return m, ro, init

def test_fitness(binary, n=300):
    rng = random.Random(1); m, ro, init = base_problem(); K = 10
    targets = [rng.getrandbits(64) for _ in range(3)]
    circs = [[rand_step(rng, m, ro, 3) for _ in range(K)] for _ in range(n)]
    txt = problem_text('check', 64, init, ro, targets, K, 3) + f'{n}\n'
    txt += ''.join(f'{t} {am} {ac} {bm} {bc}\n' for c in circs for t, am, ac, bm, bc in c)
    out = run(binary, txt, 600)
    got = [int(l.split()[1]) for l in out.splitlines() if l.startswith('FIT')]
    if len(got) != n: fail(f'check mode returned {len(got)} values for {n} circuits')
    exp = [fitness_bruteforce(init, c, targets, 64) for c in circs]
    bad = [i for i in range(n) if got[i] != exp[i]]
    if bad: fail(f'fitness mismatch on {len(bad)}/{n} circuits, e.g. #{bad[0]}: binary {got[bad[0]]} vs python {exp[bad[0]]}')
    print(f'  fitness equivalence: {n}/{n} circuits identical', flush=True)

def test_planted(binary, search_params, trials=4):
    rng = random.Random(7); m, ro, init = base_problem(); ok = 0
    for tr in range(trials):
        steps = [rand_step(rng, m, ro, 2) for _ in range(6)]
        from verify import replay
        fin = replay(init, steps, (1 << 64) - 1)
        targets = [fin[6] ^ fin[0], fin[7], fin[8] ^ fin[6]]
        txt = problem_text('search', 64, init, ro, targets, 6, 3) + search_params + '\n'
        out = run(binary, txt, 900)
        sols = [l for l in out.splitlines() if l.startswith('SOL')]
        for l in sols:
            st = [tuple(map(int, x.split(','))) for x in l.split()[1:]]
            outs, err = verify(init, st, targets, ro, 3, 64)
            if err: fail(f'binary reported an INVALID solution: {err}')
        ok += bool(sols)
        print(f'  planted instance {tr}: {len(sols)} verified solutions', flush=True)
    if ok < trials: fail(f'recovered only {ok}/{trials} easy planted instances')

def run(binary, txt, timeout):
    p = subprocess.run(binary, input=txt, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0: fail(f'{binary} exited {p.returncode}: {p.stderr.strip()[-300:]}')
    return p.stdout

def fail(msg):
    print('SELFTEST FAILED:', msg, flush=True); sys.exit(1)

if __name__ == '__main__':
    binary = sys.argv[1:]
    params = '256 200000 2000 11 4 3.0 0.05 300'
    print('self-test of', ' '.join(binary), flush=True)
    test_fitness(binary); test_planted(binary, params)
    print('SELFTEST PASSED', flush=True)
