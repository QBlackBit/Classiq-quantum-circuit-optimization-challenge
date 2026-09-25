"""QBlackBit side-circuit search worker (GPU with CPU fallback).

Searches in-place reversible circuits whose final wires span given target functions
(the "sides" of a two-round phase oracle).  Engine: anneal_gpu (CUDA, massive
independent restarts) or core_test (same algorithm on CPU, one process per core).

Guarantees:
  * refuses to search unless the self-tests pass on the chosen engine
    (fitness equivalence with an independent Python implementation + recovery of
    planted solutions);
  * every reported circuit is re-verified with verify.py (independent code) before
    it is written; an engine that ever reports an invalid circuit stops the run;
  * any error is logged with a traceback and ends the run with a non-zero exit code;
  * progress is saved after every step: re-running resumes where it stopped.

Results: OUT/results/<side key>.json, log: OUT/worker.log, state: OUT/state.json.
"""
import argparse, json, os, random, shutil, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def exe_path(name):
    """the built engine: name.exe on Windows (whatever Python flavour runs this), else name"""
    p = os.path.join(HERE, name)
    return p + '.exe' if os.path.exists(p + '.exe') else p
sys.path.insert(0, HERE)
import selftest
from verify import depth_model, replay, verify

LOG = None

def log(msg):
    line = time.strftime('%Y-%m-%d %H:%M:%S ') + msg
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def gpu_available():
    exe = exe_path('anneal_gpu')
    if not os.path.exists(exe) or shutil.which('nvidia-smi') is None:
        return False
    try:
        return subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=60).returncode == 0
    except Exception:
        return False


import threading
_TIMEOUTS = {'consecutive': 0}
_TLOCK = threading.Lock()
TIMEOUT_MARGIN = int(os.environ.get('QBB_TIMEOUT_MARGIN', '900'))   # seconds allowed beyond the engine's own limit


def run_engine(exe, txt, seconds):
    """Run one engine process.  If it overruns its own time limit by more than the margin
    (the machine slept, or the GPU stalled) the run is discarded with a loud WARNING and the
    search continues; three such timeouts in a row stop the worker with an error."""
    try:
        p = subprocess.run([exe], input=txt, capture_output=True, text=True, timeout=seconds + TIMEOUT_MARGIN)
    except subprocess.TimeoutExpired:
        log(f'WARNING: an engine process exceeded its {seconds}s limit by more than {TIMEOUT_MARGIN}s '
            f'(machine asleep or GPU stalled?) -- its results are discarded, the search continues')
        return None
    if p.returncode != 0:
        raise RuntimeError(f'{exe} exited {p.returncode}: {p.stderr.strip()[-500:]}')
    if 'DONE' not in p.stdout:
        raise RuntimeError(f'{exe} ended without DONE line')
    return p


def note_run(results):
    """count search runs (not processes) in which an engine timed out; 3 in a row -> stop"""
    with _TLOCK:
        _TIMEOUTS['consecutive'] = _TIMEOUTS['consecutive'] + 1 if any(r is None for r in results) else 0
        n = _TIMEOUTS['consecutive']
    if n >= 3:
        raise RuntimeError('3 consecutive runs with engine timeouts: please check the machine / GPU and send out/worker.log')


class Engine:
    def __init__(self, kind, chains, iters, slice_, cpus):
        self.kind, self.chains, self.iters, self.slice, self.cpus = kind, chains, iters, slice_, cpus
        self.exe = exe_path('anneal_gpu' if kind == 'gpu' else 'core_test')

    def selftest(self):
        log(f'self-test of the {self.kind} engine ...')
        selftest.run_all([self.exe], self.kind)   # sys.exit(1) on any failure
        log(f'self-test of the {self.kind} engine PASSED')

    def search(self, init, ro, targets, K, maxform, seconds, seed, maxsol):
        txt_head = selftest.problem_text('search', 64, init, set(ro), targets, K, maxform)
        if self.kind == 'gpu':
            runs = [(txt_head + f'{self.chains} {self.iters} {self.slice} {seed} {maxsol} 3.0 0.05 {seconds}\n')]
        else:  # one sequential-chain process per core, different seeds
            runs = [txt_head + f'1000000 {self.iters} 2000 {seed * 1000 + i} {maxsol} 3.0 0.05 {seconds}\n'
                    for i in range(self.cpus)]
        def one(txt):
            p = run_engine(self.exe, txt, seconds)
            if p is None:
                return [], 1 << 30, True
            sols = [[tuple(map(int, x.split(','))) for x in l.split()[1:]] for l in p.stdout.splitlines() if l.startswith('SOL')]
            best = int([l for l in p.stdout.splitlines() if l.startswith('DONE')][-1].split()[2])
            return sols, best, False
        with ThreadPoolExecutor(max_workers=len(runs)) as ex:
            res = list(ex.map(one, runs))
        note_run([None if r[2] else r for r in res])
        return [s for r in res for s in r[0]], min(r[1] for r in res)


def run_strategy(eng, strat, side, a):
    """-> list of candidate circuits (step tuples from the ORIGINAL init); verified by the caller."""
    init, ro, stage, mf = side['init'], side['ro'], side['search_targets'], side['maxform']
    seed = random.SystemRandom().randrange(1, 2**31)
    if strat[0] == 'oneshot':
        sols, _ = eng.search(init, ro, stage, strat[1], mf, a.per_k_seconds, seed, a.maxsol)
        return sols
    _, k1, k2 = strat                                     # staged: two features first, then all
    first, _ = eng.search(init, ro, stage[:2], k1, mf, a.per_k_seconds, seed, a.stage1_keep)
    out = []
    for i, st1 in enumerate(first[:a.stage1_keep]):
        mid = replay(init, st1, (1 << 64) - 1)
        more, _ = eng.search(mid, ro, stage, k2, mf, max(30, a.per_k_seconds // 2), seed + 7 * (i + 1), a.maxsol)
        out += [list(st1) + list(st2) for st2 in more]
        if out:
            break
    return out

def depth_search(eng, side, K1a, NA, seconds, seed, rep, a):
    """depth mode on one side -> list of (depth, steps); every circuit re-verified by the caller"""
    head = selftest.problem_text('depth', side['npts'], side['init'], set(side['ro']), side['search_targets'],
                                 side['K'], side['maxform'])
    if eng.kind == 'gpu':
        runs = [head + f"{a.chains} {a.it1} {a.it2} {a.slice} {seed} {a.maxsol} 3.0 0.05 20 {side['K1']} {rep} {seconds} {K1a} {NA}\n"]
    else:
        runs = [head + f"1000000 {a.it1} {a.it2} 2000 {seed * 1000 + i} {a.maxsol} 3.0 0.05 20 {side['K1']} {rep} {seconds} {K1a} {NA}\n"
                for i in range(eng.cpus)]

    def one(txt):
        p = run_engine(eng.exe, txt, seconds)
        if p is None:
            return None
        out = []
        for l in p.stdout.splitlines():
            if l.startswith('SOLD '):
                parts = l.split()
                out.append((int(parts[1]), [tuple(map(int, x.split(','))) for x in parts[2:]]))
        return out
    with ThreadPoolExecutor(max_workers=len(runs)) as ex:
        res = list(ex.map(one, runs))
    note_run(res)
    return [x for r in res if r is not None for x in r]


def depth_loop(eng, P, a, state, save, t_end):
    """round-robin over code sides; keeps the shallowest verified circuits per side"""
    order = [k for k in P['order'] if (a.only == 'all' or P['sides'][k]['kind'] == a.only.rstrip('s'))
             and P['sides'][k]['rank'] < a.top]
    best = state.setdefault('best', {})
    log(f'depth mode: {len(order)} sides ({a.only}), {a.per_run_seconds}s per run')
    for rnd in range(10 ** 9):
        for n, key in enumerate(order):
            if time.time() > t_end:
                log('time limit reached; stopping (re-run to resume)'); return 0
            side = P['sides'][key]; m = len(side['init'])
            staged = (rnd + n) % 2 == 0
            K1a, NA = (min(12, side['K1']), 2) if staged else (side['K1'], len(side['targets']))
            rep = best.get(key, 999)
            seed = random.SystemRandom().randrange(1, 2**31)
            t0 = time.time()
            found = depth_search(eng, side, K1a, NA, a.per_run_seconds, seed, rep, a)
            good = []
            for d, st in found:
                outs, err = verify(side['init'], st, side['targets'], side['ro'], side['maxform'], side['npts'])
                if err:
                    raise RuntimeError(f'engine returned an INVALID circuit for {key}: {err}')
                if depth_model(st, m) != d:
                    raise RuntimeError(f'engine reported depth {d} for {key}, independent model says {depth_model(st, m)}')
                good.append({'depth': d, 'steps': to_lists(st, m), 'outs': [[list(x), c] for x, c in outs]})
            rp = os.path.join(a.out, 'results', key + '.json')
            old = json.load(open(rp))['solutions'] if os.path.exists(rp) else []
            seen = {json.dumps(s['steps']) for s in old}
            allsol = sorted(old + [g for g in good if json.dumps(g['steps']) not in seen], key=lambda s: s['depth'])[:32]
            if allsol:
                json.dump({'key': key, 'kind': side['kind'], 'masks': side['masks'], 'npts': side['npts'],
                           'init': side['init'], 'ro': side['ro'], 'targets': side['targets'], 'solutions': allsol},
                          open(rp + '.tmp', 'w'))
                os.replace(rp + '.tmp', rp)
                if allsol[0]['depth'] < best.get(key, 999):
                    best[key] = allsol[0]['depth']
                    log(f'*** NEW BEST {key}: depth {best[key]} ***')
            save()
            log(f'round {rnd} {key} ({"staged" if staged else "one-shot"}): {len(good)} verified circuits, '
                f'best {best.get(key, "-")} ({time.time()-t0:.0f}s)')
    return 0


def to_lists(steps, m):
    return [[t, [i for i in range(m) if am >> i & 1], ac, [i for i in range(m) if bm >> i & 1], bc] for t, am, ac, bm, bc in steps]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--problems', default=os.path.join(HERE, 'problems.json'),
                    help='default: the single-comparator code sides; problems_twosplit.json = the 4+4 search')
    ap.add_argument('--engine', choices=['auto', 'gpu', 'cpu'], default='auto')
    ap.add_argument('--hours', type=float, default=1000.0, help='stop after this many hours')
    ap.add_argument('--k-list', default='20,24', help='one-shot step counts to try, in order')
    ap.add_argument('--stage1-k', default='8,12', help='staged search: step counts for the first two features')
    ap.add_argument('--stage2-k', default='10,14', help='staged search: extra steps for all features')
    ap.add_argument('--stage1-keep', type=int, default=4, help='stage-1 circuits continued into stage 2')
    ap.add_argument('--per-k-seconds', type=int, default=240)
    ap.add_argument('--chains', type=int, default=65536, help='GPU chains')
    ap.add_argument('--iters', type=int, default=400000, help='moves per chain before a restart')
    ap.add_argument('--slice', type=int, default=64, help='GPU moves per kernel launch')
    ap.add_argument('--maxsol', type=int, default=64)
    ap.add_argument('--unsafe-skip-selftest', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--only', choices=['all', 'rows', 'cols'], default='all', help='depth mode: which code sides')
    ap.add_argument('--top', type=int, default=6, help='depth mode: the N most promising codes of each kind')
    ap.add_argument('--per-run-seconds', type=int, default=900, help='depth mode: seconds per side per run')
    ap.add_argument('--it1', type=int, default=2000000, help='depth mode: moves per correctness attempt')
    ap.add_argument('--it2', type=int, default=1000000, help='depth mode: moves per depth-annealing stage')
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'results'), exist_ok=True)
    global LOG
    LOG = os.path.join(a.out, 'worker.log')
    t_end = time.time() + a.hours * 3600
    log(f'worker start: {" ".join(sys.argv[1:])}')
    P = json.load(open(a.problems))                       # fail fast on a missing / broken file
    if P.get('mode') == 'depth':
        assert all(k in P['sides'] for k in P['order']), 'bad order list'
        log(f'problems: depth mode, {len(P["sides"])} sides')
    else:
        for sp in P['splits']:
            assert len(sp['sides']) == 4 and all(k in P['sides'] for k in sp['sides']), f'bad split {sp}'
        log(f'problems: {len(P["splits"])} splits, {len(P["sides"])} sides')
    kind = a.engine
    if kind == 'auto':
        kind = 'gpu' if gpu_available() else 'cpu'
        if kind == 'cpu':
            log('WARNING: no usable NVIDIA GPU found (anneal_gpu + nvidia-smi); falling back to CPU')
    eng = Engine(kind, a.chains, a.iters, a.slice, os.cpu_count() or 4)
    try:
        if a.unsafe_skip_selftest:
            log('!!! SELF-TEST SKIPPED (development only; never do this for real runs) !!!')
        else:
            eng.selftest()
    except SystemExit:
        log(f'!!! SELF-TEST FAILED for the {kind} engine (details above in the console output) !!!')
        if kind == 'gpu' and a.engine == 'auto':
            log('!!! falling back to the CPU engine; please send out/worker.log so the GPU problem can be fixed !!!')
            eng = Engine('cpu', a.chains, a.iters, a.slice, os.cpu_count() or 4)
            try:
                eng.selftest()
            except SystemExit:
                log('!!! CPU self-test failed too; stopping !!!'); return 1
        else:
            return 1
    st_path = os.path.join(a.out, 'state.json')
    state = json.load(open(st_path)) if os.path.exists(st_path) else {'side': {}}
    def save():
        tmp = st_path + '.tmp'; json.dump(state, open(tmp, 'w'), indent=1); os.replace(tmp, st_path)
    if P.get('mode') == 'depth':
        return depth_loop(eng, P, a, state, save, t_end)
    klist = [int(k) for k in a.k_list.split(',') if k]
    s1list = [int(k) for k in a.stage1_k.split(',') if k]
    s2list = [int(k) for k in a.stage2_k.split(',') if k]
    for sp in P['splits']:
        keys = sp['sides']
        if any(state['side'].get(k, {}).get('status') == 'failed' for k in keys):
            continue
        # hardest side first so a dead split is discovered early
        for key in sorted(keys, key=lambda k: -P['sides'][k]['difficulty']):
            S = state['side'].setdefault(key, {'status': 'pending', 'tried': []})
            if S['status'] in ('found', 'failed'):
                if S['status'] == 'failed': break
                continue
            side = P['sides'][key]
            init, ro, targets, stage = side['init'], side['ro'], side['targets'], side['search_targets']
            m = len(init)
            plan = [('oneshot', K) for K in klist] + [('staged', k1, k2) for k1 in s1list for k2 in s2list]
            for strat in plan:
                tag = '/'.join(map(str, strat))
                if tag in S['tried']:
                    continue
                if time.time() > t_end:
                    log('time limit reached; stopping (re-run to resume)'); return 0
                t0 = time.time()
                cands = run_strategy(eng, strat, side, a)
                good = []
                for st in cands:
                    outs, err = verify(init, st, targets, ro, side['maxform'], 64)
                    if err:
                        raise RuntimeError(f'engine returned an INVALID circuit for {key} ({tag}): {err}')
                    good.append({'strategy': tag, 'K': len(st), 'steps': to_lists(st, m), 'outs': [[list(w), c] for w, c in outs]})
                S['tried'].append(tag)
                log(f'split {sp["rank"]} side {key}: {tag} -> {len(good)} verified circuits ({time.time()-t0:.0f}s)')
                if good:
                    rp = os.path.join(a.out, 'results', key + '.json')
                    old = json.load(open(rp))['solutions'] if os.path.exists(rp) else []
                    seen = {json.dumps(x['steps']) for x in old}
                    new = old + [g for g in good if json.dumps(g['steps']) not in seen]
                    json.dump({'key': key, 'reg': side['reg'], 'npts': 64, 'init': init, 'ro': ro, 'targets': targets,
                               'solutions': new}, open(rp + '.tmp', 'w'))
                    os.replace(rp + '.tmp', rp)
                    S['status'] = 'found'; save()
                    log(f'FOUND side {key}: {len(new)} circuits saved')
                    break
                save()
            if S['status'] != 'found':
                S['status'] = 'failed'; save()
                log(f'side {key} not found by any strategy; split {sp["rank"]} dropped')
                break
        if all(state['side'].get(k, {}).get('status') == 'found' for k in keys):
            log(f'*** SPLIT {sp["rank"]} COMPLETE: all four sides found -- please send the out/ folder ***')
    log('all splits processed')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException:
        if LOG:
            log('FATAL ERROR:\n' + traceback.format_exc())
        else:
            traceback.print_exc()
        sys.exit(2)
