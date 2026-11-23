"""Analyse steer.py output: paired comparison of every condition against 'base' (same HF greedy pipeline, no steering).

- fix rate on greedy-wrong questions, keep rate on greedy-correct ones, paired bootstrap CI of the difference vs base
  and vs the random direction at the same strength
- fix rate by the paper's response type (rs30, all 30 samples, K=30 thresholds) to see whether steering reaches C
- reference: one T=0.7 re-sample (rs07 first sample) on the same questions
usage: python analyze_steer.py <steer_dir> <rs30_dir> [<rs07_dir>]
"""
import collections, glob, json, os, sys
import numpy as np

SD, RS = sys.argv[1], sys.argv[2]
R7 = sys.argv[3] if len(sys.argv) > 3 else None
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6
C = {os.path.basename(f)[:-6]: {r['pid']: r for r in map(json.loads, open(f))} for f in glob.glob(f'{SD}/*.jsonl')}
order = ['base'] + sorted([c for c in C if c.startswith('steer')], key=lambda c: float(c.split('_')[1])) + \
        sorted([c for c in C if c.startswith('random')], key=lambda c: float(c.split('_')[1]))
pids = sorted(C['base'])
grp = np.array([C['base'][p]['group'] for p in pids])
W, O = grp == 'wrong', grp == 'ok'
print(f'questions: {W.sum()} greedy-wrong, {O.sum()} greedy-correct; conditions: {", ".join(order)}')
rows = {r['pid']: r for f in glob.glob(f'{RS}/rs_*.jsonl') for r in map(json.loads, open(f))}


def ptype(r):
    p = [key(x) for x in r['preds']]
    c = sum(ok(x, r['gold']) for x in p)
    w = collections.Counter(x for x in p if x is not None and not ok(x, r['gold']))
    top = w.most_common(1)[0][1] if w else 0
    return 'B' if c >= 15 else 'C' if top >= 15 else 'D' if abs(c - top) <= 5 and c >= 5 and top >= 5 else \
        'E' if len({x for x in p if x is not None}) > 10 else 'other'


T = np.array([ptype(rows[p]) for p in pids])
A = {c: np.array([C[c][p]['correct'] for p in pids], float) for c in order}
same = {c: np.array([C[c][p]['same_as_orig'] for p in pids], float) for c in order}
noans = {c: np.array([C[c][p]['pred'] is None for p in pids], float) for c in order}
rng = np.random.default_rng(0)
BW = [rng.choice(np.where(W)[0], W.sum()) for _ in range(2000)]
BO = [rng.choice(np.where(O)[0], O.sum()) for _ in range(2000)]


def ci(x, y, B):
    d = [x[b].mean() - y[b].mean() for b in B]
    return f'{np.mean(d) * 100:+.1f} [{np.percentile(d, 2.5) * 100:+.1f}, {np.percentile(d, 97.5) * 100:+.1f}]'


print('\n## all conditions (percent); differences are paired vs base, 95% bootstrap CI in %p')
print(f"{'condition':11s} {'fix wrong':>9s} {'vs base':>20s} {'same old':>9s} {'keep ok':>8s} {'vs base':>20s} {'no ans':>7s}")
for c in order:
    print(f'{c:11s} {A[c][W].mean():9.1%} {ci(A[c], A["base"], BW) if c != "base" else "":>20s} '
          f'{same[c][W].mean():9.1%} {A[c][O].mean():8.1%} {ci(A[c], A["base"], BO) if c != "base" else "":>20s} '
          f'{noans[c].mean():7.1%}')
print('\n## steering vs random direction at the same strength (fix wrong / keep ok, %p)')
for c in order:
    if c.startswith('steer'):
        r = 'random_' + c.split('_')[1]
        if r in A:
            print(f'  {c} - {r}: fix {ci(A[c], A[r], BW)}   keep {ci(A[c], A[r], BO)}')
print('\n## fix rate on greedy-wrong questions by paper response type (rs30, K=30)')
types = ['B', 'C', 'D', 'E', 'other']
print(f"{'condition':11s}" + ''.join(f'{t + " (n=" + str(((T == t) & W).sum()) + ")":>13s}' for t in types))
for c in order:
    print(f'{c:11s}' + ''.join(f'{A[c][(T == t) & W].mean():13.1%}' for t in types))
if R7:
    r7 = {r['pid']: r for f in glob.glob(f'{R7}/rs_*.jsonl') for r in map(json.loads, open(f))}
    if all(p in r7 for p in pids):
        re = np.array([ok(r7[p]['preds'][0], rows[p]['gold']) for p in pids], float)
        print(f'\nreference: one T=0.7 re-sample  fix wrong {re[W].mean():.1%}  keep ok {re[O].mean():.1%}  | by type ' +
              '  '.join(f'{t} {re[(T == t) & W].mean():.1%}' for t in types))
# population estimate if applied to every question: greedy acc 82.3% (rs30 greedy)
g = np.mean([r['greedy_correct'] for r in rows.values()])
print(f'\n## estimated accuracy if applied to EVERY question (greedy acc {g:.1%}): '
      + '  '.join(f'{c} {g * A[c][O].mean() + (1 - g) * A[c][W].mean():.1%}' for c in order))
