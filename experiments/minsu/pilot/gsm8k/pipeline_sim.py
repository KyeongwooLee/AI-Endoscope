"""Simulate the correction pipeline: greedy answer -> (1) wrong? -> (2) class -> class-specific fix.

Fix outcomes come only from samples 16-30 of rs30 (classes were labelled on samples 1-15) and from the
3B model's greedy answer. Cost unit = one 1.5B generation; the initial greedy answer costs 1; 3B counts as 2.
  restart : 1 new 1.5B sample                    (+1)
  vote5   : majority of 5 new 1.5B samples        (+5)
  vote15  : majority of 15 new 1.5B samples       (+15)
  big     : 3B greedy answer                      (+2)
  abstain : no answer (counted wrong for accuracy; reported separately as coverage / answered accuracy)
usage: python pipeline_sim.py <rs_dir> <probes.npz> <gen3b_dir>
"""
import collections, glob, json, sys
import numpy as np

RS, PR, G3 = sys.argv[1:4]
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6
rows = {r['greedy_id']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
big = {f"train-{r['pid']}-0": r['correct'] for f in glob.glob(f'{G3}/train_*.jsonl') for r in map(json.loads, open(f))
       if r['mode'] == 'greedy'}
Z = np.load(PR)
ids = [i for i in Z['ids'] if i in big]
sel = np.array([i in big for i in Z['ids']])
print(f'questions with probes and 3B answer: {len(ids)}')
wrong, y3 = Z['wrong'][sel], Z['y3'][sel]
CLS = ['easy', 'wobbly', 'stuck']


def vote(ps, g):
    c = collections.Counter(key(x) for x in ps if x is not None)
    return bool(c) and ok(c.most_common(1)[0][0], g)


S = {'keep': 1 - wrong, 'restart': [], 'vote5': [], 'vote15': [], 'big': [], 'abstain': np.zeros(len(ids))}
for i in ids:
    r = rows[i]; B = r['preds'][15:]
    S['restart'].append(ok(B[0], r['gold'])); S['vote5'].append(vote(B[:5], r['gold']))
    S['vote15'].append(vote(B, r['gold'])); S['big'].append(big[i])
S = {k: np.asarray(v, dtype=float) for k, v in S.items()}
COST = {'keep': 0, 'restart': 1, 'vote5': 5, 'vote15': 15, 'big': 2, 'abstain': 0}
print(f'greedy accuracy 1.5B {S["keep"].mean():.1%} | 3B {S["big"].mean():.1%}')

print('\n## fix rate by TRUE class, greedy-wrong questions (what each class needs)')
print(f"{'class':8s} {'n':>5s} " + ''.join(f'{s:>9s}' for s in ('restart', 'vote5', 'vote15', 'big')))
for k, c in enumerate(CLS):
    m = (wrong == 1) & (y3 == k)
    print(f'{c:8s} {m.sum():5d} ' + ''.join(f'{S[s][m].mean():9.1%}' for s in ('restart', 'vote5', 'vote15', 'big')))


def run(flag, plan):
    """flag: bool per question (send to correction); plan: strategy name per question. Returns acc, cost, coverage."""
    acc = np.where(flag, [S[p][j] for j, p in enumerate(plan)], S['keep'])
    cost = 1 + np.where(flag, [COST[p] for p in plan], 0)
    answered = ~(flag & (np.asarray(plan) == 'abstain'))
    return acc, cost, answered


MAPS = {'ours (easy vote5, wobbly vote15, stuck 3B)': ['vote5', 'vote15', 'big'],
        'ours, stuck abstain': ['vote5', 'vote15', 'abstain'],
        'ours, cheap (easy vote5, wobbly 3B, stuck 3B)': ['vote5', 'big', 'big']}
pw = Z['hidden_wrong'][sel][:, 1]
pc = Z['hidden_cls3'][sel].argmax(1)
pw_conf = Z['confidence_wrong'][sel][:, 1]
n = len(ids)
everyone = np.ones(n, bool)


def table(tau_q):
    tau = np.quantile(pw, 1 - tau_q)
    fl = pw >= tau
    fl_conf = pw_conf >= np.quantile(pw_conf, 1 - tau_q)
    out = [('greedy only', run(~everyone, ['keep'] * n)),
           ('all: vote15', run(everyone, ['vote15'] * n)),
           ('all: 3B (replaces 1.5B greedy)', (S['big'], np.full(n, 2.0), everyone)),
           ('detect(conf) -> vote15', run(fl_conf, ['vote15'] * n)),
           ('detect -> restart', run(fl, ['restart'] * n)),
           ('detect -> vote5', run(fl, ['vote5'] * n)),
           ('detect -> vote15', run(fl, ['vote15'] * n)),
           ('detect -> 3B', run(fl, ['big'] * n))]
    for name, mp in MAPS.items():
        out.append((f'detect -> class -> {name}', run(fl, [mp[k] for k in pc])))
        out.append((f'  same, TRUE class', run(fl, [mp[k] for k in y3])))
        out.append((f'  same, TRUE wrong + TRUE class', run(wrong == 1, [mp[k] for k in y3])))
    print(f'\n## flag top {tau_q:.0%} by hidden P(wrong) (recall of wrong {fl[wrong == 1].mean():.0%}, '
          f'precision {wrong[fl].mean():.0%})')
    print(f"{'policy':58s} {'accuracy':>9s} {'cost':>6s} {'answered':>9s} {'acc|answered':>13s}")
    for name, (a, c, ans) in out:
        print(f'{name:58s} {a.mean():9.1%} {c.mean():6.2f} {ans.mean():9.1%} {a[ans].mean():13.1%}')
    return out


for q in (0.15, 0.25, 0.40):
    table(q)

# bootstrap: ours vs detect->vote15 and vs detect->3B at the 25% flag rate
rng = np.random.default_rng(0)
fl = pw >= np.quantile(pw, 0.75)
A = {'ours': run(fl, [MAPS['ours (easy vote5, wobbly vote15, stuck 3B)'][k] for k in pc]),
     'detect->vote15': run(fl, ['vote15'] * n), 'detect->3B': run(fl, ['big'] * n)}
print('\n## bootstrap (1000x, questions), flag 25%: ours minus baseline')
for b in ('detect->vote15', 'detect->3B'):
    da, dc = [], []
    for _ in range(1000):
        i = rng.integers(0, n, n)
        da.append(A['ours'][0][i].mean() - A[b][0][i].mean()); dc.append(A['ours'][1][i].mean() - A[b][1][i].mean())
    print(f'  vs {b:15s} accuracy {np.mean(da):+.2%} [{np.percentile(da, 2.5):+.2%}, {np.percentile(da, 97.5):+.2%}]  '
          f'cost {np.mean(dc):+.2f} [{np.percentile(dc, 2.5):+.2f}, {np.percentile(dc, 97.5):+.2f}]')

# accuracy-cost curve for the plot
curve = {}
for name in ['detect -> vote15', 'detect -> 3B', 'ours']:
    pts = []
    for q in np.linspace(0, 1, 21):
        f = pw >= np.quantile(pw, 1 - q) if q > 0 else np.zeros(n, bool)
        plan = {'detect -> vote15': ['vote15'] * n, 'detect -> 3B': ['big'] * n,
                'ours': [MAPS['ours (easy vote5, wobbly vote15, stuck 3B)'][k] for k in pc]}[name]
        a, c, _ = run(f, plan)
        pts.append((float(q), float(c.mean()), float(a.mean())))
    curve[name] = pts
json.dump(curve, open('types/pipeline_curve.json', 'w'))
print('\ncurve saved to types/pipeline_curve.json')
