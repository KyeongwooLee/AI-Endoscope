"""Zero-generation checks for two correction ideas, using existing samples only.

(1) headroom for any answer SELECTOR: is the correct answer among the 15 T=0.7 samples at all (pass@15)?
(2) VETO vote: when the hidden probe flags the greedy answer as wrong, take the majority of the 15 samples AFTER
    removing the greedy answer (fall back to greedy if nothing is left). Same cost as vote15; uses the detector to
    break exactly the case where the model keeps repeating its own wrong answer.
Types: paper definitions on all 30 rs30 samples (K=30). Fixes: rs07 (15 samples, T=0.7, independent of rs30).
usage: python analyze_veto.py <rs30_dir> <rs07_dir> <gen_dir> <pipe_probes.npz> <gen3b_dir>
"""
import collections, glob, json, sys
import numpy as np

RS, R7, GEN, PZ, G3 = sys.argv[1:6]
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6
rs = {r['greedy_id']: r for f in glob.glob(f'{RS}/rs_*.jsonl') for r in map(json.loads, open(f))}
r7 = {r['greedy_id']: r for f in glob.glob(f'{R7}/rs_*.jsonl') for r in map(json.loads, open(f))}
gp = {f"train-{r['pid']}-0": r['pred'] for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f))
      if r['mode'] == 'greedy'}
big = {f"train-{r['pid']}-0": r['correct'] for f in glob.glob(f'{G3}/train_*.jsonl') for r in map(json.loads, open(f))}
Z = np.load(PZ)
pw_all = dict(zip(Z['ids'], Z['hidden_wrong'][:, 1]))
ids = [i for i in Z['ids'] if i in r7 and i in rs and i in big]
n = len(ids)


def ptype(r):
    p = [key(x) for x in r['preds']]
    c = sum(ok(x, r['gold']) for x in p)
    w = collections.Counter(x for x in p if x is not None and not ok(x, r['gold']))
    top = w.most_common(1)[0][1] if w else 0
    return 'B' if c >= 15 else 'C' if top >= 15 else 'D' if abs(c - top) <= 5 and c >= 5 and top >= 5 else \
        'E' if len({x for x in p if x is not None}) > 10 else 'other'


def vote(ps, exclude=None):
    c = collections.Counter(key(x) for x in ps if x is not None and (exclude is None or key(x) != key(exclude)))
    return c.most_common(1)[0][0] if c else None


gold = np.array([rs[i]['gold'] for i in ids])
wrong = np.array([not rs[i]['greedy_correct'] for i in ids])
T = np.array([ptype(rs[i]) for i in ids])
keep = (~wrong).astype(float)
v15 = np.array([ok(vote(r7[i]['preds']), rs[i]['gold']) for i in ids], float)
v5 = np.array([ok(vote(r7[i]['preds'][:5]), rs[i]['gold']) for i in ids], float)
veto15 = np.array([ok(vote(r7[i]['preds'], gp[i]) if vote(r7[i]['preds'], gp[i]) is not None else gp[i], rs[i]['gold'])
                   for i in ids], float)
veto5 = np.array([ok(vote(r7[i]['preds'][:5], gp[i]) if vote(r7[i]['preds'][:5], gp[i]) is not None else gp[i],
                     rs[i]['gold']) for i in ids], float)
pass15 = np.array([any(ok(x, rs[i]['gold']) for x in r7[i]['preds']) for i in ids], float)
pass5 = np.array([any(ok(x, rs[i]['gold']) for x in r7[i]['preds'][:5]) for i in ids], float)
b3 = np.array([big[i] for i in ids], float)
TY = ['B', 'C', 'D', 'E', 'other']

print(f'questions {n}; greedy wrong {wrong.mean():.1%}')
print('\n## greedy-wrong questions by paper type: fix rate (%)')
print(f"{'type':6s} {'n':>5s} {'vote5':>7s} {'veto5':>7s} {'pass@5':>7s} {'vote15':>7s} {'veto15':>7s} {'pass@15':>8s} {'3B':>6s}")
for t in TY + ['all']:
    m = wrong & ((T == t) if t != 'all' else True)
    print(f'{t:6s} {m.sum():5d} ' + ' '.join(f'{x[m].mean():7.1%}' for x in (v5, veto5, pass5, v15, veto15)) +
          f' {pass15[m].mean():8.1%} {b3[m].mean():6.1%}')
print('\n## greedy-CORRECT questions: answer still correct after (harm if wrongly flagged)')
m = ~wrong
print(f'  vote5 {v5[m].mean():.1%}  veto5 {veto5[m].mean():.1%}  vote15 {v15[m].mean():.1%}  veto15 {veto15[m].mean():.1%}  3B {b3[m].mean():.1%}')

pw = np.array([pw_all[i] for i in ids])
rng = np.random.default_rng(0)
for q in (0.15, 0.25, 0.40):
    fl = pw >= np.quantile(pw, 1 - q)
    print(f'\n## pipeline, flag top {q:.0%} by hidden P(wrong) (recall {fl[wrong].mean():.0%}, precision {wrong[fl].mean():.0%})')
    pol = {'greedy only': (keep, 0), 'detect -> vote5': (v5, 5), 'detect -> VETO vote5': (veto5, 5),
           'detect -> vote15': (v15, 15), 'detect -> VETO vote15': (veto15, 15), 'detect -> 3B': (b3, 2)}
    acc = {k: np.where(fl, s, keep) for k, (s, c) in pol.items()}
    for k, (s, c) in pol.items():
        print(f'  {k:24s} accuracy {acc[k].mean():6.1%}  cost {1 + fl.mean() * c:5.2f}')
    for a, b in (('detect -> VETO vote15', 'detect -> vote15'), ('detect -> VETO vote5', 'detect -> vote5'),
                 ('detect -> VETO vote15', 'detect -> 3B')):
        d = [acc[a][i].mean() - acc[b][i].mean() for i in (rng.integers(0, n, n) for _ in range(1000))]
        print(f'    {a} minus {b}: {np.mean(d) * 100:+.2f}%p [{np.percentile(d, 2.5) * 100:+.2f}, {np.percentile(d, 97.5) * 100:+.2f}]')
