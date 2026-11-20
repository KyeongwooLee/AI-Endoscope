"""Correction pipeline with the PAPER's response types (Orgad et al., ICLR 2025, Sec. 5), on GSM8K train (rs30).

Types are labelled on samples 1-15 only (samples 16-30 simulate the fixes), so the paper's K=30 thresholds are halved:
  A refuse  : >= 8 of 15 unparsable            B correct : >= 8 correct
  C stuck   : one wrong answer >= 8 times      D compete : |correct - top wrong| <= 2, both >= 3
  E many    : > 5 distinct answers
Types overlap; one probe per type (one-vs-rest), out-of-fold (5-fold over questions), as in the paper.
Inputs from the 1.5B greedy answer: hidden r_mean L20, hidden at the answer token L20 (paper's position), question-only
q_end L20, confidence. Then: fix rate of each strategy by TRUE type, and the pipeline
  greedy -> P(wrong) flag -> predicted type -> strategy, with the type->strategy map chosen on the fix table.
usage: python pipeline_paper.py <rs_dir> <feat_dir> <gen3b_dir> <probes_npz_from_pipeline_probe>
"""
import collections, glob, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RS, F, G3, PW = sys.argv[1:5]
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6
T = ['A', 'B', 'C', 'D', 'E']
NAME = {'A': 'refuse', 'B': 'mostly correct', 'C': 'same wrong (stuck)', 'D': 'two competing', 'E': 'many answers'}


def types(p, g):
    p = [key(x) for x in p]
    c, none = sum(ok(x, g) for x in p), sum(x is None for x in p)
    w = collections.Counter(x for x in p if x is not None and not ok(x, g))
    top = w.most_common(1)[0][1] if w else 0
    return {'A': none >= 8, 'B': c >= 8, 'C': top >= 8, 'D': abs(c - top) <= 2 and c >= 3 and top >= 3,
            'E': len({x for x in p if x is not None}) > 5}


def vote(ps, g):
    c = collections.Counter(key(x) for x in ps if x is not None)
    return bool(c) and ok(c.most_common(1)[0][0], g)


rows = {r['greedy_id']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
big = {f"train-{r['pid']}-0": r['correct'] for f in glob.glob(f'{G3}/train_*.jsonl') for r in map(json.loads, open(f))}
meta = json.load(open(f'{F}/meta_0.json'))
L20, POS = meta['layers'].index(20), meta['positions']
H, C = np.load(f'{F}/H_0.npy', mmap_mode='r'), np.load(f'{F}/C_0.npy')
mid = [f"{m['split']}-{m['pid']}-{m['sample']}" for m in meta['meta']]
J = [j for j, i in enumerate(mid) if i in rows and i in big and meta['meta'][j]['format_ok']]
ids = [mid[j] for j in J]
J = np.array(J)
X = {'hidden r_mean': np.asarray(H[J, POS.index('r_mean'), L20], dtype=np.float32),
     'hidden ans (paper)': np.asarray(H[J, POS.index('ans'), L20], dtype=np.float32),
     'question q_end': np.asarray(H[J, POS.index('q_end'), L20], dtype=np.float32),
     'confidence': C[J].astype(np.float32)}
lab = [types(rows[i]['preds'][:15], rows[i]['gold']) for i in ids]
Y = {t: np.array([l[t] for l in lab], dtype=int) for t in T}
wrong = np.array([not rows[i]['greedy_correct'] for i in ids], dtype=int)
n = len(ids)
print(f'questions {n}; greedy wrong {wrong.mean():.1%}')
print('type shares (overlapping), all / among greedy-wrong: ' +
      '  '.join(f'{t} {Y[t].mean():.1%}/{Y[t][wrong == 1].mean():.1%}' for t in T))
combo = collections.Counter(' + '.join(t for t in T if l[t]) or 'none' for l, w in zip(lab, wrong) if w)
print('combinations among greedy-wrong: ' + '  '.join(f'{k} {v}' for k, v in combo.most_common(8)))
TT = [t for t in T if Y[t].sum() >= 30]


def model(d, m):
    bigd = d > 256
    return make_pipeline(StandardScaler(), *([PCA(min(256, m - 1), random_state=0)] if bigd else []),
                         LogisticRegression(C=0.05 if bigd else 1.0, max_iter=3000, class_weight='balanced'))


def fold(name, t, tr, te):
    return name, t, te, model(X[name].shape[1], len(tr)).fit(X[name][tr], Y[t][tr]).predict_proba(X[name][te])[:, 1]


splits = {t: list(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, Y[t] * 2 + wrong)) for t in TT}
with parallel_backend('loky', inner_max_num_threads=4):
    res = Parallel(n_jobs=30)(delayed(fold)(nm, t, tr, te) for nm in X for t in TT for tr, te in splits[t])
P = {}
for nm, t, te, p in res:
    P.setdefault((nm, t), np.zeros(n))[te] = p

print('\n## one-vs-rest AUROC per type (out-of-fold): all questions | greedy-wrong questions only')
print(f"{'type':24s}" + ''.join(f'{nm:>26s}' for nm in X))
for t in TT:
    cells = []
    for nm in X:
        s = P[(nm, t)]
        w = wrong == 1
        aw = roc_auc_score(Y[t][w], s[w]) if 0 < Y[t][w].sum() < w.sum() else float('nan')
        cells.append(f'{roc_auc_score(Y[t], s):.3f} | {aw:.3f}')
    print(f'{t} {NAME[t]:22s}' + ''.join(f'{c:>26s}' for c in cells))

# fix outcomes from samples 16-30 and 3B
S = {'keep': 1.0 - wrong}
S['restart'] = np.array([ok(rows[i]['preds'][15], rows[i]['gold']) for i in ids], float)
S['vote5'] = np.array([vote(rows[i]['preds'][15:20], rows[i]['gold']) for i in ids], float)
S['vote15'] = np.array([vote(rows[i]['preds'][15:], rows[i]['gold']) for i in ids], float)
S['big'] = np.array([big[i] for i in ids], float)
COST = {'keep': 0, 'restart': 1, 'vote5': 5, 'vote15': 15, 'big': 2}
print('\n## fix rate by TRUE type, greedy-wrong questions')
print(f"{'type':24s} {'n':>5s}" + ''.join(f'{s:>9s}' for s in ('restart', 'vote5', 'vote15', 'big')))
for t in TT + ['none']:
    m = (wrong == 1) & ((Y[t] == 1) if t != 'none' else np.all([Y[u] == 0 for u in TT], 0))
    print(f'{t} {NAME.get(t, "no type"):22s} {m.sum():5d}' + ''.join(f'{S[s][m].mean():9.1%}' for s in ('restart', 'vote5', 'vote15', 'big')))

# pipeline: flag by hidden P(wrong) from pipeline_probe; route by predicted types (hidden r_mean probes)
Z = np.load(PW)
zrow = {i: k for k, i in enumerate(Z['ids'])}
pw = Z['hidden_wrong'][[zrow[i] for i in ids], 1]


def route(pred, rule):
    """pred: dict type -> bool array. rule: list of (type, strategy) checked in order; default last."""
    plan = np.array([rule[-1][1]] * n, dtype=object)
    done = np.zeros(n, bool)
    for t, s in rule[:-1]:
        m = pred[t] & ~done
        plan[m] = s; done |= m
    return plan


def run(flag, plan):
    acc = np.where(flag, [S[p][j] for j, p in enumerate(plan)], S['keep'])
    return acc.mean(), (1 + np.where(flag, [COST[p] for p in plan], 0)).mean()


pred = {t: P[('hidden r_mean', t)] >= 0.5 for t in TT}
true = {t: Y[t] == 1 for t in TT}
RULES = {'C or E -> 3B, else vote5': [('C', 'big'), ('E', 'big'), (None, 'vote5')],
         'C -> 3B, E -> vote15, else vote5': [('C', 'big'), ('E', 'vote15'), (None, 'vote5')],
         'B -> vote5, else 3B': [('B', 'vote5'), (None, 'big')]}
for q in (0.25, 0.40):
    fl = pw >= np.quantile(pw, 1 - q)
    print(f'\n## pipeline, flag top {q:.0%} by hidden P(wrong) (recall of wrong {fl[wrong == 1].mean():.0%})')
    print(f"{'policy':58s} {'accuracy':>9s} {'cost':>6s}")
    base = [('greedy only', run(np.zeros(n, bool), ['keep'] * n)), ('all 3B', (S['big'].mean(), 2.0)),
            ('detect -> vote15', run(fl, ['vote15'] * n)), ('detect -> 3B', run(fl, ['big'] * n))]
    for nm, (a, c) in base:
        print(f'{nm:58s} {a:9.1%} {c:6.2f}')
    for nm, rule in RULES.items():
        for tag, pr in (('predicted', pred), ('TRUE', true)):
            a, c = run(fl, route(pr, rule))
            print(f'detect -> {tag:9s} type -> {nm:30s} {a:9.1%} {c:6.2f}')
