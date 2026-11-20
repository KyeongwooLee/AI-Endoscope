"""Which hidden state predicts the answer-distribution pattern best? All positions x layers of extract_gen.py.

Labels as in probe_pattern.py (all restarts in the given fix dirs). Tasks: stuck vs spread (AUROC) and the 4-way pattern
(balanced accuracy). Each (position, layer) is scored with 5-fold CV x 3 seeds. Because picking the best of 28 cells
on the same folds overstates it, a nested estimate is also given: in each outer fold the cell is chosen by inner CV on
the training part only, then scored on the held-out part.
usage: python probe_pattern_sweep.py <feat_dir> <gen_dir> <fix_dir> [<fix_dir> ...]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, GEN, FIXES = sys.argv[1], sys.argv[2], sys.argv[3:]
pred, gold = collections.defaultdict(list), {}
for D in FIXES:
    for f in sorted(glob.glob(f'{D}/fix_*.jsonl')) + sorted(glob.glob(f'{D}/more/fix_*.jsonl')):
        for r in map(json.loads, open(f)):
            if r['cond'] == 'restart':
                pred[r['id']] += r['preds']
                gold[r['id']] = r['gold']
key = lambda x: None if x is None else round(x, 4)


def pattern(i):  # as in probe_pattern.py
    p = [key(x) for x in pred[i]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / len(p) if wrong else 0)
    return 'dominant' if c >= .5 else 'stuck' if w >= .5 else 'competing' if c >= .2 and w >= .2 else \
        'spread' if len(set(p)) > 10 else 'other'


P4 = ['dominant', 'stuck', 'competing', 'spread']
ids = sorted(i for i in pred if len(pred[i]) == 32 and pattern(i) in P4)
lab = np.array([P4.index(pattern(i)) for i in ids])
print('items ' + str(len(ids)) + ': ' + '  '.join(f'{n} {(lab == k).sum()}' for k, n in enumerate(P4)), flush=True)
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for d in metas for m in d['meta'] if m['format_ok']]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
Hall = np.asarray(np.load(f'{F}/H_all.npy', mmap_mode='r')[np.array([row[i] for i in ids])], dtype=np.float32)
CELLS = [(p, l) for p in range(len(POS)) for l in range(len(LAYERS))]
TASKS = {'stuck vs spread': (np.isin(lab, [1, 3]), (lab == 1).astype(int)), '4-way': (np.ones(len(ids), bool), lab)}


def model(n):
    return make_pipeline(StandardScaler(), PCA(min(256, n - 1), random_state=0),
                         LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced'))


def score(task, y, s):
    return balanced_accuracy_score(y, s.argmax(1)) if task == '4-way' else roc_auc_score(y, s)


def cv(task, X, y, seed, k=5):
    s = np.zeros((len(y), 4)) if task == '4-way' else np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(X, y):
        m = model(len(tr)).fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te]) if task == '4-way' else m.decision_function(X[te])
    return s


def cell_score(task, c, seed):
    m, y = TASKS[task]
    return task, c, seed, score(task, y[m], cv(task, Hall[m][:, c[0], c[1]], y[m], seed))


def nested(task, seed):
    m, y = TASKS[task]
    X, y = Hall[m], y[m]
    s = np.zeros((len(y), 4)) if task == '4-way' else np.zeros(len(y))
    picks = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(X, y):
        inner = {c: score(task, y[tr], cv(task, X[tr][:, c[0], c[1]], y[tr], seed, k=4)) for c in CELLS}
        c = max(inner, key=inner.get)
        picks.append(f'{POS[c[0]]}/L{LAYERS[c[1]]}')
        mm = model(len(tr)).fit(X[tr][:, c[0], c[1]], y[tr])
        s[te] = mm.predict_proba(X[te][:, c[0], c[1]]) if task == '4-way' else mm.decision_function(X[te][:, c[0], c[1]])
    return task, score(task, y, s), picks


with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=-1)(delayed(cell_score)(t, c, s) for t in TASKS for c in CELLS for s in range(3))
    nest = Parallel(n_jobs=-1)(delayed(nested)(t, s) for t in TASKS for s in range(3))
R = collections.defaultdict(list)
for t, c, s, v in res:
    R[(t, c)].append(v)
for t in TASKS:
    print(f'\n## {t} ({"AUROC" if t != "4-way" else "balanced accuracy"}), rows = position, columns = layer')
    print(f"{'':8s}" + ''.join(f'{"L" + str(l):>7s}' for l in LAYERS))
    for p in range(len(POS)):
        print(f'{POS[p]:8s}' + ''.join(f'{np.mean(R[(t, (p, l))]):7.3f}' for l in range(len(LAYERS))))
    best = max(CELLS, key=lambda c: np.mean(R[(t, c)]))
    vals = [v for tt, v, _ in nest if tt == t]
    print(f'best cell {POS[best[0]]}/L{LAYERS[best[1]]} {np.mean(R[(t, best)]):.3f} (picked on the same folds); '
          f'nested estimate {np.mean(vals):.3f} (seeds {", ".join(f"{v:.3f}" for v in vals)}); '
          f'cells picked: {collections.Counter(p for tt, _, ps in nest if tt == t for p in ps).most_common(4)}')
