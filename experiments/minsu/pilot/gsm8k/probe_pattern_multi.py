"""Multi-label answer-distribution patterns: each pattern is its own yes/no label (criteria may overlap).
dominant: correct >= 50%; stuck: one wrong answer >= 50%; competing: correct >= 20% and one wrong answer >= 20%;
spread: > 10 distinct answers (of 32). Items with no label stay in (all four 'no').
Per label: AUROC, and recall / precision of the class-balanced classifier; exact match of all four labels.
Inputs: self hidden r_mean L20, confidence, Qwen2.5-7B reader L20. 5-fold CV x 5 seeds (folds stratified by label set).
usage: python probe_pattern_multi.py <feat_dir> <reader_root> <fix_dir> [<fix_dir> ...]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
F, RROOT, FIXES = sys.argv[1], sys.argv[2], sys.argv[3:]
pred, gold = collections.defaultdict(list), {}
for D in FIXES:
    for f in sorted(glob.glob(f'{D}/fix_*.jsonl')) + sorted(glob.glob(f'{D}/more/fix_*.jsonl')):
        for r in map(json.loads, open(f)):
            if r['cond'] == 'restart':
                pred[r['id']] += r['preds']; gold[r['id']] = r['gold']
key = lambda x: None if x is None else round(x, 4)
NAMES = ['dominant', 'stuck', 'competing', 'spread']
def labels(i):
    p = [key(x) for x in pred[i]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / len(p) if wrong else 0)
    return [c >= .5, w >= .5, c >= .2 and w >= .2, len(set(p)) > 10]
ids = sorted(i for i in pred if len(pred[i]) == 32)
Y = np.array([labels(i) for i in ids], dtype=int)
print(f'items {len(ids)}; ' + '  '.join(f'{n} {Y[:, k].sum()} ({Y[:, k].mean():.1%})' for k, n in enumerate(NAMES)))
combo = np.array([''.join(map(str, y)) for y in Y])
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
keep = np.array([m['format_ok'] for d in metas for m in d['meta']])
meta = [m for d in metas for m in d['meta'] if m['format_ok']]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}
J = np.array([row[i] for i in ids]); LAYERS, POS = metas[0]['layers'], metas[0]['positions']
m7 = json.load(open(f'{RROOT}/qwen7b/meta.json')); p7 = {i: k for k, i in enumerate(m7['ids'])}
X = {'self hidden': np.asarray(np.load(f'{F}/H_all.npy', mmap_mode='r')[J, POS.index('r_mean'), LAYERS.index(20)], dtype=np.float32),
     'confidence': np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep][J].astype(np.float32),
     'Qwen7B reader': np.asarray(np.load(f'{RROOT}/qwen7b/H.npy', mmap_mode='r')[[p7[i] for i in ids], 1, m7['layers'].index(20)], dtype=np.float32)}
strat = np.unique(combo, return_inverse=True)[1]
rare = np.bincount(strat) < 5
strat = np.where(rare[strat], -1, strat)  # rare label sets pooled for stratification
def run(name, k, seed):
    s = np.zeros(len(ids)); A = X[name]; big = A.shape[1] > 256
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, strat):
        clf = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                            LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced')).fit(A[tr], Y[tr, k])
        s[te] = clf.decision_function(A[te])
    return name, k, s
with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=-1)(delayed(run)(n, k, s) for n in X for k in range(4) for s in range(5))
S = collections.defaultdict(list)
for n, k, s in res: S[(n, k)].append(s)
for n in X:
    print(f'\n## {n}')
    print(f"{'label':10s} {'share':>7s} {'AUROC':>7s} {'recall':>7s} {'precision':>10s}")
    P = np.zeros_like(Y)
    for k in range(4):
        s = np.mean(S[(n, k)], 0); p = (s > 0).astype(int); P[:, k] = p
        rec = (p & Y[:, k]).sum() / Y[:, k].sum(); prec = (p & Y[:, k]).sum() / max(p.sum(), 1)
        print(f'{NAMES[k]:10s} {Y[:, k].mean():7.1%} {roc_auc_score(Y[:, k], s):7.3f} {rec:7.0%} {prec:10.0%}')
    print(f'all four labels right: {np.mean((P == Y).all(1)):.1%}')
