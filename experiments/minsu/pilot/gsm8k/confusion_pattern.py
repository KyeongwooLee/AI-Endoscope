"""4-way pattern probe: class shares, confusion matrix, per-class recall/precision, overall accuracy.
Inputs: self hidden r_mean L20, confidence, Qwen2.5-7B reader L20 (extract_reader.py). Two classifiers: class-balanced
(as in probe_pattern.py) and unweighted (maximizes plain accuracy). 5-fold CV x 5 seeds, probabilities averaged.
usage: python confusion_pattern.py <feat_dir> <reader_root> <fix_dir> [<fix_dir> ...]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
def pattern(i):
    p = [key(x) for x in pred[i]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / len(p) if wrong else 0)
    return 'dominant' if c >= .5 else 'stuck' if w >= .5 else 'competing' if c >= .2 and w >= .2 else \
        'spread' if len(set(p)) > 10 else 'other'
allp = collections.Counter(pattern(i) for i in pred if len(pred[i]) == 32)
n = sum(allp.values())
print(f'all {n} wrong answers (32 restarts each): ' + '  '.join(f'{k} {allp[k]} ({allp[k]/n:.1%})' for k in ['dominant', 'stuck', 'competing', 'spread', 'other']))
P4 = ['dominant', 'stuck', 'competing', 'spread']
ids = sorted(i for i in pred if len(pred[i]) == 32 and pattern(i) in P4)
y = np.array([P4.index(pattern(i)) for i in ids])
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
def run(name, bal, seed):
    out = np.zeros((len(y), 4)); A = X[name]; big = A.shape[1] > 256
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(A, y):
        clf = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                            LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced' if bal else None)).fit(A[tr], y[tr])
        out[te] = clf.predict_proba(A[te])
    return name, bal, out
with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=-1)(delayed(run)(nm, b, s) for nm in X for b in (True, False) for s in range(5))
S = collections.defaultdict(list)
for nm, b, o in res: S[(nm, b)].append(o)
print(f'\n4 classes, {len(ids)} items; always-"dominant" accuracy {np.mean(y == 0):.1%}, chance balanced 25%')
for b in (True, False):
    print(f"\n######## {'class-balanced' if b else 'unweighted'} classifier")
    for nm in X:
        p = np.mean(S[(nm, b)], 0).argmax(1)
        cm = np.array([[np.sum((y == a) & (p == c)) for c in range(4)] for a in range(4)])
        rec = cm.diagonal() / cm.sum(1); prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
        print(f'\n## {nm}: accuracy {np.mean(p == y):.1%}, balanced {rec.mean():.1%}')
        print(f"{'true/pred':12s}" + ''.join(f'{c:>11s}' for c in P4) + f"{'recall':>9s}")
        for a in range(4):
            print(f'{P4[a]:12s}' + ''.join(f'{cm[a, c]:11d}' for c in range(4)) + f'{rec[a]:9.0%}')
        print(f"{'precision':12s}" + ''.join(f'{v:11.0%}' for v in prec))
