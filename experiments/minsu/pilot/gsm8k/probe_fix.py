"""Can the wrong answer's hidden state tell which correction will fix it? (multi-label: one probe per correction)

Items: the 1,000 wrong answers of fix_exp.py (one per problem, so plain folds are problem folds). Targets from the N=8
attempts per correction (works = at least half fixed):
  restart        restart works
  code           a Python program works
  cut            continuing from just before the audited first wrong line works
  code_only      code works and restart does not (when to pick code instead of restarting)
  restart_rate   share of the 8 restarts that are right (Spearman)
Inputs: hidden r_mean L20 (pre-fixed, as in all earlier probes), question-only hidden q_end L20 (difficulty), confidence +
length, response TF-IDF, and as a reference that is not available in practice: the share of the problem's other 4
original solutions that were right (difficulty measured directly).
5-fold CV x 5 seeds, scores averaged; paired bootstrap over items (500) for hidden minus each other input; within-difficulty AUROC inside groups of the
other-4 accuracy.
usage: python probe_fix.py <feat_dir> <fix_dir> <gen_dir>
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, FIX, GEN = sys.argv[1:4]
REPS = 5
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for m in (m for d in metas for m in d['meta']) if m['format_ok']]  # same row order as H_all.npy
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
C_all = np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[np.array([m['format_ok'] for d in metas for m in d['meta']])]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}

by = collections.defaultdict(dict)
for f in glob.glob(f'{FIX}/fix_*.jsonl') + glob.glob(f'{FIX}/cut_*.jsonl'):
    for r in map(json.loads, open(f)):
        by[r['cond']][r['id']] = np.mean(r['correct'])
ids = sorted(by['restart'])
gen = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f))}
sib = collections.defaultdict(list)
for k, r in gen.items():
    sib[r['pid']].append((k, r['correct']))
J = np.array([row[i] for i in ids])
H = np.load(f'{F}/H_all.npy', mmap_mode='r')
X = {'hidden r_mean': np.asarray(H[J, POS.index('r_mean'), LAYERS.index(20)], dtype=np.float32),
     'question q_end': np.asarray(H[J, POS.index('q_end'), LAYERS.index(20)], dtype=np.float32),
     'confidence+len': C_all[J]}
TEXT = np.array([gen[i]['text'] for i in ids], dtype=object)
other_acc = np.array([np.mean([c for k, c in sib[gen[i]['pid']] if k != i]) for i in ids])

rs = np.array([by['restart'][i] for i in ids])
cd = np.array([by['code'][i] for i in ids])
ct = np.array([by['cut'].get(i, np.nan) for i in ids])
Y = {'restart': rs >= .5, 'code': cd >= .5, 'cut': ct >= .5, 'code_only': (cd >= .5) & (rs < .5)}
print(f'items {len(ids)}; positives: ' + '  '.join(f'{k} {v.sum()} ({v.mean():.1%})' for k, v in Y.items()))


def scores(name, target, seed):
    y = Y[target].astype(int) if target != 'restart_rate' else rs
    s = np.zeros(len(ids))
    split = (StratifiedKFold(5, shuffle=True, random_state=seed).split(s, y) if target != 'restart_rate'
             else KFold(5, shuffle=True, random_state=seed).split(s))
    for tr, te in split:
        if name == 'text':
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
            A, B = vec.fit_transform(TEXT[tr]), vec.transform(TEXT[te])
            m = Ridge(alpha=10.0) if target == 'restart_rate' else LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')
            m.fit(A, y[tr])
        else:
            A, B = X[name][tr], X[name][te]
            pre = [StandardScaler()] + ([PCA(256, random_state=0)] if A.shape[1] > 256 else [])
            last = Ridge(alpha=100.0) if target == 'restart_rate' else \
                LogisticRegression(C=0.05 if A.shape[1] > 256 else 1.0, max_iter=3000, class_weight='balanced')
            m = make_pipeline(*pre, last).fit(A, y[tr])
        s[te] = m.predict(B) if target == 'restart_rate' else m.decision_function(B)
    return name, target, seed, s


NAMES = list(X) + ['text']
TARGETS = list(Y) + ['restart_rate']
with parallel_backend('loky', inner_max_num_threads=1):
    out = Parallel(n_jobs=-1)(delayed(scores)(n, t, s) for n in NAMES for t in TARGETS for s in range(REPS))
S = collections.defaultdict(list)
for n, t, _, s in out:
    S[(n, t)].append(s)
S = {k: np.mean(v, 0) for k, v in S.items()}
for t in TARGETS:
    S[('other 4 solutions (reference)', t)] = other_acc
NAMES.append('other 4 solutions (reference)')


def metric(t, s, m=None):
    m = np.ones(len(ids), bool) if m is None else m
    if t == 'restart_rate':
        return spearmanr(s[m], rs[m]).correlation
    ok = m & ~np.isnan(ct) if t == 'cut' else m
    return roc_auc_score(Y[t][ok], s[ok])


rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(500)]
print('\n## AUROC (restart_rate: Spearman); hidden minus input [95% CI] over 500 item bootstraps')
print(f"{'input':32s} " + ' '.join(f'{t:>24s}' for t in TARGETS))
for n in NAMES:
    cells = []
    for t in TARGETS:
        v = metric(t, S[(n, t)])
        if n == 'hidden r_mean':
            cells.append(f'{v:24.3f}')
            continue
        ds = []
        for idx in boot:
            try:
                if t == 'restart_rate':
                    a = spearmanr(S[('hidden r_mean', t)][idx], rs[idx]).correlation - spearmanr(S[(n, t)][idx], rs[idx]).correlation
                else:
                    yy = Y[t][idx]
                    ok = ~np.isnan(ct[idx]) if t == 'cut' else np.ones(len(idx), bool)
                    a = roc_auc_score(yy[ok], S[('hidden r_mean', t)][idx][ok]) - roc_auc_score(yy[ok], S[(n, t)][idx][ok])
            except ValueError:
                continue
            ds.append(a)
        cells.append(f'{v:6.3f} ({np.percentile(ds, 2.5):+.2f},{np.percentile(ds, 97.5):+.2f})')
    print(f'{n:32s} ' + ' '.join(f'{c:>24s}' for c in cells))
print('(parentheses: 95% CI of hidden r_mean minus this input; above 0 = hidden better)')

print('\n## within the same difficulty: AUROC inside groups of the problem\'s other-4 accuracy (0/4 ... 4/4), averaged by size')
for t in ('restart', 'code', 'code_only'):
    line = []
    for n in ('hidden r_mean', 'question q_end', 'confidence+len', 'text'):
        vals, ws = [], []
        for g in np.unique(other_acc):
            m = other_acc == g
            if Y[t][m].sum() >= 5 and (~Y[t][m]).sum() >= 5:
                vals.append(roc_auc_score(Y[t][m], S[(n, t)][m])); ws.append(m.sum())
        line.append(f'{n} {np.average(vals, weights=ws):.3f}' if vals else f'{n} n/a')
    print(f'  {t:10s} ' + '  '.join(line))
