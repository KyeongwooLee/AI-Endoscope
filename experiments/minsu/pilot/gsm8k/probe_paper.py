"""Replication of Orgad et al. (ICLR 2025) Sec. 5 on GSM8K train with Qwen2.5-1.5B-Instruct.

Taxonomy from K = 30 resamples at T = 1 (resample_all.py), every question, overlapping types (paper's wording):
  (A) refuses        no answer (no number) in at least half
  (B) consistently correct    correct in at least half;  B1 always correct, B2 mostly correct
  (C) consistently incorrect  the same incorrect answer in at least half;  C1 correct never produced, C2 at least once
  (D) two competing  correct and (most common) incorrect answer at similar rates: counts differ by <= 5, each >= 5
  (E) many answers   more than 10 distinct answers;  E1 correct never generated, E2 at least once
Probe: one-vs-rest logistic regression per type on the representation of the GREEDY answer (paper: last exact answer
token, best token/layer). Here: scaler + PCA(256) + LR (C=0.05, balanced) as in our other probes.
Evaluation: 5 seeds of a stratified 80/20 question split; on the 80% the layer is chosen by 3-fold CV, then refit and
scored (AUROC) on the 20%; mean +- std over seeds (the paper reports mean +- std too). Also scored on the test
questions whose greedy answer is wrong (closer to our earlier wrong-answers-only setting).
Inputs: self ans / r_mean / q_end (question only = difficulty), confidence, and Qwen2.5-7B reading the greedy answer
(extract_reader.py ... greedy) when <reader_dir> exists.
usage: python probe_paper.py <rs_dir> <feat_dir> [<reader_dir>]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RS, F = sys.argv[1], sys.argv[2]
RD = sys.argv[3] if len(sys.argv) > 3 and os.path.exists(f'{sys.argv[3]}/meta.json') else None
rows = [r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))]
key = lambda x: None if x is None else round(x, 4)


def types(r):
    p = [key(x) for x in r['preds']]
    ok = [x is not None and abs(x - r['gold']) < 1e-6 for x in p]
    K, c, none = len(p), sum(ok), sum(x is None for x in p)
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o and x is not None)
    w = wrong.most_common(1)[0][1] if wrong else 0
    distinct = len({x for x in p if x is not None})
    t = {'A': none >= K / 2, 'B': c >= K / 2, 'C': w >= K / 2, 'D': abs(c - w) <= 5 and c >= 5 and w >= 5,
         'E': distinct > 10}
    t.update({'B1': c == K, 'B2': t['B'] and c < K, 'C1': t['C'] and c == 0, 'C2': t['C'] and c > 0,
              'E1': t['E'] and c == 0, 'E2': t['E'] and c > 0})
    return t


T = ['A', 'B', 'C', 'D', 'E', 'B1', 'B2', 'C1', 'C2', 'E1', 'E2']
NAME = {'A': 'refuses', 'B': 'consistently correct', 'C': 'consistently incorrect', 'D': 'two competing',
        'E': 'many answers', 'B1': 'always correct', 'B2': 'mostly correct', 'C1': 'C, correct never',
        'C2': 'C, correct sometimes', 'E1': 'E, correct never', 'E2': 'E, correct sometimes'}
lab = {r['greedy_id']: types(r) for r in rows}
gwrong = {r['greedy_id']: not r['greedy_correct'] for r in rows}
n = len(rows)
print(f'questions {n}, 30 samples each at T=1; greedy accuracy {1 - np.mean(list(gwrong.values())):.1%}')
print('type shares (overlapping): ' + '  '.join(f"{t} {sum(l[t] for l in lab.values())} ({np.mean([l[t] for l in lab.values()]):.1%})" for t in T))
combo = collections.Counter(' + '.join(t for t in 'ABCDE' if l[t]) or 'none' for l in lab.values())
print('top-level combinations: ' + '  '.join(f'{k} {v}' for k, v in combo.most_common()))
print('among greedy-wrong questions: ' + '  '.join(f"{t} {np.mean([lab[i][t] for i in lab if gwrong[i]]):.1%}" for t in 'ABCDE'))

shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
keep = np.array([m['format_ok'] for d in metas for m in d['meta']])
meta = [m for d in metas for m in d['meta'] if m['format_ok']]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}
ids = sorted(i for i in lab if i in row)
print(f'greedy answers with features (format ok): {len(ids)} of {n}')
J = np.array([row[i] for i in ids])
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
H = np.load(f'{F}/H_all.npy', mmap_mode='r')
FAM = {f'self {p}': [np.asarray(H[J, POS.index(p), li], dtype=np.float32) for li in range(len(LAYERS))]
       for p in ('ans', 'r_mean', 'q_end')}
FAM['confidence'] = [np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep][J].astype(np.float32)]
if RD:
    rm = json.load(open(f'{RD}/meta.json'))
    rp = {i: k for k, i in enumerate(rm['ids'])}
    RH = np.load(f'{RD}/H.npy', mmap_mode='r')
    RJ = [rp[i] for i in ids]
    for p in ('ans', 'r_mean'):
        FAM[f'7B reader {p}'] = [np.asarray(RH[RJ, rm['positions'].index(p), li], dtype=np.float32) for li in range(len(rm['layers']))]
Y = {t: np.array([lab[i][t] for i in ids], dtype=int) for t in T}
GW = np.array([gwrong[i] for i in ids])
TT = [t for t in T if Y[t].sum() >= 30]


def model(big, n):
    return make_pipeline(StandardScaler(), *([PCA(min(256, n - 1), random_state=0)] if big else []),
                         LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))


def job(fam, t, seed):
    y, Xs = Y[t], FAM[fam]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    big = Xs[0].shape[1] > 256
    if len(Xs) > 1:
        inner = []
        for li, X in enumerate(Xs):
            s = np.zeros(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=seed).split(tr, y[tr]):
                s[b] = model(big, len(a)).fit(X[tr[a]], y[tr[a]]).decision_function(X[tr[b]])
            inner.append(roc_auc_score(y[tr], s))
        li = int(np.argmax(inner))
    else:
        li = 0
    s = model(big, len(tr)).fit(Xs[li][tr], y[tr]).decision_function(Xs[li][te])
    w = GW[te]
    aw = roc_auc_score(y[te][w], s[w]) if 0 < y[te][w].sum() < w.sum() else np.nan
    return fam, t, seed, li, roc_auc_score(y[te], s), aw


with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=-1)(delayed(job)(f, t, s) for f in FAM for t in TT for s in range(5))
R = collections.defaultdict(list)
for f, t, s, li, a, aw in res:
    R[(f, t)].append((li, a, aw))
for title, k in (('all questions (paper setting)', 1), ('test questions whose greedy answer is wrong', 2)):
    print(f'\n## AUROC, {title}; mean +- std over 5 splits; [layer most often chosen]')
    print(f"{'type':28s}" + ''.join(f'{f:>22s}' for f in FAM))
    for t in TT:
        cells = []
        for f in FAM:
            v = np.array([x[k] for x in R[(f, t)]], dtype=float)
            lay = collections.Counter(x[0] for x in R[(f, t)]).most_common(1)[0][0]
            ls = '' if f == 'confidence' else (f'L{LAYERS[lay]}' if f.startswith('self') else f'L{rm["layers"][lay]}')
            cells.append(f'{np.nanmean(v):.3f}+-{np.nanstd(v):.3f} {ls:>4s}')
        print(f'({t}) {NAME[t]:23s}' + ''.join(f'{c:>22s}' for c in cells))
