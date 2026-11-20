"""Out-of-fold probes for the correction pipeline on GSM8K train (rs30 questions).

(1) wrong: is the greedy answer wrong?             (binary, AUROC)
(2) cls3 : response class of the question, from samples 1-15 only (samples 16-30 are kept for the
           correction simulation):  easy = >=TH correct, stuck = one wrong answer >=TH times, wobbly = rest.
Inputs (all from the 1.5B model's greedy answer): hidden r_mean L20, question-only q_end L20, confidence (9 feats).
Every prediction is out-of-fold (5-fold over questions), so the simulation never sees a probe trained on that question.
usage: python pipeline_probe.py <rs_dir> <feat_dir> <out.npz> [TH=8]
"""
import collections, glob, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RS, F, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TH = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CLS = ['easy', 'wobbly', 'stuck']
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6


def cls3(preds, gold):
    c = sum(ok(x, gold) for x in preds)
    w = collections.Counter(key(x) for x in preds if x is not None and not ok(x, gold))
    top = w.most_common(1)[0][1] if w else 0
    return 0 if c >= TH else (2 if top >= TH else 1)


rows = {r['greedy_id']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
meta = json.load(open(f'{F}/meta_0.json'))
L20, POS = meta['layers'].index(20), meta['positions']
H = np.load(f'{F}/H_0.npy', mmap_mode='r')
C = np.load(f'{F}/C_0.npy')
mid = [f"{m['split']}-{m['pid']}-{m['sample']}" for m in meta['meta']]
J = [j for j, i in enumerate(mid) if i in rows and meta['meta'][j]['format_ok']]
ids = [mid[j] for j in J]
print(f'questions {len(rows)}; greedy answers with features and format ok: {len(ids)}')
J = np.array(J)
X = {'hidden r_mean': np.asarray(H[J, POS.index('r_mean'), L20], dtype=np.float32),
     'question q_end': np.asarray(H[J, POS.index('q_end'), L20], dtype=np.float32),
     'confidence': C[J].astype(np.float32)}
wrong = np.array([not rows[i]['greedy_correct'] for i in ids], dtype=int)
y3 = np.array([cls3(rows[i]['preds'][:15], rows[i]['gold']) for i in ids])
print(f'greedy wrong {wrong.mean():.1%}; classes (TH={TH}): ' +
      '  '.join(f'{c} {np.mean(y3 == k):.1%}' for k, c in enumerate(CLS)) +
      ' | among greedy-wrong: ' + '  '.join(f'{c} {np.mean(y3[wrong == 1] == k):.1%}' for k, c in enumerate(CLS)))


def model(d, n):
    big = d > 256
    return make_pipeline(StandardScaler(), *([PCA(min(256, n - 1), random_state=0)] if big else []),
                         LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))


def fold(name, task, tr, te):
    y = wrong if task == 'wrong' else y3
    m = model(X[name].shape[1], len(tr)).fit(X[name][tr], y[tr])
    return name, task, te, m.predict_proba(X[name][te])


strat = wrong * 3 + y3
splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, strat))
with parallel_backend('loky', inner_max_num_threads=4):
    res = Parallel(n_jobs=30)(delayed(fold)(n, t, tr, te) for n in X for t in ('wrong', 'cls3') for tr, te in splits)
P = {}
for name, task, te, p in res:
    P.setdefault((name, task), np.zeros((len(ids), p.shape[1])))[te] = p

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(1000)]
print('\n## (1) greedy answer wrong: AUROC [95% CI]')
for n in X:
    s = P[(n, 'wrong')][:, 1]
    b = [roc_auc_score(wrong[i], s[i]) for i in boot[:300]]
    print(f'  {n:16s} {roc_auc_score(wrong, s):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]')
print('\n## (2) 3-class, all questions and greedy-wrong only: balanced accuracy, per-class recall, stuck-vs-rest AUROC')
for n in X:
    p = P[(n, 'cls3')]
    pred = p.argmax(1)
    for grp, msk in (('all', np.ones(len(ids), bool)), ('greedy-wrong', wrong == 1)):
        rec = '  '.join(f'{c} {np.mean(pred[msk & (y3 == k)] == k):.0%}' for k, c in enumerate(CLS))
        print(f'  {n:16s} {grp:13s} bal.acc {balanced_accuracy_score(y3[msk], pred[msk]):.3f}  recall {rec}  '
              f'stuck AUROC {roc_auc_score(y3[msk] == 2, p[msk, 2]):.3f}')
np.savez(OUT, ids=np.array(ids), wrong=wrong, y3=y3, TH=TH,
         **{f"{n.split()[0]}_{t}": v for (n, t), v in P.items()})
print('saved', OUT)
