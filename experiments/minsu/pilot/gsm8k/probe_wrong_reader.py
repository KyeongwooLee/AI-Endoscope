"""External-reader control for correct/wrong detection of the 1.5B greedy answers (GSM8K train).

Self = the 1.5B writer's own hidden states while producing its solution (feat_g). Reader = another model reading the
same prompt + solution (extract_reader.py). Same labels, same 5 folds over questions, same probe.
Pre-registered comparison: the layer at 5/7 depth for every model (1.5B L20). All layers are printed as well.
usage: python probe_wrong_reader.py <rs_dir> <feat_dir> <reader_dir> [<reader_dir> ...]
"""
import glob, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RS, F, RDS = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {r['greedy_id']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
meta = json.load(open(f'{F}/meta_0.json'))
mid = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta['meta']) if m['format_ok']}
readers = [(d, json.load(open(f'{d}/meta.json'))) for d in RDS]
ids = sorted(i for i in rows if i in mid and all(i in set(rm['ids']) for _, rm in readers))
y = np.array([not rows[i]['greedy_correct'] for i in ids], dtype=int)
print(f'greedy answers {len(ids)}, wrong {y.mean():.1%}')
H = np.load(f'{F}/H_0.npy', mmap_mode='r')
J = np.array([mid[i] for i in ids])
POS, LAY = meta['positions'], meta['layers']
FAM = {}  # name -> list of per-layer matrices
for p in ('r_mean', 'ans'):
    FAM[f'self 1.5B {p}'] = [np.asarray(H[J, POS.index(p), li], dtype=np.float32) for li in range(len(LAY))]
FAM['self 1.5B confidence'] = [np.load(f'{F}/C_0.npy')[J].astype(np.float32)]
for d, rm in readers:
    RH = np.load(f'{d}/H.npy', mmap_mode='r')
    rp = {i: k for k, i in enumerate(rm['ids'])}
    RJ = np.array([rp[i] for i in ids])
    name = rm['model'].split('/')[-1]
    for p in ('r_mean', 'ans', 'q_end'):
        FAM[f'{name} {p}'] = [np.asarray(RH[RJ, rm['positions'].index(p), li], dtype=np.float32)
                              for li in range(len(rm['layers']))]
splits = list(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y))


def oof(X):
    big = X.shape[1] > 256
    s = np.zeros(len(y))
    for tr, te in splits:
        m = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                          LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))
        s[te] = m.fit(X[tr], y[tr]).decision_function(X[te])
    return s


jobs = [(f, li) for f in FAM for li in range(len(FAM[f]))]
with parallel_backend('loky', inner_max_num_threads=4):
    out = Parallel(n_jobs=32)(delayed(oof)(FAM[f][li]) for f, li in jobs)
S = {(f, li): s for (f, li), s in zip(jobs, out)}
print('\n## AUROC (out-of-fold) at every layer; * = pre-registered 5/7-depth layer')
for f in FAM:
    n = len(FAM[f])
    pre = 4 if n == 7 else 0
    print(f'  {f:32s} ' + '  '.join(f"{roc_auc_score(y, S[(f, li)]):.3f}{'*' if li == pre and n > 1 else ''}"
                                    for li in range(n)))
rng = np.random.default_rng(0)
B = [rng.integers(0, len(y), len(y)) for _ in range(500)]
pick = lambda f: S[(f, 4 if len(FAM[f]) == 7 else 0)]
ref = pick('self 1.5B r_mean')
print('\n## pre-registered layer: model minus self 1.5B r_mean, AUROC difference [95% bootstrap CI]')
for f in FAM:
    if f == 'self 1.5B r_mean':
        continue
    s = pick(f)
    d = [roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], ref[b]) for b in B]
    print(f'  {f:32s} {roc_auc_score(y, s):.3f}  diff {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
# does self add to the best reader? stack self r_mean L20 + reader r_mean at 5/7 depth
print('\n## stacking: reader r_mean + self r_mean (both pre-registered layer) vs reader alone')
for d, rm in readers:
    name = rm['model'].split('/')[-1]
    X = np.concatenate([FAM[f'{name} r_mean'][4], FAM['self 1.5B r_mean'][4]], 1)
    s = oof(X)
    r = pick(f'{name} r_mean')
    dd = [roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], r[b]) for b in B]
    print(f'  {name:24s} reader {roc_auc_score(y, r):.3f}  reader+self {roc_auc_score(y, s):.3f}  '
          f'gain {np.mean(dd):+.3f} [{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]')
