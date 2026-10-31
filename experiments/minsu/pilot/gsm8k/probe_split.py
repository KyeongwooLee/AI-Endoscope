"""Key configs of probe_gen.py, AUROC reported separately for GSM8K test vs train problems (train may be seen in pretraining)."""
import json, sys, collections, warnings, os
import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
F = sys.argv[1]
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in (0, 1)]
meta = [m for d in metas for m in d['meta']]
LAYERS, POS, CONF = metas[0]['layers'], metas[0]['positions'], metas[0]['conf']
keep = np.array([m['format_ok'] for m in meta])
C = np.concatenate([np.load(f'{F}/C_{s}.npy') for s in (0, 1)])[keep]
meta = [m for m, k in zip(meta, keep) if k]
N = len(meta)
y = np.array([not m['correct'] for m in meta], dtype=int)
split = np.array([m['split'] for m in meta])
gkey = {k: i for i, k in enumerate(sorted({(m['split'], m['pid']) for m in meta}))}
g = np.array([gkey[(m['split'], m['pid'])] for m in meta])
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(N), y, g))
ci = CONF.index('length'); conf_cols = [i for i in range(len(CONF)) if i != ci]


def run(kind, p=None, l=None, cols=None):
    s = np.zeros(N)
    X = None if kind == 'scalar' else np.load(f'{F}/H_all.npy', mmap_mode='r')[:, POS.index(p), LAYERS.index(l)].astype(np.float32)
    for tr, te in folds:
        if kind == 'scalar':
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(C[tr][:, cols], y[tr])
            s[te] = m.decision_function(C[te][:, cols]); continue
        red = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr])
        Z = red.transform(X)
        if kind == 'hidden+conf':
            Z = np.hstack([Z, StandardScaler().fit(C[tr]).transform(C)])
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        s[te] = m.decision_function(Z[te])
    return s


def within(s, mask):
    gi = collections.defaultdict(list)
    for i in np.where(mask)[0]:
        gi[g[i]].append(i)
    num = den = 0.0
    for idx in gi.values():
        idx = np.array(idx)
        if 0 < y[idx].sum() < len(idx):
            d = s[idx][y[idx] == 1][:, None] - s[idx][y[idx] == 0][None, :]
            num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return num / den


cfg = [('confidence', ('scalar', None, None, conf_cols)), ('confidence + length', ('scalar', None, None, list(range(len(CONF))))),
       ('question only: hidden q_end L20', ('hidden', 'q_end', 20, None)), ('hidden r_mean L20', ('hidden', 'r_mean', 20, None)),
       ('hidden r_mean L20 + conf + length', ('hidden+conf', 'r_mean', 20, None))]
S = Parallel(n_jobs=len(cfg))(delayed(run)(*a) for _, a in cfg)
for sp in ['test', 'train']:
    mk = split == sp
    print(f'## {sp}: solutions={mk.sum()}, wrong rate={y[mk].mean():.3f}')
    print(f"  {'input':36s} {'all':>6s} {'within':>7s}")
    for (n, _), s in zip(cfg, S):
        print(f'  {n:36s} {roc_auc_score(y[mk], s[mk]):6.3f} {within(s, mk):7.3f}')
