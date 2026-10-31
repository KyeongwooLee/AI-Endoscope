"""Solution-level correct/wrong classification on the model's own GSM8K solutions.

Inputs compared: length, confidence scalars, question-only hidden (difficulty baseline), hidden states, hidden+confidence.
Metrics (all AUROC, 'wrong' = positive):
  all     : every solution, across problems (difficulty included)
  greedy  : greedy solutions only (comparable to the teammate's setting)
  within  : only problems with both correct and wrong samples; P(risk(wrong) > risk(correct)) for pairs of the
            SAME problem -> difficulty removed
Problem-grouped 5-fold CV; format failures excluded.
usage: python probe_gen.py <feat_dir>
"""
import json, sys, collections, warnings, os
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
F = sys.argv[1]
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for d in metas for m in d['meta']]
LAYERS, POS, CONF = metas[0]['layers'], metas[0]['positions'], metas[0]['conf']
keep = np.array([m['format_ok'] for m in meta])
if not os.path.exists(f'{F}/H_all.npy'):
    np.save(f'{F}/H_all.npy', np.concatenate([np.load(f'{F}/H_{s}.npy') for s in shards])[keep])
C = np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep]
meta = [m for m, k in zip(meta, keep) if k]
N = len(meta)
y = np.array([not m['correct'] for m in meta], dtype=int)
greedy = np.array([m['mode'] == 'greedy' for m in meta])
gkey = {k: i for i, k in enumerate(sorted({(m['split'], m['pid']) for m in meta}))}
g = np.array([gkey[(m['split'], m['pid'])] for m in meta])
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(N), y, g))
gi = collections.defaultdict(list)
for i, k in enumerate(g):
    gi[k].append(i)
mixed = {k: np.array(v) for k, v in gi.items() if 0 < y[v].sum() < len(v)}
print(f'solutions={N} (format failures excluded: {int((~keep).sum())}), problems={len(gkey)}, '
      f'acc greedy={1 - y[greedy].mean():.3f} sampled={1 - y[~greedy].mean():.3f}, '
      f'mixed problems={len(mixed)} ({sum(int(y[v].sum()) * int((1 - y[v]).sum()) for v in mixed.values())} wrong-correct pairs)')


mkeys = sorted(mixed)


def within_counts(s):
    """per mixed problem: (#wrong-correct pairs ranked right, #pairs)"""
    num, den = np.zeros(len(mkeys)), np.zeros(len(mkeys))
    for j, k in enumerate(mkeys):
        idx = mixed[k]
        d = s[idx][y[idx] == 1][:, None] - s[idx][y[idx] == 0][None, :]
        num[j], den[j] = (d > 0).sum() + 0.5 * (d == 0).sum(), d.size
    return num, den


def within_auc(s):
    num, den = within_counts(s)
    return num.sum() / den.sum()


def scores(builder):
    s = np.zeros(N)
    for tr, te in folds:
        s[te] = builder(tr, te)
    return s


ci = CONF.index('length')
conf_cols = [i for i in range(len(CONF)) if i != ci]


def scalar_model(cols):
    def b(tr, te):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(C[tr][:, cols], y[tr])
        return m.decision_function(C[te][:, cols])
    return b


def hidden_model(p, l, with_conf=False):
    def b(tr, te):
        X = np.load(f'{F}/H_all.npy', mmap_mode='r')[:, POS.index(p), LAYERS.index(l)].astype(np.float32)
        red = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr])
        Z = red.transform(X)  # no down-scaling: keeps hidden on a scale the L2 penalty doesn't suppress
        if with_conf:
            sc = StandardScaler().fit(C[tr])
            Z = np.hstack([Z, sc.transform(C)])
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        return m.decision_function(Z[te])
    return b


configs = [('length', scalar_model([ci])), ('confidence', scalar_model(conf_cols)),
           ('confidence + length', scalar_model(list(range(len(CONF)))))]
for p in POS:
    for l in LAYERS:
        configs.append((f'hidden {p} L{l}', hidden_model(p, l)))
        if p != 'q_end':
            configs.append((f'hidden {p} L{l} + conf + length', hidden_model(p, l, True)))
with parallel_backend('loky', inner_max_num_threads=4):
    S = Parallel(n_jobs=min(len(configs), 30))(delayed(scores)(b) for _, b in configs)
R = {n: s for (n, _), s in zip(configs, S)}
res = {n: (roc_auc_score(y, s), roc_auc_score(y[greedy], s[greedy]), within_auc(s)) for n, s in R.items()}

print('\n## solution-level correct/wrong, AUROC (all / greedy only / within same problem)')
print(f"  {'input':40s} {'all':>6s} {'greedy':>7s} {'within':>7s}")
for n in ['length', 'confidence', 'confidence + length']:
    print(f'  {n:40s} {res[n][0]:6.3f} {res[n][1]:7.3f} {res[n][2]:7.3f}')
for p in POS:
    print(f'  -- {p}')
    for l in LAYERS:
        for n in [f'hidden {p} L{l}', f'hidden {p} L{l} + conf + length']:
            if n in res:
                print(f'  {n:40s} {res[n][0]:6.3f} {res[n][1]:7.3f} {res[n][2]:7.3f}')

# bootstrap over problems for the key comparisons
best_h = max([n for n in res if n.startswith('hidden') and '+' not in n and 'q_end' not in n], key=lambda n: res[n][0])
best_hc = max([n for n in res if '+ conf' in n], key=lambda n: res[n][0])
rng = np.random.default_rng(0)
keys = np.array(sorted(gi))
print(f'\n## paired bootstrap over problems (1000x), 95% CI   [best hidden = {best_h}; best combined = {best_hc}]')
for A, B in [(best_h, 'confidence'), (best_hc, 'confidence + length')]:
    (na, den), (nb, _) = within_counts(R[A]), within_counts(R[B])
    da, dw = [], []
    for _ in range(1000):
        pick = rng.choice(keys, len(keys))
        idx = np.concatenate([gi[k] for k in pick])
        da.append(roc_auc_score(y[idx], R[A][idx]) - roc_auc_score(y[idx], R[B][idx]))
        mp = rng.integers(0, len(mkeys), len(mkeys))
        dw.append(na[mp].sum() / den[mp].sum() - nb[mp].sum() / den[mp].sum())
    print(f'  {A} vs {B}: all {np.mean(da):+.3f} [{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}]'
          f' | within {np.mean(dw):+.3f} [{np.percentile(dw, 2.5):+.3f}, {np.percentile(dw, 97.5):+.3f}]')
