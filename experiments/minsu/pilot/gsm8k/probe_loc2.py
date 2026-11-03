"""Location probe v2: better hidden features / training for finding the first wrong equation (see probe_loc.py).

Changes tested against v1 (independent per-step LR on hidden res L20 + position):
  1. token positions: pre (just before the step), res, mean over the step, and all three together
  2. listwise training: softmax over the steps of a solution ('which step is the first wrong one?'), and
     solution-centered features (each step minus the mean of its solution)
  3. several layers together (L12/16/20/24)
  5. stronger confidence: step-start entropy, step entropy / margin, also solution-centered
Baselines get the same help (listwise + centered) so the comparison stays fair.
Model selection: problem-grouped 5-fold CV on GSM8K train-split problems only (dev). Final check, once: train on all
dev, evaluate on test-split problems. The selected model is the hidden config with the best dev accuracy; it is
compared on test with the best non-hidden config and v1, paired bootstrap over problems.
usage: python probe_loc2.py <loc_dir>
"""
import json, os, sys, warnings
import numpy as np, scipy.sparse as sp, torch
import torch.nn.functional as F
from joblib import Parallel, delayed, parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

warnings.filterwarnings('ignore')
D = sys.argv[1]
shards = sorted(int(f[9:-5]) for f in os.listdir(D) if f.startswith('meta_loc_'))
metas = [json.load(open(f'{D}/meta_loc_{s}.json')) for s in shards]
M = [m for d in metas for m in d['meta']]
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
HF = f'{D}/H_loc.npy'
if not os.path.exists(HF):
    np.save(HF, np.concatenate([np.load(f'{D}/H_loc_{s}.npy') for s in shards]))

N = len(M)
sol = np.array([f"{m['split']}-{m['pid']}-{m['sample']}" for m in M])
prob = np.array([f"{m['split']}-{m['pid']}" for m in M])
split = np.array([m['split'] for m in M])
e, n, k = (np.array([m[f] for m in M]) for f in ('e', 'n_eq', 'k'))
cat = np.array([m['cat'] for m in M])
y = (e == k).astype(int)
C = np.array([m['conf'] for m in M], dtype=np.float32)
TEXT = np.array([m['line'] for m in M], dtype=object)
oh = lambda v: np.eye(6)[np.minimum(v, 5)]
POSF = np.hstack([np.stack([e, n, e / np.maximum(n - 1, 1)], 1), oh(e), oh(n - 1 - e)]).astype(np.float32)
sid = np.unique(sol, return_inverse=True)[1]
G = sp.csr_matrix((np.ones(N), (sid, np.arange(N))))
solmean = lambda Z: (G @ Z / np.asarray(G.sum(1)))[sid]
DEV, TEST = np.where(split == 'train')[0], np.where(split == 'test')[0]
SPLITS = [(DEV[a], DEV[b]) for a, b in GroupKFold(5).split(DEV, groups=prob[DEV])] + [(DEV, TEST)]  # last = final
LAMS = [3e-4, 3e-3, 3e-2]
print(f'skipped solutions (not aligned): {sum(d["skipped"] for d in metas)}')
print(f'steps={N}  solutions={len(set(sol))}  dev solutions={len(set(sol[DEV]))}  test solutions={len(set(sol[TEST]))}')


def hidden(blocks):
    def f(tr):
        dim = 256 if len(blocks) == 1 else 128
        H = np.load(HF, mmap_mode='r')
        parts = []
        for p, l in blocks:
            X = H[:, POS.index(p), LAYERS.index(l)].astype(np.float32)
            parts.append(make_pipeline(StandardScaler(), PCA(dim, random_state=0)).fit(X[tr]).transform(X))
        return np.hstack(parts)
    return f


def text_svd(tr):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                          token_pattern=r'(?u)\b\w+\b|[+\-*/=]').fit(TEXT[tr])
    X = vec.transform(TEXT)
    return TruncatedSVD(256, random_state=0).fit(X[tr]).transform(X)


def featurize(parts, centered, tr):
    """parts: list of 'pos' / 'conf' / callables(tr) -> dense features of all N steps"""
    Z = [POSF if p == 'pos' else C if p == 'conf' else p(tr) for p in parts]
    Z = np.hstack(Z)
    if centered:
        Z = np.hstack([Z, Z - solmean(Z)])  # each step relative to its own solution
    return StandardScaler().fit(Z[tr]).transform(Z)


def listwise(Z, tr, te, lam):
    """linear scorer, softmax over the steps of each training solution, target = the first wrong step"""
    torch.set_num_threads(3)
    s_tr = sid[tr]
    u, inv = np.unique(s_tr, return_inverse=True)
    cnt = np.bincount(inv)
    order = np.argsort(inv, kind='stable')
    slot = np.arange(len(inv)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    idx = np.full((len(u), cnt.max()), -1)
    idx[inv[order], slot] = order
    tgt = np.zeros(len(u), int)
    tgt[inv[order][y[tr][order] == 1]] = slot[y[tr][order] == 1]
    X = torch.tensor(Z[tr], dtype=torch.float32)
    I, mask, T = torch.tensor(np.maximum(idx, 0)), torch.tensor(idx >= 0), torch.tensor(tgt)
    w = torch.zeros(Z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        S = (X @ w)[I].masked_fill(~mask, -1e9)
        loss = F.cross_entropy(S, T) + lam * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    return Z[te] @ w.detach().numpy()


def indep(Z, tr, te, c):
    return LogisticRegression(C=c, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr]).decision_function(Z[te])


def run(parts, centered, trainer):
    """scores for every (split, lambda): dev CV scores on DEV and final scores on TEST"""
    out = {}
    for f, (tr, te) in enumerate(SPLITS):
        Z = featurize(parts, centered, tr)
        for lam in (LAMS if trainer == 'list' else [None]):
            if trainer == 'list':
                s = listwise(Z, tr, te, lam)
            else:
                s = indep(Z, tr, te, 0.05 if any(callable(p) for p in parts) else 1.0)
            out.setdefault(lam, (np.zeros(N), np.zeros(N)))[0 if f < 5 else 1][te] = s
    return out


L20 = lambda p: hidden([(p, 20)])
ALL = hidden([(p, l) for p in ('pre', 'res', 'mean') for l in (12, 16, 20, 24)])
CONFIGS = [  # name, parts, centered, trainer, uses hidden
    ('position (indep)', ['pos'], False, 'indep', False),
    ('position (list)', ['pos'], False, 'list', False),
    ('confidence + position (indep)', ['pos', 'conf'], False, 'indep', False),
    ('confidence + position (list, centered)', ['pos', 'conf'], True, 'list', False),
    ('text SVD + position (list, centered)', ['pos', text_svd], True, 'list', False),
    ('v1: res L20 + position (indep)', ['pos', L20('res')], False, 'indep', True),
    ('res L20 + position (list)', ['pos', L20('res')], False, 'list', True),
    ('res L20 + position (list, centered)', ['pos', L20('res')], True, 'list', True),
    ('mean L20 + position (list, centered)', ['pos', L20('mean')], True, 'list', True),
    ('pre L20 + position (list, centered)', ['pos', L20('pre')], True, 'list', True),
    ('pre+res+mean L20 + position (list, centered)', ['pos', hidden([(p, 20) for p in ('pre', 'res', 'mean')])],
     True, 'list', True),
    ('pre+res+mean L12-24 + position (indep)', ['pos', ALL], False, 'indep', True),
    ('pre+res+mean L12-24 + position (list, centered)', ['pos', ALL], True, 'list', True),
    ('pre+res+mean L12-24 + conf + position (list, centered)', ['pos', 'conf', ALL], True, 'list', True),
]
with parallel_backend('loky', inner_max_num_threads=3):
    OUT = Parallel(n_jobs=len(CONFIGS))(delayed(run)(p, c, t) for _, p, c, t, _ in CONFIGS)

order = np.argsort(sol, kind='stable')
keys, first = np.unique(sol[order], return_index=True)
groups = np.split(order, first[1:])
S_k = np.array([k[g[0]] for g in groups]); S_n = np.array([n[g[0]] for g in groups])
S_cat = np.array([cat[g[0]] for g in groups]); S_prob = np.array([prob[g[0]] for g in groups])
S_dev = np.array([split[g[0]] == 'train' for g in groups])


def hits(s):
    pred = np.array([e[g][np.argmax(s[g])] for g in groups])
    return pred == S_k, (pred == S_k) | (pred == S_k - 1)


def fmt(hit, near, m):
    sub = lambda c: f'{hit[m & c].mean():6.1%}'
    return (f'{hit[m].mean():6.1%} {near[m].mean():8.1%} {sub(S_n >= 3)} {sub(S_k >= 1)} {sub(S_cat == "ARITH")} '
            f'{sub(S_cat == "SETUP")}')


head = f"  {'input':56s} {'top-1':>6s} {'k or k-1':>8s} {'>=3 eq':>6s} {'k>=1':>6s} {'ARITH':>6s} {'SETUP':>6s}"
RES = {}
print(f'\n## DEV: 5-fold CV on train-split problems (solutions={S_dev.sum()}); listwise: best lambda on dev')
print(head)
print(f"  {'random (expected)':56s} {np.mean(1 / S_n[S_dev]):6.1%}")
for (name, _, _, _, uses_h), out in zip(CONFIGS, OUT):
    best = max(out, key=lambda lam: hits(out[lam][0])[0][S_dev].mean())
    RES[name] = (best, uses_h, out[best])
    lab = f'{name}' + (f' lam={best:g}' if best is not None else '')
    print(f'  {lab:56s} ' + fmt(*hits(out[best][0]), S_dev))

sel = max((n_ for n_ in RES if RES[n_][1]), key=lambda n_: hits(RES[n_][2][0])[0][S_dev].mean())
base = max((n_ for n_ in RES if not RES[n_][1]), key=lambda n_: hits(RES[n_][2][0])[0][S_dev].mean())
print(f'\nselected on dev: hidden = {sel}; best non-hidden = {base}')

T = ~S_dev
print(f'\n## TEST: train on all dev, evaluate once on test-split problems (solutions={T.sum()})')
print(head)
print(f"  {'random (expected)':56s} {np.mean(1 / S_n[T]):6.1%}")
H = {}
for name in RES:
    H[name] = hits(RES[name][2][1])[0]
    mark = ' <- selected' if name == sel else ' <- best non-hidden' if name == base else ''
    print(f'  {name:56s} ' + fmt(*hits(RES[name][2][1]), T) + mark)

rng = np.random.default_rng(0)
pk = np.unique(S_prob[T])
gi = {p: np.where(T & (S_prob == p))[0] for p in pk}
for B in [base, 'position (list)', 'v1: res L20 + position (indep)']:
    d = []
    for _ in range(1000):
        ix = np.concatenate([gi[p] for p in rng.choice(pk, len(pk))])
        d.append(H[sel][ix].mean() - H[B][ix].mean())
    print(f'  bootstrap TEST selected - {B}: {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')

# out-of-sample predicted location per solution (dev: CV fold predictions; test: final model), for later checks
P = {}
for name in RES:
    s = RES[name][2][0].copy()
    s[TEST] = RES[name][2][1][TEST]
    P[name] = {kk: int(e[g][np.argmax(s[g])]) for kk, g in zip(keys, groups)}
json.dump(dict(k={kk: int(v) for kk, v in zip(keys, S_k)}, cat={kk: str(v) for kk, v in zip(keys, S_cat)},
               selected=sel, best_non_hidden=base, pred=P), open(f'{D}/preds_loc2.json', 'w'))
