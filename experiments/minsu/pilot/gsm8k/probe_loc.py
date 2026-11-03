"""Can hidden states find WHERE the model's own wrong GSM8K solution first goes wrong?

Steps = checkable equations (extract_loc.py). Label k = first equation whose result appears in no correct solution of
the same problem (auto_types.py). A per-step scorer ('is this the first wrong step?', logistic regression trained on
all steps of the training problems) scores every step; the predicted location is the argmax within the solution.
Metric: top-1 accuracy; 'k or k-1' also counts the step before (errors that start in the prose before the equation).
Every model gets the step's position (index, #steps, relative position), so 'position' is the baseline to beat; the
others add confidence (step logprobs), text (TF-IDF of the step's line) or hidden states.
Primary comparison fixed in advance: hidden res L20 + position vs position, vs confidence + position, vs text +
position; paired bootstrap over problems. Problem-grouped 5-fold CV.
usage: python probe_loc.py <loc_dir>
"""
import json, os, sys, warnings
import numpy as np, scipy.sparse as sp
from joblib import Parallel, delayed, parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
e, n, k = (np.array([m[f] for m in M]) for f in ('e', 'n_eq', 'k'))
cat = np.array([m['cat'] for m in M])
y = (e == k).astype(int)
C = np.array([m['conf'] for m in M], dtype=np.float32)
TEXT = np.array([m['line'] for m in M], dtype=object)
prev = np.where(e > 0, np.arange(N) - 1, np.arange(N))
assert (sol[prev] == sol).all() and (e[prev] == np.maximum(e - 1, 0)).all()
oh = lambda v: np.eye(6)[np.minimum(v, 5)]
POSF = np.hstack([np.stack([e, n, e / np.maximum(n - 1, 1)], 1), oh(e), oh(n - 1 - e)]).astype(np.float32)
FOLDS = list(GroupKFold(5).split(np.zeros(N), y, prob))
print(f'skipped solutions (not aligned): {sum(d["skipped"] for d in metas)}')
print(f'steps={N}  solutions={len(set(sol))}  problems={len(set(prob))}  mean steps/solution={N / len(set(sol)):.1f}')


def cv(builder):
    s = np.zeros(N)
    for tr, te in FOLDS:
        s[te] = builder(tr, te)
    return s


LR = lambda c: LogisticRegression(C=c, max_iter=3000, class_weight='balanced')


def scalar_model(with_conf):
    def b(tr, te):
        X = np.hstack([POSF, C]) if with_conf else POSF
        Z = StandardScaler().fit(X[tr]).transform(X)
        return LR(1.0).fit(Z[tr], y[tr]).decision_function(Z[te])
    return b


def text_model(tr, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                          token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
    P = StandardScaler().fit(POSF[tr]).transform(POSF)
    X = sp.hstack([vec.fit(TEXT[tr]).transform(TEXT), sp.csr_matrix(P)]).tocsr()
    return LR(1.0).fit(X[tr], y[tr]).decision_function(X[te])


def hidden_model(p, l, use_pos=True, with_conf=False, delta=False):
    def b(tr, te):
        X = np.load(HF, mmap_mode='r')[:, POS.index(p), LAYERS.index(l)].astype(np.float32)
        if delta:
            X = np.hstack([X, X - X[prev]])  # change from the previous step
        parts = [make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr]).transform(X)]
        if use_pos:
            parts.append(StandardScaler().fit(POSF[tr]).transform(POSF))
        if with_conf:
            parts.append(StandardScaler().fit(C[tr]).transform(C))
        Z = np.hstack(parts)
        return LR(0.05).fit(Z[tr], y[tr]).decision_function(Z[te])
    return b


configs = [('position', scalar_model(False)), ('confidence + position', scalar_model(True)),
           ('text TF-IDF + position', text_model)]
configs += [(f'hidden {p} L{l} + position', hidden_model(p, l)) for p in POS for l in LAYERS]
configs += [('hidden res L20 (no position)', hidden_model('res', 20, use_pos=False)),
            ('hidden res L20 + delta + position', hidden_model('res', 20, delta=True)),
            ('hidden res L20 + conf + position', hidden_model('res', 20, with_conf=True))]
with parallel_backend('loky', inner_max_num_threads=4):
    S = Parallel(n_jobs=len(configs))(delayed(cv)(b) for _, b in configs)
R = dict(zip([c[0] for c in configs], S))

order = np.argsort(sol, kind='stable')
keys, first = np.unique(sol[order], return_index=True)
groups = np.split(order, first[1:])
S_k = np.array([k[g[0]] for g in groups]); S_n = np.array([n[g[0]] for g in groups])
S_cat = np.array([cat[g[0]] for g in groups]); S_prob = np.array([prob[g[0]] for g in groups])


def preds(s):
    return np.array([e[g][np.argmax(s[g])] for g in groups])


def row(name, pred):
    hit, near = pred == S_k, (pred == S_k) | (pred == S_k - 1)
    sub = lambda m: f'{hit[m].mean():6.1%}'
    return (f'  {name:36s} {hit.mean():6.1%} {near.mean():8.1%} {sub(S_n >= 3)} {sub(S_k >= 1)} '
            f'{sub(S_cat == "ARITH")} {sub(S_cat == "SETUP")}'), hit


print(f'\n## first wrong step, top-1 accuracy (solutions={len(groups)}; ARITH={np.sum(S_cat == "ARITH")}, '
      f'SETUP={np.sum(S_cat == "SETUP")}, k>=1: {np.sum(S_k >= 1)})')
print(f"  {'input':36s} {'top-1':>6s} {'k or k-1':>8s} {'>=3 eq':>6s} {'k>=1':>6s} {'ARITH':>6s} {'SETUP':>6s}")
print(f"  {'random (expected)':36s} {np.mean(1 / S_n):6.1%}")
print(row('always first step', np.zeros(len(groups), int))[0])
print(row('always last step', S_n - 1)[0])
HIT = {}
for name, _ in configs:
    line, HIT[name] = row(name, preds(R[name]))
    print(line)

A = 'hidden res L20 + position'
print(f'\n## accuracy by true location k: {A} vs position')
for kk in range(4):
    m = (S_k == kk) if kk < 3 else (S_k >= 3)
    print(f'  k={kk if kk < 3 else "3+"} (n={m.sum()}): hidden {HIT[A][m].mean():.1%}  position {HIT["position"][m].mean():.1%}')

rng = np.random.default_rng(0)
pk = np.unique(S_prob)
gi = {p: np.where(S_prob == p)[0] for p in pk}
for B in ['position', 'confidence + position', 'text TF-IDF + position']:
    d = []
    for _ in range(1000):
        ix = np.concatenate([gi[p] for p in rng.choice(pk, len(pk))])
        d.append(HIT[A][ix].mean() - HIT[B][ix].mean())
    print(f'  bootstrap {A} - {B}: {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
