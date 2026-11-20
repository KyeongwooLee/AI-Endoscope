"""Is the arithmetic-heavy cluster of the first-wrong-equation hidden an internal signal, or just the equation's shape?

Rows: rep step_raw of cluster_errors.py. The target cluster is the one with the highest rule ARITH share on TRAIN.
Shape = what anyone can read off the equation's left side: operator counts, number of terms, digits, decimals, size of
the true value, position in the solution. The written result is not used (result vs true value IS the ARITH label).
Scores, each compared on the same rows and metric (AUROC, ARITH / audited CALC = positive):
  cluster       minus the distance to the target cluster's centroid (no labels used in fitting)
  shape         logistic regression on shape
  confidence    logistic regression on the 10 confidence features of that equation
  hidden eq     probe on hidden L20 at '=' (left side written, result not yet)
  hidden mean   probe on hidden L20, mean over the equation's tokens
  mean + shape  both
Train = 5-fold over problems, test = fitted on all train. Paired bootstrap over test problems.
usage: python cluster_check.py <out_dir> <loc_feat_dir> <k>
"""
import collections, json, os, pickle, re, sys, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations

warnings.filterwarnings('ignore')
OUT, LOC, K = sys.argv[1], sys.argv[2], int(sys.argv[3])
T = pickle.load(open(f'{OUT}/rows.pkl', 'rb'))
d = np.load(f'{OUT}/rep_step_raw.npz')
idx, it, lab, lt, c = d['idx'], d['idx_t'], d[f'lab_{K}'], d[f'lab_t_{K}'], d[f'c_{K}']
locs = sorted(int(f[9:-5]) for f in os.listdir(LOC) if f.startswith('meta_loc_'))
lm = [json.load(open(f'{LOC}/meta_loc_{s}.json')) for s in locs]
lay = lm[0]['layers'].index(20)
get = lambda p: np.concatenate([np.load(f'{LOC}/H_loc_{s}.npy', mmap_mode='r')[:, lm[0]['positions'].index(p), lay]
                                for s in locs]).astype(np.float32)
H_eq, H_mean = get('eq'), get('mean')
lmeta = [m for x in lm for m in x['meta']]
sol = collections.defaultdict(list)
for j, m in enumerate(lmeta):
    sol[(m['split'], m['pid'], m['sample'])].append(j)
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def shape(i):
    key = T['key'][i]
    lhs, lv = equations(T['text'][i])[T['k'][key]][:2]
    nums = NUM.findall(lhs)
    digits = [len(re.sub(r'\D', '', x)) for x in nums]
    return [lhs.count('+'), lhs.count('-'), lhs.count('*'), lhs.count('/'), len(nums), max(digits, default=0), sum(digits),
            any('.' in x for x in nums), np.log10(abs(lv) + 1), lv != round(lv), T['k'][key], T['n_eq'][i]]


NAMES = ['+', '-', '*', '/', 'terms', 'max_digits', 'digits', 'decimal', 'log_value', 'non_integer', 'k', 'n_eq']
step_of = lambda ii: np.array([sol[T['key'][i]][T['k'][T['key'][i]]] for i in ii])
s_tr, s_te = step_of(idx), step_of(it)
X = {'shape': (np.array([shape(i) for i in idx], float), np.array([shape(i) for i in it], float)),
     'confidence': (np.array([lmeta[j]['conf'] for j in s_tr]), np.array([lmeta[j]['conf'] for j in s_te])),
     'hidden eq': (H_eq[s_tr], H_eq[s_te]), 'hidden mean': (H_mean[s_tr], H_mean[s_te])}
X['mean + shape'] = (np.c_[X['hidden mean'][0], X['shape'][0]], np.c_[X['hidden mean'][1], X['shape'][1]])
cat, cat_t, aud = T['cat'][idx], T['cat'][it], T['aud'][it]
y, y_t, y_a = (cat == 'ARITH').astype(int), (cat_t == 'ARITH').astype(int), (aud == 'CALC').astype(int)
g = np.array([f'{T["key"][i][0]}-{T["key"][i][1]}' for i in idx])
g_t = np.array([f'{T["key"][i][0]}-{T["key"][i][1]}' for i in it])
target = int(np.argmax([y[lab == j].mean() for j in range(K)]))
print(f'k={K}: target cluster {target}, train {(lab == target).sum()} rows, rule ARITH {y[lab == target].mean():.1%} vs {y.mean():.1%}; '
      f'test {(lt == target).sum()} rows, rule ARITH {y_t[lt == target].mean():.1%} vs {y_t.mean():.1%}, '
      f'audited CALC {y_a[lt == target].mean():.1%} vs {y_a.mean():.1%}, holds {y_a[lt == target].sum() / y_a.sum():.0%} of audited CALC')

print('\n## equation shape per cluster, train (share of equations with the operator; mean terms, digits)')
print(f"{'c':>2s} {'n':>5s} {'ARITH':>6s} {'>=2 +':>6s} {'any +':>6s} {'any -':>6s} {'any *':>6s} {'any /':>6s} {'terms':>6s} {'digits':>6s} {'decim':>6s} {'k=0':>6s}")
Sh = X['shape'][0]
for j in range(K):
    m = lab == j
    print(f'{j:2d} {m.sum():5d} {y[m].mean():6.1%} {(Sh[m, 0] >= 2).mean():6.1%} {(Sh[m, 0] > 0).mean():6.1%} {(Sh[m, 1] > 0).mean():6.1%} '
          f'{(Sh[m, 2] > 0).mean():6.1%} {(Sh[m, 3] > 0).mean():6.1%} {Sh[m, 4].mean():6.2f} {Sh[m, 6].mean():6.1f} {Sh[m, 7].mean():6.1%} {(Sh[m, 10] == 0).mean():6.1%}')
print(f"ARITH rate by number of '+' on the left side, train: " + '  '.join(
    f"{'>=4' if n == 4 else n}: {y[(Sh[:, 0] >= 4) if n == 4 else (Sh[:, 0] == n)].mean():.1%} (n={((Sh[:, 0] >= 4) if n == 4 else (Sh[:, 0] == n)).sum()})"
    for n in range(5)))


def fit(name):
    a, b = X[name]
    if name == 'shape' or name == 'confidence':
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced'))
    else:
        mk = lambda: make_pipeline(StandardScaler(), PCA(256, random_state=0),
                                   LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced'))
    oof = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(a, y, g):
        oof[te] = mk().fit(a[tr], y[tr]).decision_function(a[te])
    return oof, mk().fit(a, y).decision_function(b)


Sc = {'cluster': (-((d['Z'] - c[target]) ** 2).sum(1), -((d['Zt'] - c[target]) ** 2).sum(1))}
for n in X:
    Sc[n] = fit(n)
N = int((lt == target).sum())
print(f'\n## AUROC on the same rows (train {len(idx)}, test {len(it)}); top{N} = audited CALC share among the {N} highest test scores')
print(f"{'score':14s} {'train ARITH':>11s} {'test ARITH':>10s} {'test CALC':>9s} {f'top{N} CALC':>10s}")
for n, (a, b) in Sc.items():
    print(f'{n:14s} {roc_auc_score(y, a):11.3f} {roc_auc_score(y_t, b):10.3f} {roc_auc_score(y_a, b):9.3f} {y_a[np.argsort(-b)[:N]].mean():10.1%}')
print(f"{'cluster member':14s} {roc_auc_score(y, lab == target):11.3f} {roc_auc_score(y_t, lt == target):10.3f} {roc_auc_score(y_a, lt == target):9.3f} {y_a[lt == target].mean():10.1%}")

print('\n## paired bootstrap over test problems, AUROC difference on audited CALC')
rng = np.random.default_rng(0)
keys = np.unique(g_t)
gi = {k: np.where(g_t == k)[0] for k in keys}
samples = [np.concatenate([gi[k] for k in rng.choice(keys, len(keys))]) for _ in range(2000)]
for a, b in [('hidden mean', 'shape'), ('hidden eq', 'shape'), ('mean + shape', 'shape'), ('cluster', 'shape'),
             ('hidden mean', 'confidence'), ('hidden mean', 'cluster')]:
    A, B = Sc[a][1], Sc[b][1]
    ds = [roc_auc_score(y_a[s], A[s]) - roc_auc_score(y_a[s], B[s]) for s in samples if y_a[s].sum()]
    print(f'  {a} - {b}: {roc_auc_score(y_a, A) - roc_auc_score(y_a, B):+.3f} [{np.percentile(ds, 2.5):+.3f}, {np.percentile(ds, 97.5):+.3f}]')
