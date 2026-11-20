"""Three checks on cluster_methods.py results.

1. Fair wrong/correct control: in cluster_methods.py wrong answers keep the method's own label while correct solutions get a
   15-nearest-neighbour label, and that difference alone inflates V for methods whose labels are not local (GMM). Here both
   get a nearest-neighbour label (wrong answers: from other problems only).
2. Contrastive directions: outliers or a broad effect? Variance ratio vs interquartile-range ratio on held-out problems,
   against random directions; extreme rows are written to contrast_extremes.txt.
3. Is the arithmetic-heavy cluster of each method the long-addition cluster, and does equation shape match it on test?
usage: python cluster_methods_check.py <out_dir>
"""
import glob, os, pickle, re, sys, warnings
import numpy as np
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations

warnings.filterwarnings('ignore')
OUT = sys.argv[1]
T = pickle.load(open(f'{OUT}/rows.pkl', 'rb'))
prob = np.array([f'{k[0]}-{k[1]}' for k in T['key']])


def cramers_v(a, b):
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    t = t[t.sum(1) > 0]
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return np.sqrt(((t - e) ** 2 / e).sum() / t.sum() / (min(t.shape) - 1))


print('## 1. V with nearest-neighbour labels on both sides (reported V in parentheses is not comparable across methods)')
print(f"{'rep':11s} {'config':24s} {'V fair':>7s}")
for f in sorted(glob.glob(f'{OUT}/rep_*__*.npz')):
    d = np.load(f)
    rep, conf = os.path.basename(f)[4:-4].split('__')
    if rep in ('step', 'step_raw') or not any(conf.startswith(x) for x in ('kmeans_k5', 'white_k5', 'gmm_diag_k5', 'gmm_full_k5', 'ward_k5', 'louvain')):
        continue
    K = int([k for k in d.files if k.startswith('lab_t_')][0][6:])
    Z, lab, lp, idx = d['Z'], d[f'lab_{K}'], d[f'lab_p_{K}'], d['idx']
    cv = np.zeros(len(lab), dtype=int)
    for a, b in GroupKFold(5).split(Z, groups=prob[idx]):
        cv[b] = KNeighborsClassifier(15).fit(Z[a], lab[a]).predict(Z[b])
    print(f'{rep:11s} {conf:24s} {cramers_v(np.r_[cv, lp], np.r_[np.zeros(len(cv), int), np.ones(len(lp), int)]):7.3f}')

print('\n## 2. contrastive directions on held-out problems: wrong / correct, variance and squared IQR')
iqr = lambda x: np.subtract(*np.percentile(x, [75, 25], axis=0))
out = open(f'{OUT}/contrast_extremes.txt', 'w', encoding='utf-8')
for new, src in (('contrast_d', 'diff'), ('contrast_q', 'qres')):
    d = np.load(f'{OUT}/rep_{src}.npz')
    Z, Zp, gz, gp = d['Z'], d['Zp'], prob[d['idx']], prob[d['idx_p']]
    ups = np.unique(np.r_[gz, gp])
    rng = np.random.default_rng(0)
    var, rob, base = [], [], []
    for a, b in GroupKFold(5).split(ups, groups=ups):
        tr = list(ups[a])
        mz, mp = np.isin(gz, tr), np.isin(gp, tr)
        _, V = eigh(np.cov(Z[mz].T), np.cov(Zp[mp].T) + 1e-6 * np.eye(Z.shape[1]))
        V = V[:, ::-1][:, :10]
        A, B = Z[~mz] @ V, Zp[~mp] @ V
        var.append(A.var(0) / B.var(0))
        rob.append((iqr(A) / iqr(B)) ** 2)
        R = rng.normal(size=(Z.shape[1], 200))
        base.append((iqr(Z[~mz] @ R) / iqr(Zp[~mp] @ R)) ** 2)
    print(f'{new}: random directions, squared IQR ratio median {np.median(base):.2f}')
    print('  variance ' + ' '.join(f'{x:5.2f}' for x in np.mean(var, 0)))
    print('  IQR^2    ' + ' '.join(f'{x:5.2f}' for x in np.mean(rob, 0)))
    w, V = eigh(np.cov(Z.T), np.cov(Zp.T) + 1e-6 * np.eye(Z.shape[1]))
    V = V[:, ::-1][:, :4]
    S = (Z - Z.mean(0)) @ V
    idx = d['idx']
    print('  excess kurtosis  ' + ' '.join(f'{(((S[:, j] - S[:, j].mean()) / S[:, j].std()) ** 4).mean() - 3:7.1f}' for j in range(4)))
    for j in range(4):
        for side, order in (('high', np.argsort(-S[:, j])), ('low', np.argsort(S[:, j]))):
            out.write(f'\n######## {new} direction {j + 1} {side}\n')
            for x in order[:6]:
                i = idx[x]
                out.write(f"\n--- {T['key'][i]} rule={T['cat'][i]} score={S[x, j]:.1f} len={np.exp(T['loglen'][i]):.0f}\nQ: {T['question'][i][:160]}\nA: {T['text'][i][:420]}\n... {T['text'][i][-160:]}\n")
out.close()

print('\n## 3. arithmetic-heavy cluster of each method on the first wrong equation (test: audited CALC among its rows)')
d0 = np.load(f'{OUT}/rep_step_raw.npz')
idx0, it0 = d0['idx'], d0['idx_t']
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def shape(i):
    key = T['key'][i]
    lhs, lv = equations(T['text'][i])[T['k'][key]][:2]
    nums = NUM.findall(lhs)
    digits = [len(re.sub(r'\D', '', x)) for x in nums]
    return [lhs.count('+'), lhs.count('-'), lhs.count('*'), lhs.count('/'), len(nums), max(digits, default=0), sum(digits),
            any('.' in x for x in nums), np.log10(abs(lv) + 1), lv != round(lv), T['k'][key], T['n_eq'][i]]


St, Se = np.array([shape(i) for i in idx0], float), np.array([shape(i) for i in it0], float)
y, ya = (T['cat'][idx0] == 'ARITH').astype(int), (T['aud'][it0] == 'CALC').astype(int)
sc = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')).fit(St, y).decision_function(Se)
print(f"{'rep':9s} {'config':24s} {'n train':>7s} {'ARITH':>6s} {'>=2 +':>6s} {'n test':>6s} {'CALC':>6s} {'shape top-n CALC':>16s}")
for f in sorted(glob.glob(f'{OUT}/rep_step*__*.npz')):
    d = np.load(f)
    rep, conf = os.path.basename(f)[4:-4].split('__')
    K = int([k for k in d.files if k.startswith('lab_t_')][0][6:])
    lab, lt = d[f'lab_{K}'], d[f'lab_t_{K}']
    has_noise = conf.startswith(('hdbscan', 'umap')) and (d[f'c_{K}'].shape[0] == K)
    big = [c for c in range(K) if (lab == c).sum() >= 30]
    if not big:
        continue
    c = max(big, key=lambda x: y[lab == x].mean())
    n = int((lt == c).sum())
    if y[lab == c].mean() / y.mean() < 3 or n < 15:
        continue
    print(f"{rep:9s} {conf:24s} {(lab == c).sum():7d} {y[lab == c].mean():6.1%} {(St[lab == c, 0] >= 2).mean():6.1%} {n:6d} {ya[lt == c].mean():6.1%} {ya[np.argsort(-sc)[:n]].mean():16.1%}")
