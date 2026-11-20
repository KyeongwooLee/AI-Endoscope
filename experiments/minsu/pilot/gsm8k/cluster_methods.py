"""Does anything besides k-means find error groups? Other methods on the representations saved by cluster_errors.py.

Everything is fitted on train wrong answers (saved PCA(50) coordinates). Test rows and correct solutions take the
majority label of their 15 nearest train rows (noise included).
Methods:
  kmeans        baseline, k = 3,5,8
  white         k-means after giving every axis equal variance, so small-variance axes count as much as large ones
  gmm_diag      Gaussian mixture, diagonal covariance, all dims
  gmm_full      Gaussian mixture, full covariance, top 20 dims (elongated clusters)
  ward          hierarchical, Ward linkage
  louvain       communities of the 15-nearest-neighbour graph
  hdbscan       density clusters, points in no cluster = noise; min_cluster_size m on the top d dims
  umap_hdbscan  UMAP to 5 dims (n_neighbors nn, min_dist 0), then HDBSCAN
Extra representations, contrastive PCA: the directions along which wrong answers vary most RELATIVE to correct solutions
(generalized eigenvectors of cov(wrong) vs cov(correct), top 10), so style shared with correct solutions drops out without
listing it. contrast_d is built from diff, contrast_q from qres. Held-out variance ratios (5 folds over problems) show
whether such directions are real.
Columns:
  n, noise, top  clusters, share of noise, share of the largest cluster
  sil, null      silhouette in the same 50 dims (contrast: its 10 dims) without noise; same method on the data with every
                 axis shuffled separately (keeps each axis' distribution, destroys any joint structure)
  nulln          clusters found on the shuffled data
  stab           ARI with the same method on 80% of problems, on shared non-noise rows
  q              balanced accuracy predicting the cluster from question-only hidden (held-out problems, chance 1/n)
  style len      AMI with first word of the response; share of log-length variance explained
  rule audit     AMI with rule type (train) / AI-audited class (test), noise excluded
  a_lift a_test  cluster (n >= 30) with the highest rule ARITH rate on train: rate / overall rate; audited CALC rate of
                 that cluster on test, then its test size. t_* the same for TARGET
  V              Cramer's V of cluster vs wrong/correct
usage: python cluster_methods.py <feat_dir> <out_dir>
"""
import json, os, pickle, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.linalg import eigh
from sklearn.cluster import AgglomerativeClustering, HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score, roc_auc_score, \
    silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, OUT = sys.argv[1:3]
T = pickle.load(open(f'{OUT}/rows.pkl', 'rb'))
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
Q = np.asarray(np.load(f'{F}/H_all.npy', mmap_mode='r')[:, metas[0]['positions'].index('q_end'), metas[0]['layers'].index(20)],
               dtype=np.float32)
prob = np.array([f'{k[0]}-{k[1]}' for k in T['key']])
REPS = ['raw', 'qres_ns', 'diff', 'diff_ns', 'step', 'step_raw']
CONFIGS = [(m, dict(k=k)) for m in ('kmeans', 'white', 'gmm_diag', 'gmm_full', 'ward') for k in (3, 5, 8)] + [('louvain', {})] + \
          [('hdbscan', dict(d=d, m=m)) for d in (10, 50) for m in (15, 50, 150)] + \
          [('umap_hdbscan', dict(nn=nn, m=m)) for nn in (15, 50) for m in (30, 100)]


def contrast(Z, Zp, top=10):
    w, V = eigh(np.cov(Z.T), np.cov(Zp.T) + 1e-6 * np.eye(Z.shape[1]))
    return w[::-1], V[:, ::-1][:, :top]


data = {}
for name in REPS:
    d = np.load(f'{OUT}/rep_{name}.npz')
    data[name] = {k: d[k] for k in ('Z', 'Zt', 'Zp', 'idx', 'idx_t', 'idx_p')}

print('## contrastive PCA: variance of wrong answers / variance of correct solutions along each direction')
corr = lambda a, b: np.corrcoef(a, b)[0, 1]
for new, src in (('contrast_d', 'diff'), ('contrast_q', 'qres')):
    d = np.load(f'{OUT}/rep_{src}.npz')
    Z, Zt, Zp, idx = d['Z'], d['Zt'], d['Zp'], d['idx']
    idx_p = d['idx_p']
    w, V = contrast(Z, Zp)
    held = []
    gz, gp = prob[idx], prob[idx_p]
    ups = np.unique(np.r_[gz, gp])
    for a, b in GroupKFold(5).split(ups, groups=ups):
        tr, te = set(ups[a]), set(ups[b])
        mz, mp = np.isin(gz, list(tr)), np.isin(gp, list(tr))
        _, Vf = contrast(Z[mz], Zp[mp])
        held.append((Z[~mz] @ Vf).var(0) / (Zp[~mp] @ Vf).var(0))
    print(f'{new} (from {src}; wrong {len(Z)}, correct {len(Zp)}): overall variance ratio {np.trace(np.cov(Z.T)) / np.trace(np.cov(Zp.T)):.2f}')
    print('  train    ' + ' '.join(f'{x:5.2f}' for x in w[:10]))
    print('  held-out ' + ' '.join(f'{x:5.2f}' for x in np.mean(held, 0)))
    m = Z.mean(0)
    data[new] = dict(Z=(Z - m) @ V, Zt=(Zt - m) @ V, Zp=(Zp - m) @ V, idx=idx, idx_t=d['idx_t'], idx_p=idx_p)
    Zc = data[new]['Z']
    cats = T['cat'][idx]
    print(f"  {'dir':>3s} {'R2_q':>6s} {'len':>6s} {'nll':>6s} {'latex':>6s} {'step':>6s} {'bold':>6s} {'A/S':>6s} {'T/S':>6s}")
    p5 = np.zeros((len(Zc), 5))
    q_model = lambda: make_pipeline(StandardScaler(), PCA(256, random_state=0), Ridge(alpha=100.0))
    for a, b in GroupKFold(5).split(Zc, groups=gz):
        p5[b] = q_model().fit(Q[idx][a], Zc[a][:, :5]).predict(Q[idx][b])
    for j in range(5):
        z = Zc[:, j]
        au = []
        for c in ('ARITH', 'TARGET'):
            mk = np.isin(cats, [c, 'SETUP'])
            x = roc_auc_score(cats[mk] == c, z[mk])
            au.append(max(x, 1 - x))
        print(f"  {j + 1:3d} {1 - ((z - p5[:, j]) ** 2).sum() / ((z - z.mean()) ** 2).sum():6.2f} {corr(z, T['loglen'][idx]):6.2f} {corr(z, T['nll'][idx]):6.2f} "
              f"{corr(z, T['latex'][idx]):6.2f} {corr(z, T['step'][idx]):6.2f} {corr(z, T['bold'][idx]):6.2f} {au[0]:6.2f} {au[1]:6.2f}")
    line = []
    for c in ('ARITH', 'TARGET'):
        mk = np.isin(cats, [c, 'SETUP'])
        y, s = (cats[mk] == c).astype(int), np.zeros(mk.sum())
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(s, y, gz[mk]):
            s[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced')).fit(Zc[mk][a], y[a]).decision_function(Zc[mk][b])
        line.append(f'{c} vs SETUP {roc_auc_score(y, s):.3f}')
    print('  supervised probe on these 10 directions (AUROC, held-out problems): ' + '  '.join(line))
sys.stdout.flush()


def cluster(Z, method, p, seed):
    if method == 'kmeans':
        return KMeans(p['k'], n_init=10, random_state=seed).fit_predict(Z)
    if method == 'white':
        return KMeans(p['k'], n_init=10, random_state=seed).fit_predict(Z / Z.std(0))
    if method == 'gmm_diag':
        return GaussianMixture(p['k'], covariance_type='diag', reg_covar=1e-3, n_init=3, random_state=seed).fit(Z).predict(Z)
    if method == 'gmm_full':
        X = Z[:, :20]
        return GaussianMixture(p['k'], covariance_type='full', reg_covar=1e-3, n_init=3, random_state=seed).fit(X).predict(X)
    if method == 'ward':
        return AgglomerativeClustering(p['k'], linkage='ward').fit_predict(Z)
    if method == 'louvain':
        import networkx as nx
        ind = NearestNeighbors(n_neighbors=16).fit(Z).kneighbors(Z, return_distance=False)
        G = nx.Graph()
        G.add_nodes_from(range(len(Z)))
        G.add_edges_from((i, j) for i, row in enumerate(ind) for j in row[1:] if i != j)
        lab = np.zeros(len(Z), dtype=int)
        for c, nodes in enumerate(nx.community.louvain_communities(G, seed=seed)):
            lab[list(nodes)] = c
        return lab
    if method == 'hdbscan':
        return HDBSCAN(min_cluster_size=p['m'], min_samples=5).fit_predict(Z[:, :p['d']])
    if method == 'umap_hdbscan':
        import umap
        E = umap.UMAP(n_neighbors=p['nn'], min_dist=0.0, n_components=5, random_state=seed, n_jobs=1).fit_transform(Z)
        return HDBSCAN(min_cluster_size=p['m'], min_samples=5).fit_predict(E)
    raise ValueError(method)


def cramers_v(a, b):
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    t = t[t.sum(1) > 0]
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return np.sqrt(((t - e) ** 2 / e).sum() / t.sum() / (min(t.shape) - 1))


def sil(Z, lab):
    m = lab >= 0
    return silhouette_score(Z[m], lab[m]) if len(set(lab[m])) > 1 else float('nan')


def run(name, method, p, d):
    Z, g, cats, aud = d['Z'], d['g'], d['cat'], d['aud']
    lab = cluster(Z, method, p, 0)
    rng = np.random.default_rng(0)
    ns, nn, nz = [], [], []
    for r in range(2):
        Zn = np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])
        ln = cluster(Zn, method, p, r + 1)
        ns.append(sil(Zn, ln))
        nn.append(len(set(ln)) - (-1 in ln))
        nz.append((ln < 0).mean())
    st = []
    ug = np.unique(g)
    for r in range(3):
        m = np.isin(g, rng.choice(ug, int(.8 * len(ug)), replace=False))
        l2 = cluster(Z[m], method, p, r + 1)
        ok = (lab[m] >= 0) & (l2 >= 0)
        st.append(adjusted_rand_score(lab[m][ok], l2[ok]) if ok.sum() > 10 else float('nan'))
    n = len(set(lab)) - (-1 in lab)
    out = dict(rep=name, method=method, p=p, n=n, noise=(lab < 0).mean(), top=np.bincount(lab[lab >= 0]).max() / len(lab) if n else 0,
               sil=sil(Z, lab), null=np.nanmean(ns), nulln=np.mean(nn), nullz=np.mean(nz), stab=np.nanmean(st), lab=lab)
    nnb = KNeighborsClassifier(15).fit(Z, lab)
    lt = nnb.predict(d['Zt'])
    lp = nnb.predict(d['Zp']) if len(d['Zp']) else np.array([], dtype=int)
    out.update(lab_t=lt, lab_p=lp)
    for key in ('q', 'style', 'len', 'rule', 'audit', 'a_lift', 'a_test', 'a_n', 't_lift', 't_test', 't_n', 'V'):
        out[key] = float('nan')
    if n < 2:
        return out
    m = lab >= 0
    keep = m & np.isin(lab, [c for c in set(lab[m]) if (lab == c).sum() >= 10])
    y = lab[keep]
    if len(set(y)) > 1:
        pr = np.zeros(len(y), dtype=int)
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g[keep]):
            pr[b] = make_pipeline(StandardScaler(), PCA(min(256, len(a) - 1), random_state=0),
                                  LogisticRegression(C=0.05, max_iter=500, class_weight='balanced')).fit(d['Q'][keep][a], y[a]).predict(d['Q'][keep][b])
        out['q'] = balanced_accuracy_score(y, pr)
    ll = d['loglen'][m]
    out['len'] = sum((lab[m] == c).sum() * (ll[lab[m] == c].mean() - ll.mean()) ** 2 for c in set(lab[m])) / ((ll - ll.mean()) ** 2).sum()
    out['style'] = adjusted_mutual_info_score(d['first'][m], lab[m])
    out['rule'] = adjusted_mutual_info_score(cats[m], lab[m])
    ma = (lt >= 0) & (aud != '')
    if len(set(lt[ma])) > 1:
        out['audit'] = adjusted_mutual_info_score(aud[ma], lt[ma])
    for pre, rule, audited in (('a', 'ARITH', 'CALC'), ('t', 'TARGET', 'TARGET')):
        if (cats == rule).sum() < 10:
            continue
        big = [c for c in set(lab[m]) if (lab == c).sum() >= 30]
        if big:
            c = max(big, key=lambda x: (cats[lab == x] == rule).mean())
            out[f'{pre}_lift'] = (cats[lab == c] == rule).mean() / (cats == rule).mean()
            sel = (lt == c) & (aud != '')
            out[f'{pre}_test'] = (aud[sel] == audited).mean() if sel.sum() else float('nan')
            out[f'{pre}_n'] = sel.sum()
    if len(lp):
        a, b = np.r_[lab[m], lp[lp >= 0]], np.r_[np.zeros(m.sum(), int), np.ones((lp >= 0).sum(), int)]
        out['V'] = cramers_v(a, b)
    return out


def bic_best(name, kind, Z, seed):
    X = Z if kind == 'diag' else Z[:, :20]
    rng = np.random.default_rng(seed)
    if seed:
        X = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    b = [GaussianMixture(k, covariance_type=kind, reg_covar=1e-3, n_init=2, random_state=0).fit(X).bic(X) for k in range(1, 13)]
    return name, kind, seed, int(np.argmin(b)) + 1, b[0] - min(b)


for name, d in data.items():
    d.update(g=prob[d['idx']], Q=Q[d['idx']], cat=T['cat'][d['idx']], first=T['first'][d['idx']], loglen=T['loglen'][d['idx']],
             aud=T['aud'][d['idx_t']])
    base = d['aud'][d['aud'] != '']
    print(f"{name}: train {len(d['Z'])}, rule ARITH {(d['cat'] == 'ARITH').mean():.1%}, TARGET {(d['cat'] == 'TARGET').mean():.1%}; "
          f"test audited CALC {(base == 'CALC').mean():.1%}, TARGET {(base == 'TARGET').mean():.1%} (n={len(base)})")

with parallel_backend('loky', inner_max_num_threads=1):
    bics = Parallel(n_jobs=64)(delayed(bic_best)(n, kind, data[n]['Z'], s) for n in data for kind in ('diag', 'full') for s in (0, 1))
    res = Parallel(n_jobs=64)(delayed(run)(n, m, p, data[n]) for n in data for m, p in CONFIGS)

print('\n## Gaussian mixture: number of components (1..12) chosen by BIC, real data vs every axis shuffled')
for name in data:
    print(f'{name:11s} ' + '  '.join(f"{kind}: real {[b[3] for b in bics if b[:3] == (name, kind, 0)][0]} / shuffled {[b[3] for b in bics if b[:3] == (name, kind, 1)][0]}" for kind in ('diag', 'full')))

print('\n## all configurations')
cols = ['n', 'noise', 'top', 'sil', 'null', 'nulln', 'nullz', 'stab', 'q', 'style', 'len', 'rule', 'audit', 'a_lift', 'a_test', 'a_n', 't_lift', 't_test', 't_n', 'V']
print(f"{'rep':11s} {'config':24s} " + ' '.join(f'{c:>6s}' for c in cols))
for r in res:
    conf = r['method'] + ''.join(f'_{k}{v}' for k, v in r['p'].items())
    print(f"{r['rep']:11s} {conf:24s} " + ' '.join(f'{r[c]:6.0f}' if c in ('n', 'a_n', 't_n') else f'{r[c]:6.3f}' for c in cols))
    # noise becomes the last label so cluster_detail.py can read this like any other clustering
    K = r['n'] + (-1 in r['lab'])
    fix = lambda x: np.where(x < 0, K - 1, x)
    lab = fix(r['lab'])
    Z = data[r['rep']]['Z']
    if r['n'] >= 2:
        np.savez(f"{OUT}/rep_{r['rep']}__{conf}.npz", Z=Z.astype(np.float32), idx=data[r['rep']]['idx'], idx_t=data[r['rep']]['idx_t'],
                 idx_p=data[r['rep']]['idx_p'], **{f'lab_{K}': lab, f'lab_t_{K}': fix(r['lab_t']), f'lab_p_{K}': fix(r['lab_p']),
                                                   f'c_{K}': np.stack([Z[lab == c].mean(0) for c in range(K)])})
