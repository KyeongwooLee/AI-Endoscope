"""Do the model's own wrong GSM8K answers form error groups in hidden space, or only topics / writing styles?

No labels are used for fitting: k-means on PCA(50) of the TRAIN-split wrong answers. Test-split wrong answers are only
assigned afterwards and compared with the AI class audit. Rule NOEQ is excluded (length alone separates it).
Representations (hidden L20):
  raw       r_mean as is
  qres      r_mean minus what the question-only hidden q_end predicts (ridge on PCA(256), cross-fitted over problems)
  diff      r_mean minus the mean r_mean of the same problem's correct solutions
  step      mean over the rule's first wrong equation minus the mean over that solution's equations (>= 2 equations)
  step_raw  the same equation without the subtraction
raw / qres / diff share the same rows (correct sibling required). raw_all / qres_all use every non-NOEQ wrong answer.
raw_noeq puts NOEQ back as a sanity check. qres_ns / diff_ns additionally regress out writing style (first word, LaTeX,
'Step 1', bold, log length, '=' count, line count), because style is the top axis of qres and diff.
Columns:
  sil / null    silhouette; same on a Gaussian with the PCA variances (no clusters by construction)
  stab / null   ARI with the clustering refit on 80% of problems
  same_p        same-problem pairs in one cluster, chance-corrected (0 = no topic effect); diff shares a subtraction
                within a problem, which inflates this one, so q is the fair topic check across representations
  q conf text   balanced accuracy predicting the cluster from question-only hidden / confidence+length / response TF-IDF
                (held-out problems, chance = 1/k)
  len           share of log-length variance explained by the cluster
  rule style audit  adjusted mutual information with rule type / first word of the response / AI-audited class (test)
  arith target  largest cluster rate of that rule type divided by its overall rate
  V             Cramer's V of cluster vs wrong/correct, correct solutions assigned in the same representation
usage: python cluster_errors.py <feat_dir> <auto_types.jsonl> <gen_dir> <cls_labels.tsv> <loc_feat_dir> <out_dir>
"""
import collections, glob, json, os, pickle, re, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score, roc_auc_score, \
    silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, TYPES, GEN, AUDIT, LOC, OUT = sys.argv[1:7]
K = [2, 3, 4, 5, 6, 8, 10]
os.makedirs(OUT, exist_ok=True)

shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for d in metas for m in d['meta']]
LAYERS, POS, CONF = metas[0]['layers'], metas[0]['positions'], metas[0]['conf']
keep = np.array([m['format_ok'] for m in meta])  # same row order as H_all.npy (see probe_gen.py)
C = np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep]
meta = [m for m, k in zip(meta, keep) if k]
H = np.load(f'{F}/H_all.npy', mmap_mode='r')
il = LAYERS.index(20)
Q = np.asarray(H[:, POS.index('q_end'), il], dtype=np.float32)
R = np.asarray(H[:, POS.index('r_mean'), il], dtype=np.float32)
key = [(m['split'], m['pid'], m['sample']) for m in meta]
index = {k: i for i, k in enumerate(key)}
N = len(meta)
train = np.array([m['split'] == 'train' for m in meta])
correct = np.array([m['correct'] for m in meta])
prob = np.array([f"{m['split']}-{m['pid']}" for m in meta])
typ = {(t['split'], t['pid'], t['sample']): t for t in map(json.loads, open(TYPES))}
cat = np.array([typ[k]['cat'] if k in typ else '' for k in key])
gen = {(r['split'], r['pid'], r['sample']): r for f in glob.glob(f'{GEN}/*_*.jsonl') for r in map(json.loads, open(f))}
text = np.array([gen[k]['text'] for k in key], dtype=object)
question = np.array([gen[k]['question'] for k in key], dtype=object)
audit = {}
for line in open(AUDIT, encoding='utf-8').read().splitlines()[1:]:
    x = line.split('\t')
    audit[x[0]] = x[1]
aud = np.array([audit.get(f'{s}-{p}-{n}', '') for s, p, n in key])


def first_word(t):
    m = re.match(r'\W*(\w+)', t)
    return m.group(1).lower() if m else ''


first = np.array([first_word(t) for t in text])
top_first = {w for w, _ in collections.Counter(first[train & ~correct]).most_common(8)}
first_cat = np.array([w if w in top_first else 'other' for w in first])
latex = np.array([bool(re.search(r'\\\(|\\\[|\\frac|\$\$', t)) for t in text])
step_style = np.array([bool(re.match(r'\W*step\s*1', t, re.I)) for t in text])
bold = np.array(['**' in t for t in text])
n_eq = np.array([typ[k]['n_eq'] if k in typ else -1 for k in key])
loglen = np.log(C[:, CONF.index('length')] + 1)
pickle.dump(dict(key=key, cat=cat, aud=aud, text=text, question=question, first=first_cat, latex=latex, step=step_style,
                 bold=bold, n_eq=n_eq, loglen=loglen, nll=C[:, CONF.index('nll_mean')], correct=correct,
                 k={k: typ[k]['k'] for k in typ}, line={k: typ[k]['line'] for k in typ}), open(f'{OUT}/rows.pkl', 'wb'))

# how much of r_mean is the problem at all?
tr = np.where(train)[0]
X = StandardScaler().fit_transform(R[tr])
sst = (X ** 2).sum()
ug, ginv = np.unique(prob[tr], return_inverse=True)
gm = np.zeros((len(ug), X.shape[1]), dtype=np.float64)
np.add.at(gm, ginv, X)
cnt = np.bincount(ginv)
print(f'train solutions {len(tr)}, problems {len(ug)}: between-problem share of r_mean variance '
      f'{((gm ** 2 / cnt[:, None]).sum()) / sst:.1%} (all solutions)')

# question-predicted part of r_mean, cross-fitted over problems
q_model = lambda: make_pipeline(StandardScaler(), PCA(256, random_state=0), Ridge(alpha=100.0))
qhat = np.zeros_like(R)
for a, b in GroupKFold(5).split(tr, groups=prob[tr]):
    qhat[tr[b]] = q_model().fit(Q[tr[a]], R[tr[a]]).predict(Q[tr[b]])
qhat[~train] = q_model().fit(Q[tr], R[tr]).predict(Q[~train])
print(f'r_mean variance predicted by q_end on held-out problems: '
      f'{1 - ((R[tr] - qhat[tr]) ** 2).sum() / ((R[tr] - R[tr].mean(0)) ** 2).sum():.1%}')

sib = collections.defaultdict(list)
for i in np.where(correct)[0]:
    sib[prob[i]].append(i)
wrong = ~correct & (cat != 'NOEQ') & (cat != '')
has_sib = np.array([p in sib for p in prob])
diff_w = lambda ii: np.stack([R[i] - R[sib[prob[i]]].mean(0) for i in ii])


def diff_c(ii):  # correct solution minus the OTHER correct solutions of its problem: same operation without an error
    return np.stack([R[i] - R[[j for j in sib[prob[i]] if j != i]].mean(0) for i in ii])


idx = lambda m: np.where(m)[0]
common_tr, common_te = idx(wrong & has_sib & train), idx(wrong & has_sib & ~train)
all_tr, all_te = idx(wrong & train), idx(wrong & ~train)
noeq_tr, noeq_te = idx(~correct & train & (cat != '')), idx(~correct & ~train & (cat != ''))
wrong_probs = set(prob[common_tr])
pl_common = idx(correct & train & np.isin(prob, list(wrong_probs)))
pl_diff = np.array([i for i in pl_common if len(sib[prob[i]]) >= 2])
pl_all = idx(correct & train)

REPS = {
    'raw': (common_tr, R[common_tr], common_te, R[common_te], pl_common, R[pl_common]),
    'qres': (common_tr, (R - qhat)[common_tr], common_te, (R - qhat)[common_te], pl_common, (R - qhat)[pl_common]),
    'diff': (common_tr, diff_w(common_tr), common_te, diff_w(common_te), pl_diff, diff_c(pl_diff)),
    'raw_all': (all_tr, R[all_tr], all_te, R[all_te], pl_all, R[pl_all]),
    'qres_all': (all_tr, (R - qhat)[all_tr], all_te, (R - qhat)[all_te], pl_all, (R - qhat)[pl_all]),
    'raw_noeq': (noeq_tr, R[noeq_tr], noeq_te, R[noeq_te], pl_all, R[pl_all]),
}

S = np.column_stack([first_cat == w for w in sorted(set(first_cat))] +
                    [latex, step_style, bold, loglen, [t.count('=') for t in text], [t.count('\n') for t in text]]).astype(np.float32)


def no_style(name, St, Se, Sp):
    it, Xt, ie, Xe, ip, Xp = REPS[name]
    m = LinearRegression().fit(St, Xt)
    return it, Xt - m.predict(St), ie, Xe - m.predict(Se), ip, Xp - m.predict(Sp)


ds_w = lambda ii: np.stack([S[i] - S[sib[prob[i]]].mean(0) for i in ii])
ds_c = lambda ii: np.stack([S[i] - S[[j for j in sib[prob[i]] if j != i]].mean(0) for i in ii])
REPS['qres_ns'] = no_style('qres', S[common_tr], S[common_te], S[pl_common])
REPS['diff_ns'] = no_style('diff', ds_w(common_tr), ds_w(common_te), ds_c(pl_diff))

# first wrong equation (rule k), from extract_loc.py
locs = sorted(int(f[9:-5]) for f in os.listdir(LOC) if f.startswith('meta_loc_'))
lm = [json.load(open(f'{LOC}/meta_loc_{s}.json')) for s in locs]
lpos, llay = lm[0]['positions'].index('mean'), lm[0]['layers'].index(20)
LH = np.concatenate([np.load(f'{LOC}/H_loc_{s}.npy', mmap_mode='r')[:, lpos, llay] for s in locs]).astype(np.float32)
lmeta = [m for d in lm for m in d['meta']]
sol = collections.defaultdict(list)
for j, m in enumerate(lmeta):
    sol[(m['split'], m['pid'], m['sample'])].append(j)
rows = [k for k, v in sorted(sol.items()) if len(v) >= 2 and typ[k]['k'] is not None and typ[k]['k'] < len(v)]
s_at = np.stack([LH[sol[k][typ[k]['k']]] for k in rows])
s_mean = np.stack([LH[sol[k]].mean(0) for k in rows])
s_idx = np.array([index[k] for k in rows])
s_tr = train[s_idx]
none = np.array([], dtype=int)
REPS['step'] = (s_idx[s_tr], (s_at - s_mean)[s_tr], s_idx[~s_tr], (s_at - s_mean)[~s_tr], none, None)
REPS['step_raw'] = (s_idx[s_tr], s_at[s_tr], s_idx[~s_tr], s_at[~s_tr], none, None)


def cramers_v(a, b):
    t = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(t, (a, b), 1)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    return np.sqrt(((t - e) ** 2 / e).sum() / t.sum() / (min(t.shape) - 1))


def eta2(v, lab):
    m = v.mean()
    return sum((lab == c).sum() * (v[lab == c].mean() - m) ** 2 for c in np.unique(lab)) / ((v - m) ** 2).sum()


def same_problem(lab, g):
    p = np.bincount(lab) / len(lab)
    chance = (p ** 2).sum()
    same, tot = 0, 0
    by = collections.defaultdict(list)
    for c, x in zip(lab, g):
        by[x].append(c)
    for v in by.values():
        n = len(v)
        tot += n * (n - 1) / 2
        same += sum(c * (c - 1) / 2 for c in collections.Counter(v).values())
    return (same / tot - chance) / (1 - chance)


def cv_acc(y, g, fit_predict):
    p = np.zeros(len(y), dtype=int)
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g):
        p[b] = fit_predict(a, b, y)
    return balanced_accuracy_score(y, p)


def eval_k(name, k, d):
    Z, g = d['Z'], d['g']
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    lab = km.labels_
    rng = np.random.default_rng(k)
    ug = np.unique(g)
    stab = []
    for r in range(10):
        m = np.isin(g, rng.choice(ug, int(.8 * len(ug)), replace=False))
        stab.append(adjusted_rand_score(lab, KMeans(k, n_init=5, random_state=r + 1).fit(Z[m]).predict(Z)))
    sn, stn = [], []
    for r in range(5):
        Zn = rng.normal(size=Z.shape) * Z.std(0)
        ln = KMeans(k, n_init=5, random_state=r).fit(Zn)
        sn.append(silhouette_score(Zn, ln.labels_))
        m = rng.random(len(Z)) < .8
        stn.append(adjusted_rand_score(ln.labels_, KMeans(k, n_init=5, random_state=r + 100).fit(Zn[m]).predict(Zn)))
    lr = lambda c=0.05: LogisticRegression(C=c, max_iter=1000, class_weight='balanced')
    q = cv_acc(lab, g, lambda a, b, y: make_pipeline(StandardScaler(), PCA(256, random_state=0), lr())
               .fit(d['Q'][a], y[a]).predict(d['Q'][b]))
    conf = cv_acc(lab, g, lambda a, b, y: make_pipeline(StandardScaler(), lr(1.0)).fit(d['C'][a], y[a]).predict(d['C'][b]))

    def text_fp(a, b, y):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                              token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
        return lr(1.0).fit(vec.fit_transform(d['text'][a]), y[a]).predict(vec.transform(d['text'][b]))

    tx = cv_acc(lab, g, text_fp)
    lt = km.predict(d['Zt'])
    has = d['aud'] != ''
    lift = lambda c: max(((d['cat'][lab == j] == c).mean() for j in range(k))) / (d['cat'] == c).mean()
    out = dict(rep=name, k=k, sil=silhouette_score(Z, lab), sil_null=np.mean(sn), stab=np.mean(stab), stab_null=np.mean(stn),
               same_p=same_problem(lab, g), q=q, conf=conf, text=tx, len=eta2(d['loglen'], lab),
               rule=adjusted_mutual_info_score(d['cat'], lab), style=adjusted_mutual_info_score(d['first'], lab),
               audit=adjusted_mutual_info_score(d['aud'][has], lt[has]), arith=lift('ARITH'), target=lift('TARGET'))
    if len(d['Zp']):
        lp = km.predict(d['Zp'])
        out['V'] = cramers_v(np.r_[lab, lp], np.r_[np.zeros(len(lab), int), np.ones(len(lp), int)])
        out['lab_p'] = lp
    else:
        out['V'], out['lab_p'] = float('nan'), none
    out.update(lab=lab, lab_t=lt, centers=km.cluster_centers_)
    return out


data = {}
for name, (it, Xt, ie, Xe, ip, Xp) in REPS.items():
    pre = make_pipeline(StandardScaler(), PCA(50, random_state=0)).fit(Xt)
    data[name] = dict(Z=pre.transform(Xt), Zt=pre.transform(Xe), Zp=pre.transform(Xp) if len(ip) else np.zeros((0, 50)),
                      g=prob[it], Q=Q[it], C=C[it], text=text[it], cat=cat[it], first=first_cat[it], loglen=loglen[it],
                      aud=aud[ie], idx=it, idx_t=ie, idx_p=ip, evr=pre[-1].explained_variance_ratio_)
    print(f'{name}: train {len(it)} rows / {len(np.unique(prob[it]))} problems, test {len(ie)}, correct solutions {len(ip)}, '
          f'PCA(50) keeps {data[name]["evr"].sum():.0%}, top5 ' + ' '.join(f'{x:.1%}' for x in data[name]['evr'][:5]), flush=True)

with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=64)(delayed(eval_k)(n, k, data[n]) for n in REPS for k in K)

print('\n## clustering, by representation and k')
cols = ['sil', 'sil_null', 'stab', 'stab_null', 'same_p', 'q', 'conf', 'text', 'len', 'rule', 'style', 'audit', 'arith',
        'target', 'V']
print(f"{'rep':9s} {'k':>2s} " + ' '.join(f'{c:>9s}' for c in cols))
for r in res:
    print(f"{r['rep']:9s} {r['k']:2d} " + ' '.join(f'{r[c]:9.3f}' for c in cols))

print('\n## PCA axes: variance, share predicted from the question (held-out problems), correlations, single-axis AUROC')
print(f"{'rep':9s} {'PC':>2s} {'var':>6s} {'R2_q':>6s} {'len':>6s} {'nll':>6s} {'n_eq':>6s} {'latex':>6s} {'step':>6s} {'bold':>6s} "
      f"{'A/S':>6s} {'T/S':>6s}")
corr = lambda a, b: np.corrcoef(a, b)[0, 1]
for name, d in data.items():
    it, Z5 = d['idx'], d['Z'][:, :5]
    p5 = np.zeros_like(Z5)
    for a, b in GroupKFold(5).split(it, groups=d['g']):
        p5[b] = q_model().fit(d['Q'][a], Z5[a]).predict(d['Q'][b])
    for j in range(5):
        z = Z5[:, j]
        r2 = 1 - ((z - p5[:, j]) ** 2).sum() / ((z - z.mean()) ** 2).sum()
        au = []
        for c in ('ARITH', 'TARGET'):
            m = np.isin(d['cat'], [c, 'SETUP'])
            au.append(roc_auc_score(d['cat'][m] == c, z[m]) if (d['cat'] == c).sum() >= 10 else float('nan'))
        print(f"{name:9s} {j + 1:2d} {d['evr'][j]:6.1%} {r2:6.2f} {corr(z, loglen[it]):6.2f} {corr(z, C[it, 0]):6.2f} "
              f"{corr(z, n_eq[it]):6.2f} {corr(z, latex[it]):6.2f} {corr(z, step_style[it]):6.2f} {corr(z, bold[it]):6.2f} "
              f"{max(au[0], 1 - au[0]):6.2f} {max(au[1], 1 - au[1]):6.2f}")

print('\n## reference: supervised probe on the same 50 PCA dims (AUROC, held-out problems)')
for name, d in data.items():
    line = []
    for c in ('ARITH', 'TARGET'):
        m = np.isin(d['cat'], [c, 'SETUP'])
        if (d['cat'] == c).sum() < 10:
            line.append(f'{c} vs SETUP n/a')
            continue
        y, s = (d['cat'][m] == c).astype(int), np.zeros(m.sum())
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(s, y, d['g'][m]):
            s[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000, class_weight='balanced')) \
                .fit(d['Z'][m][a], y[a]).decision_function(d['Z'][m][b])
        line.append(f'{c} vs SETUP {roc_auc_score(y, s):.3f}')
    print(f'{name:9s} ' + '  '.join(line))

for name, d in data.items():
    rs = [r for r in res if r['rep'] == name]
    np.savez(f'{OUT}/rep_{name}.npz', Z=d['Z'].astype(np.float32), Zt=d['Zt'].astype(np.float32), Zp=d['Zp'].astype(np.float32), idx=d['idx'], idx_t=d['idx_t'],
             idx_p=d['idx_p'], evr=d['evr'], **{f"lab_{r['k']}": r['lab'] for r in rs},
             **{f"lab_t_{r['k']}": r['lab_t'] for r in rs}, **{f"lab_p_{r['k']}": r['lab_p'] for r in rs},
             **{f"c_{r['k']}": r['centers'] for r in rs})
