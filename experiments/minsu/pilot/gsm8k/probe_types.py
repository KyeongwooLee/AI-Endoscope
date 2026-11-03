"""Can hidden states tell error TYPES apart among the model's own wrong GSM8K answers?

Labels come from rule-based typing (auto_types.py, no human/LLM judgment):
  ARITH  first error is a written equation computed wrong
  TARGET final answer is a valid intermediate (stopped early / answered another quantity)
  SETUP  first error is a correctly computed equation of a wrong quantity
Binary tasks, AUROC (first type = positive), problem-grouped 5-fold CV. Inputs: length, confidence,
question-only hidden q_end (topic/difficulty control), text-only TF-IDF of the response (can the TEXT alone tell the
types apart? if so, hidden may only re-encode the text), hidden at r_end / r_mean / ans, hidden + confidence.
within = only pairs of the two types from the SAME problem (topic removed).
Primary comparison fixed in advance: hidden r_mean L20 (best in the correct/wrong experiment) vs q_end L20 and
vs confidence + length, paired bootstrap over problems.
usage: python probe_types.py <feat_dir> <types.jsonl> <gen_dir> [cache_dir, default feat_dir] [tasks, e.g. A:B]
"""
import glob, json, sys, collections, warnings, os
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
F, TYPES, GEN = sys.argv[1], sys.argv[2], sys.argv[3]
CACHE = sys.argv[4] if len(sys.argv) > 4 else F
TASKS = [tuple(t.split(':')) for t in sys.argv[5].split(',')] if len(sys.argv) > 5 else \
    [('ARITH', 'SETUP'), ('TARGET', 'SETUP'), ('ARITH', 'TARGET')]
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for d in metas for m in d['meta']]
LAYERS, POS, CONF = metas[0]['layers'], metas[0]['positions'], metas[0]['conf']
keep = np.array([m['format_ok'] for m in meta])  # same row order as H_all.npy (see probe_gen.py)
C_all = np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep]
meta = [m for m, k in zip(meta, keep) if k]
cat = {(t['split'], t['pid'], t['sample']): t['cat'] for t in map(json.loads, open(TYPES))}
TYPES3 = tuple(dict.fromkeys(c for t in TASKS for c in t))
idx = np.array([i for i, m in enumerate(meta) if cat.get((m['split'], m['pid'], m['sample'])) in TYPES3])
assert all(not meta[i]['correct'] for i in idx)
lab = np.array([cat[(meta[i]['split'], meta[i]['pid'], meta[i]['sample'])] for i in idx])
C = C_all[idx]
prob = np.array([f"{meta[i]['split']}-{meta[i]['pid']}" for i in idx])
text = {(r['split'], r['pid'], r['sample']): r['text'] for f in glob.glob(f'{GEN}/*_*.jsonl') for r in map(json.loads, open(f))}
TEXT = np.array([text[(meta[i]['split'], meta[i]['pid'], meta[i]['sample'])] for i in idx], dtype=object)
SUB = f'{CACHE}/H_types.npy'  # typed subset, so workers don't stride through the full 3.7 GB file
if not (os.path.exists(SUB) and np.array_equal(np.load(f'{CACHE}/H_types_idx.npy'), idx)):
    np.save(SUB, np.load(f'{F}/H_all.npy', mmap_mode='r')[idx]); np.save(f'{CACHE}/H_types_idx.npy', idx)

ci = CONF.index('length')
print(f'wrong answers typed: {len(idx)}  ' + '  '.join(f'{t}={(lab == t).sum()}' for t in TYPES3))
print('mean reasoning length (tokens): ' + '  '.join(f'{t}={C[lab == t, ci].mean():.0f}' for t in TYPES3))


def task_scores(sel, y, g, builder):
    s = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g):
        s[te] = builder(sel, tr, te, y)
    return s


def scalar_model(cols):
    def b(sel, tr, te, y):
        X = C[sel][:, cols]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced')).fit(X[tr], y[tr])
        return m.decision_function(X[te])
    return b


def text_model(sel, tr, te, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                          token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
    Xtr = vec.fit_transform(TEXT[sel][tr])
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(Xtr, y[tr])
    return m.decision_function(vec.transform(TEXT[sel][te]))


def hidden_model(p, l, with_conf=False):
    def b(sel, tr, te, y):
        X = np.load(SUB, mmap_mode='r')[sel][:, POS.index(p), LAYERS.index(l)].astype(np.float32)
        Z = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr]).transform(X)
        if with_conf:
            Z = np.hstack([Z, StandardScaler().fit(C[sel][tr]).transform(C[sel])])
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        return m.decision_function(Z[te])
    return b


def within(s, y, g):
    """P(score(pos) > score(neg)) over pos/neg pairs from the same problem; (auc, #pairs)"""
    num = den = 0
    for k in np.unique(g[y == 1]):
        m = g == k
        d = s[m & (y == 1)][:, None] - s[m & (y == 0)][None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return (num / den if den else float('nan')), int(den)


conf_cols = [i for i in range(len(CONF)) if i != ci]
configs = [('length', scalar_model([ci])), ('confidence', scalar_model(conf_cols)),
           ('confidence + length', scalar_model(list(range(len(CONF))))), ('text TF-IDF', text_model)]
configs += [(f'hidden {p} L{l}', hidden_model(p, l)) for p in POS for l in LAYERS]
configs += [('hidden r_mean L20 + conf + length', hidden_model('r_mean', 20, True))]
jobs = []
for a, b in TASKS:
    sel = np.where((lab == a) | (lab == b))[0]
    y, g = (lab[sel] == a).astype(int), prob[sel]
    jobs += [((a, b), n, sel, y, g, bl) for n, bl in configs]
with parallel_backend('loky', inner_max_num_threads=4):
    S = Parallel(n_jobs=32)(delayed(task_scores)(sel, y, g, bl) for _, _, sel, y, g, bl in jobs)
R = {(t, n): (s, y, g) for (t, n, _, y, g, _), s in zip(jobs, S)}

rng = np.random.default_rng(0)
for t in TASKS:
    _, y, g = R[(t, 'length')]
    print(f'\n## {t[0]} vs {t[1]}  (n={y.sum()} vs {(1 - y).sum()}, problems={len(set(g))}), AUROC  [within = same-problem pairs]')
    print(f"  {'input':36s} {'AUROC':>6s} {'within':>7s} {'pairs':>6s}")
    rows = ['length', 'confidence', 'confidence + length', 'text TF-IDF'] + \
           [f'hidden {p} L{l}' for p in POS for l in LAYERS] + ['hidden r_mean L20 + conf + length']
    for n in rows:
        s, y, g = R[(t, n)]
        w, npair = within(s, y, g)
        print(f'  {n:36s} {roc_auc_score(y, s):6.3f} {w:7.3f} {npair:6d}')
    keys = np.unique(g)
    gi = {k: np.where(g == k)[0] for k in keys}
    for A, B in [('hidden r_mean L20', 'hidden q_end L20'), ('hidden r_mean L20', 'confidence + length'),
                 ('hidden r_mean L20', 'text TF-IDF')]:
        sa, sb = R[(t, A)][0], R[(t, B)][0]
        d = []
        for _ in range(1000):
            ix = np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])
            if 0 < y[ix].sum() < len(ix):
                d.append(roc_auc_score(y[ix], sa[ix]) - roc_auc_score(y[ix], sb[ix]))
        print(f'  bootstrap {A} - {B}: {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
