"""Can hidden states tell Sunghun's error types apart? (his 1,100 AI-labeled wrong answers, GSM8K + SVAMP)

Labels: his `math_error_type` (labels.jsonl; 5-labeler re-check agreed 93%). Each type vs modeling_relation (the
catch-all, 732), AUROC (first type = positive), question-grouped 5-fold CV repeated over 5 seeds (out-of-fold scores
averaged), same inputs and probe as probe_types.py:
  length, confidence, confidence + length, text TF-IDF of the response,
  our replay features (extract_gen.py on his exact token ids): hidden q_end (question only: topic control) /
  r_end / r_mean / ans at layers 4..28, r_mean L20 + confidence + length,
  his released h28 features (mean, question).
within = only pairs of the two types from the SAME question (GSM8K has 8 samples per question).
Primary comparison fixed in advance: hidden r_mean L20 vs q_end L20, vs confidence + length, vs text TF-IDF.
usage: python -I probe_sh_types.py <labels.jsonl> <feat_dir> <release_dir>
"""
import collections, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
LABELS, F, REL = sys.argv[1:4]
REF = 'modeling_relation'
TASKS = ['arithmetic_execution', 'reasoning_state', 'unit_conversion', 'undetermined', 'none_observed']
REPS = 5
rows = {r['id']: r for r in map(json.loads, open(LABELS, encoding='utf-8'))}
meta = json.load(open(f'{F}/meta_0.json'))
LAYERS, POS, CONF = meta['layers'], meta['positions'], meta['conf']
ids = [m['sample'] for m in meta['meta']]
C = np.load(f'{F}/C_0.npy')                                  # H_0.npy: (N, pos, layer, 1536), rows in meta order
his = {}
for d in ('gsm8k-sampled-posthoc-v1', 'svamp-inference-v1'):
    z = np.load(f'{REL}/{d}/features.npz', allow_pickle=False)
    zm, zq = z['mean'], z['question']  # npz re-reads an array on every key access
    for k, i in enumerate(z['ids']):
        his[str(i)] = (zm[k], zq[k])
HIS = {'his h28 mean': 'his_mean', 'his h28 question': 'his_question'}
np.save(f'{F}/his_mean.npy', np.stack([his[i][0] for i in ids]))  # workers memory-map these instead of copying
np.save(f'{F}/his_question.npy', np.stack([his[i][1] for i in ids]))
lab = np.array([rows[i]['math_error_type'] for i in ids])
grp = np.array([rows[i]['question_id'] for i in ids])
TEXT = np.array([rows[i]['response'] for i in ids], dtype=object)
ci = CONF.index('length')
print('types: ' + '  '.join(f'{t}={c}' for t, c in collections.Counter(lab).most_common()))
print('mean reasoning length (tokens): ' + '  '.join(f'{t}={C[lab == t, ci].mean():.0f}' for t in [REF] + TASKS))


def scalar(cols):
    def b(sel, tr, te, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
        return m.fit(C[sel][tr][:, cols], y[tr]).decision_function(C[sel][te][:, cols])
    return b


def text(sel, tr, te, y):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                          token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(vec.fit_transform(TEXT[sel][tr]), y[tr])
    return m.decision_function(vec.transform(TEXT[sel][te]))


def hidden(get, with_conf=False):
    def b(sel, tr, te, y):
        X = get()[sel].astype(np.float32)
        Z = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr]).transform(X)
        if with_conf:
            Z = np.hstack([Z, StandardScaler().fit(C[sel][tr]).transform(C[sel])])
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        return m.decision_function(Z[te])
    return b


ours = lambda p, l: (lambda: np.load(f'{F}/H_0.npy', mmap_mode='r')[:, POS.index(p), LAYERS.index(l)])
conf_cols = [i for i in range(len(CONF)) if i != ci]
configs = [('length', scalar([ci])), ('confidence', scalar(conf_cols)),
           ('confidence + length', scalar(list(range(len(CONF))))), ('text TF-IDF', text),
           ('hidden q_end L20 (question only)', hidden(ours('q_end', 20)))]
configs += [(f'hidden r_mean L{l}', hidden(ours('r_mean', l))) for l in LAYERS]
configs += [('hidden r_end L20', hidden(ours('r_end', 20))), ('hidden ans L20', hidden(ours('ans', 20))),
            ('hidden r_mean L20 + conf + length', hidden(ours('r_mean', 20), True))]
configs += [(n, hidden(lambda f=f: np.load(f'{F}/{f}.npy', mmap_mode='r'))) for n, f in HIS.items()]
NAMES = [n for n, _ in configs]


def oof(sel, y, g, b):
    s = np.zeros((REPS, len(y)))
    for rep in range(REPS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, g):
            s[rep, te] = b(sel, tr, te, y)
    return s.mean(0)


def within(s, y, g):
    num = den = 0
    for k in np.unique(g[y == 1]):
        m = g == k
        d = s[m & (y == 1)][:, None] - s[m & (y == 0)][None, :]
        num += (d > 0).sum() + 0.5 * (d == 0).sum(); den += d.size
    return (num / den if den else float('nan')), int(den)


jobs = []
for t in TASKS:
    sel = np.where((lab == t) | (lab == REF))[0]
    jobs += [(t, n, sel, (lab[sel] == t).astype(int), grp[sel], b) for n, b in configs]
with parallel_backend('loky', inner_max_num_threads=1):  # one BLAS thread per worker, else the cores oversubscribe
    S = Parallel(n_jobs=-1)(delayed(oof)(sel, y, g, b) for _, _, sel, y, g, b in jobs)
R = {(t, n): (s, y, g) for (t, n, _, y, g, _), s in zip(jobs, S)}

rng = np.random.default_rng(0)
for t in TASKS:
    _, y, g = R[(t, 'length')]
    print(f'\n## {t} vs {REF}  (n={y.sum()} vs {(1 - y).sum()}, questions={len(set(g))}), AUROC  [within = same-question pairs]')
    print(f"  {'input':36s} {'AUROC':>6s} {'within':>7s} {'pairs':>6s}")
    for n in NAMES:
        s, y, g = R[(t, n)]
        w, npair = within(s, y, g)
        print(f'  {n:36s} {roc_auc_score(y, s):6.3f} {w:7.3f} {npair:6d}')
    keys = np.unique(g)
    gi = {k: np.where(g == k)[0] for k in keys}
    for A, B in [('hidden r_mean L20', 'hidden q_end L20 (question only)'), ('hidden r_mean L20', 'confidence + length'),
                 ('hidden r_mean L20', 'text TF-IDF')]:
        sa, sb = R[(t, A)][0], R[(t, B)][0]
        d = []
        for _ in range(1000):
            ix = np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])
            if 0 < y[ix].sum() < len(ix):
                d.append(roc_auc_score(y[ix], sa[ix]) - roc_auc_score(y[ix], sb[ix]))
        print(f'  bootstrap {A} - {B}: {np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')
