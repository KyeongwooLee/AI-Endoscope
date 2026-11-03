"""Multi-class version of probe_sh_types.py: predict one of Sunghun's error types (6 present; answer_reporting has 0).

Same data, features and probe as probe_sh_types.py, but one multinomial logistic regression over all types,
question-grouped 5-fold CV repeated over 5 seeds (out-of-fold probabilities averaged, then argmax).
Metrics: balanced accuracy (mean per-class recall; chance = 1/6) and macro-F1 as primary, accuracy for reference
(always predicting modeling_relation already gives 66.5%). Per-class recall and the hidden confusion matrix.
Primary comparison fixed in advance: hidden r_mean L20 vs confidence + length, vs q_end L20 (question only), vs text.
usage: python -I probe_sh_multi.py <labels.jsonl> <feat_dir>   (feat_dir from probe_sh_types.py, incl. his_*.npy)
"""
import collections, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
LABELS, F = sys.argv[1:3]
REPS = 5
rows = {r['id']: r for r in map(json.loads, open(LABELS, encoding='utf-8'))}
meta = json.load(open(f'{F}/meta_0.json'))
LAYERS, POS, CONF = meta['layers'], meta['positions'], meta['conf']
ids = [m['sample'] for m in meta['meta']]
C = np.load(f'{F}/C_0.npy')
lab_s = np.array([rows[i]['math_error_type'] for i in ids])
TYPES = [t for t, _ in collections.Counter(lab_s).most_common()]
y = np.array([TYPES.index(t) for t in lab_s])
grp = np.array([rows[i]['question_id'] for i in ids])
TEXT = np.array([rows[i]['response'] for i in ids], dtype=object)
ci = CONF.index('length')
print('types: ' + '  '.join(f'{t}={(y == k).sum()}' for k, t in enumerate(TYPES)))


def scalar(cols):
    def b(tr, te):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight='balanced'))
        return m.fit(C[tr][:, cols], y[tr]).predict_proba(C[te][:, cols])
    return b


def text(tr, te):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                          token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
    m = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(vec.fit_transform(TEXT[tr]), y[tr])
    return m.predict_proba(vec.transform(TEXT[te]))


def hidden(get, with_conf=False):
    def b(tr, te):
        X = get().astype(np.float32)
        Z = make_pipeline(StandardScaler(), PCA(256, random_state=0)).fit(X[tr]).transform(X)
        if with_conf:
            Z = np.hstack([Z, StandardScaler().fit(C[tr]).transform(C)])
        m = LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced').fit(Z[tr], y[tr])
        return m.predict_proba(Z[te])
    return b


ours = lambda p, l: (lambda: np.load(f'{F}/H_0.npy', mmap_mode='r')[:, POS.index(p), LAYERS.index(l)])
conf_cols = [i for i in range(len(CONF)) if i != ci]
configs = [('length', scalar([ci])), ('confidence', scalar(conf_cols)),
           ('confidence + length', scalar(list(range(len(CONF))))), ('text TF-IDF', text),
           ('hidden q_end L20 (question only)', hidden(ours('q_end', 20)))]
configs += [(f'hidden r_mean L{l}', hidden(ours('r_mean', l))) for l in LAYERS]
configs += [('hidden r_mean L20 + conf + length', hidden(ours('r_mean', 20), True)),
            ('his h28 mean', hidden(lambda: np.load(f'{F}/his_mean.npy', mmap_mode='r')))]
NAMES = [n for n, _ in configs]


def oof(b):
    P = np.zeros((REPS, len(y), len(TYPES)))
    for rep in range(REPS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(np.zeros(len(y)), y, grp):
            P[rep, te] = b(tr, te)  # every class appears in every training fold (smallest has 28 items)
    return P.mean(0).argmax(1)


with parallel_backend('loky', inner_max_num_threads=1):
    PRED = dict(zip(NAMES, Parallel(n_jobs=-1)(delayed(oof)(b) for _, b in configs)))

print(f'\n## one of {len(TYPES)} types, {len(y)} items, {len(set(grp))} questions (question-grouped CV)')
print(f"  {'input':36s} {'bal.acc':>7s} {'macroF1':>7s} {'acc':>6s}   recall per type: " +
      ' '.join(t[:6] for t in TYPES))
base = np.zeros(len(y), int)
for n, p in [('always modeling_relation', base)] + list(PRED.items()):
    rec = [(p[y == k] == k).mean() for k in range(len(TYPES))]
    print(f'  {n:36s} {balanced_accuracy_score(y, p):7.3f} {f1_score(y, p, average="macro"):7.3f} '
          f'{(p == y).mean():6.3f}   ' + ' '.join(f'{r:6.2f}' for r in rec))

rng = np.random.default_rng(0)
keys = np.unique(grp)
gi = {k: np.where(grp == k)[0] for k in keys}
A = 'hidden r_mean L20'
mf1 = lambda t, p: f1_score(t, p, average='macro', labels=range(len(TYPES)))
for B in ['confidence', 'confidence + length', 'hidden q_end L20 (question only)', 'text TF-IDF']:
    d = []
    for _ in range(1000):
        ix = np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])
        if len(set(y[ix])) == len(TYPES):
            d.append([f(y[ix], PRED[A][ix]) - f(y[ix], PRED[B][ix]) for f in (balanced_accuracy_score, mf1)])
    d = np.array(d)
    print(f'  bootstrap {A} - {B}: ' + '  '.join(
        f'{m} {d[:, j].mean():+.3f} [{np.percentile(d[:, j], 2.5):+.3f}, {np.percentile(d[:, j], 97.5):+.3f}]'
        for j, m in enumerate(['bal.acc', 'macroF1'])))

print(f'\n## confusion matrix, {A} (rows = Sunghun label, columns = predicted)')
print('  ' + ' ' * 21 + ' '.join(f'{t[:6]:>6s}' for t in TYPES))
for k, row in enumerate(confusion_matrix(y, PRED[A], labels=range(len(TYPES)))):
    print(f'  {TYPES[k]:21s}' + ' '.join(f'{v:6d}' for v in row))
