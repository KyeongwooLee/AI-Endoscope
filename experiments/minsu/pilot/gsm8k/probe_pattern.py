"""Can the wrong answer's hidden state tell its answer-distribution pattern (analyze_patterns.py) before any resampling?

Pattern from all 32 restarts: dominant (correct >= half), stuck (one wrong answer >= half), competing (correct and one
wrong answer each >= 20%), spread (> 10 distinct answers), other (dropped here).
Tasks:
  4-way      dominant / stuck / competing / spread, balanced accuracy and macro-F1
  stuck vs spread   both rarely fixed by restarting, so difficulty alone should not separate them (AUROC)
  stuck vs rest, dominant vs rest (AUROC)
Inputs: hidden r_mean L20, question-only q_end L20, confidence + length, response TF-IDF; reference not usable in
practice: other-4-solutions accuracy. 5-fold CV x 5 seeds (one wrong answer per problem); paired bootstrap for hidden
minus each input.
usage: python probe_pattern.py <feat_dir> <gen_dir> <fix_dir> [<fix_dir> ...]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, GEN, FIXES = sys.argv[1], sys.argv[2], sys.argv[3:]
REPS = 5
pred, gold = collections.defaultdict(list), {}
for FIX in FIXES:
    for f in sorted(glob.glob(f'{FIX}/fix_*.jsonl')) + sorted(glob.glob(f'{FIX}/more/fix_*.jsonl')):
        for r in map(json.loads, open(f)):
            if r['cond'] == 'restart':
                pred[r['id']] += r['preds']
                gold[r['id']] = r['gold']
key = lambda x: None if x is None else round(x, 4)


def pattern(i):
    p = [key(x) for x in pred[i]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / len(p) if wrong else 0)
    return 'dominant' if c >= .5 else 'stuck' if w >= .5 else 'competing' if c >= .2 and w >= .2 else \
        'spread' if len(set(p)) > 10 else 'other'


P4 = ['dominant', 'stuck', 'competing', 'spread']
ids = sorted(i for i in pred if len(pred[i]) == 32 and pattern(i) in P4)
lab = np.array([P4.index(pattern(i)) for i in ids])
print('items ' + str(len(ids)) + ': ' + '  '.join(f'{n} {(lab == k).sum()}' for k, n in enumerate(P4)))

shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
keep = np.array([m['format_ok'] for d in metas for m in d['meta']])
meta = [m for d in metas for m in d['meta'] if m['format_ok']]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}
J = np.array([row[i] for i in ids])
H = np.load(f'{F}/H_all.npy', mmap_mode='r')
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
X = {'hidden r_mean': np.asarray(H[J, POS.index('r_mean'), LAYERS.index(20)], dtype=np.float32),
     'question q_end': np.asarray(H[J, POS.index('q_end'), LAYERS.index(20)], dtype=np.float32),
     'confidence+len': np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep][J]}
gen = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f))}
TEXT = np.array([gen[i]['text'] for i in ids], dtype=object)
sib = collections.defaultdict(list)
for k, r in gen.items():
    sib[r['pid']].append((k, r['correct']))
other_acc = np.array([np.mean([c for k, c in sib[gen[i]['pid']] if k != i]) for i in ids])

TASKS = {'4-way': (np.ones(len(ids), bool), lab),
         'stuck vs spread': (np.isin(lab, [1, 3]), (lab == 1).astype(int)),
         'stuck vs rest': (np.ones(len(ids), bool), (lab == 1).astype(int)),
         'dominant vs rest': (np.ones(len(ids), bool), (lab == 0).astype(int))}


def run(name, task, seed):
    m, y = TASKS[task]
    idx = np.where(m)[0]
    y = y[idx]
    out = np.zeros((len(idx), 4)) if task == '4-way' else np.zeros(len(idx))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        if name == 'text':
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
            A, B = vec.fit_transform(TEXT[idx[tr]]), vec.transform(TEXT[idx[te]])
            clf = LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced').fit(A, y[tr])
        elif name == 'other-4 accuracy (reference)':
            A, B = other_acc[idx[tr], None], other_acc[idx[te], None]
            clf = LogisticRegression(max_iter=3000, class_weight='balanced').fit(A, y[tr])
        else:
            A, B = X[name][idx[tr]], X[name][idx[te]]
            big = A.shape[1] > 256
            clf = make_pipeline(StandardScaler(), *([PCA(min(256, len(tr) - 1), random_state=0)] if big else []),
                                LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced')).fit(A, y[tr])
        out[te] = clf.predict_proba(B) if task == '4-way' else clf.decision_function(B)
    return name, task, out


NAMES = list(X) + ['text', 'other-4 accuracy (reference)']
with parallel_backend('loky', inner_max_num_threads=1):
    res = Parallel(n_jobs=-1)(delayed(run)(n, t, s) for n in NAMES for t in TASKS for s in range(REPS))
S = collections.defaultdict(list)
for n, t, o in res:
    S[(n, t)].append(o)
S = {k: np.mean(v, 0) for k, v in S.items()}
rng = np.random.default_rng(0)


def metric(t, s, b):
    m, y = TASKS[t]
    y = y[m][b]
    if t == '4-way':
        return balanced_accuracy_score(y, s[b].argmax(1))
    return roc_auc_score(y, s[b])


print('\n## 4-way: balanced accuracy (chance 0.25) / macro-F1; binary tasks: AUROC; [95% CI of hidden minus input]')
for n in NAMES:
    cells = []
    for t in TASKS:
        nn = TASKS[t][0].sum()
        v = metric(t, S[(n, t)], np.arange(nn))
        if n == 'hidden r_mean':
            cells.append(f'{v:.3f}')
        else:
            d = []
            for _ in range(500):
                b = rng.integers(0, nn, nn)
                try:
                    d.append(metric(t, S[('hidden r_mean', t)], b) - metric(t, S[(n, t)], b))
                except ValueError:
                    pass
            cells.append(f'{v:.3f} [{np.percentile(d, 2.5):+.2f},{np.percentile(d, 97.5):+.2f}]')
    f1 = f1_score(lab, S[(n, '4-way')].argmax(1), average='macro')
    print(f'  {n:30s} ' + '  '.join(f'{t} {c}' for t, c in zip(TASKS, cells)) + f'  (4-way macro-F1 {f1:.3f})')
print('\nper-class recall, hidden r_mean 4-way: ' + '  '.join(
    f'{n} {np.mean(S[("hidden r_mean", "4-way")].argmax(1)[lab == k] == k):.0%}' for k, n in enumerate(P4)))
print('per-class recall, confidence+len 4-way: ' + '  '.join(
    f'{n} {np.mean(S[("confidence+len", "4-way")].argmax(1)[lab == k] == k):.0%}' for k, n in enumerate(P4)))
