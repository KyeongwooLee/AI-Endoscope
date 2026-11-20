"""RQ1 controls: is the pattern signal in the writer's own hidden state, or in the text any reader can see?

Patterns and tasks as in probe_pattern.py (32 restarts; stuck vs spread / stuck vs rest / dominant vs rest AUROC,
4-way balanced accuracy). Inputs:
  self            writer Qwen2.5-1.5B hidden r_mean L20 (feat/H_all.npy, generation-time token ids)
  confidence+len  writer's 9 confidence features
  reader X        another model reading the same question + solution (extract_reader.py), r_mean; every layer is
                  scored and the best layer is reported (picked on the same folds, which favors the reader)
  combinations    self+confidence, reader+confidence, self+reader (does self add anything beyond the reader?)
5-fold CV x 5 seeds, same folds for every input; paired bootstrap (1000) over items.
POS=q_end (env) uses the last prompt token instead, i.e. every model has read only the question: any advantage of
self there would be knowledge of its own tendencies, since the text is identical for all.
usage: python probe_reader.py <feat_dir> <reader_root> <fix_dir> [<fix_dir> ...]
"""
import collections, glob, json, os, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, RROOT, FIXES = sys.argv[1], sys.argv[2], sys.argv[3:]
REPS = 5
POSN = os.environ.get('POS', 'r_mean')
pred, gold = collections.defaultdict(list), {}
for D in FIXES:
    for f in sorted(glob.glob(f'{D}/fix_*.jsonl')) + sorted(glob.glob(f'{D}/more/fix_*.jsonl')):
        for r in map(json.loads, open(f)):
            if r['cond'] == 'restart':
                pred[r['id']] += r['preds']
                gold[r['id']] = r['gold']
key = lambda x: None if x is None else round(x, 4)


def pattern(i):  # as in probe_pattern.py
    p = [key(x) for x in pred[i]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / len(p) if wrong else 0)
    return 'dominant' if c >= .5 else 'stuck' if w >= .5 else 'competing' if c >= .2 and w >= .2 else \
        'spread' if len(set(p)) > 10 else 'other'


P4 = ['dominant', 'stuck', 'competing', 'spread']
ids = sorted(i for i in pred if len(pred[i]) == 32 and pattern(i) in P4)
lab = np.array([P4.index(pattern(i)) for i in ids])
print('items ' + str(len(ids)) + ': ' + '  '.join(f'{n} {(lab == k).sum()}' for k, n in enumerate(P4)), flush=True)

shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
keep = np.array([m['format_ok'] for d in metas for m in d['meta']])
meta = [m for d in metas for m in d['meta'] if m['format_ok']]
row = {f"{m['split']}-{m['pid']}-{m['sample']}": j for j, m in enumerate(meta)}
J = np.array([row[i] for i in ids])
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
X = {'self': np.asarray(np.load(f'{F}/H_all.npy', mmap_mode='r')[J, POS.index(POSN), LAYERS.index(20)], dtype=np.float32),
     'conf': np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep][J].astype(np.float32)}
READERS = []
for d in sorted(glob.glob(f'{RROOT}/*/meta.json')):
    m = json.load(open(d))
    name = os.path.basename(os.path.dirname(d))
    pos = {i: k for k, i in enumerate(m['ids'])}
    H = np.load(os.path.join(os.path.dirname(d), 'H.npy'), mmap_mode='r')
    for li, l in enumerate(m['layers']):
        X[f'{name} L{l}'] = np.asarray(H[[pos[i] for i in ids], m['positions'].index(POSN), li], dtype=np.float32)
    READERS.append((name, [f'{name} L{l}' for l in m['layers']]))
print('readers: ' + ', '.join(f'{n} ({len(c)} layers)' for n, c in READERS), flush=True)

TASKS = {'stuck vs spread': (np.isin(lab, [1, 3]), (lab == 1).astype(int)),
         '4-way': (np.ones(len(ids), bool), lab),
         'stuck vs rest': (np.ones(len(ids), bool), (lab == 1).astype(int)),
         'dominant vs rest': (np.ones(len(ids), bool), (lab == 0).astype(int))}


def model(blocks, n):
    """blocks: list of (name, width). Hidden blocks get scaler + PCA, confidence only a scaler."""
    parts, start = [], 0
    for k, (b, w) in enumerate(blocks):
        cols = list(range(start, start + w))
        start += w
        tf = make_pipeline(StandardScaler(), PCA(min(256, n - 1), random_state=0)) if b != 'conf' else StandardScaler()
        parts.append((f'block{k}', tf, cols))  # a transformer named 'self' clashes with sklearn internals
    only_conf = len(blocks) == 1 and blocks[0][0] == 'conf'
    return make_pipeline(ColumnTransformer(parts),
                         LogisticRegression(C=1.0 if only_conf else 0.05, max_iter=3000, class_weight='balanced'))


def run(names, task, seed):
    m, y = TASKS[task]
    idx = np.where(m)[0]
    y = y[idx]
    A = np.concatenate([X[n][idx] for n in names], 1)
    blocks = [(n, X[n].shape[1]) for n in names]
    out = np.zeros((len(idx), 4)) if task == '4-way' else np.zeros(len(idx))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y):
        clf = model(blocks, len(tr)).fit(A[tr], y[tr])
        out[te] = clf.predict_proba(A[te]) if task == '4-way' else clf.decision_function(A[te])
    return ' + '.join(names), task, out


def metric(t, s, b):
    y = TASKS[t][1][TASKS[t][0]][b]
    return balanced_accuracy_score(y, s[b].argmax(1)) if t == '4-way' else roc_auc_score(y, s[b])


def collect(jobs):
    with parallel_backend('loky', inner_max_num_threads=1):
        res = Parallel(n_jobs=-1)(delayed(run)(n, t, s) for n in jobs for t in TASKS for s in range(REPS))
    S = collections.defaultdict(list)
    for n, t, o in res:
        S[(n, t)].append(o)
    return {k: np.mean(v, 0) for k, v in S.items()}


singles = [['self'], ['conf']] + [[c] for _, cs in READERS for c in cs]
S = collect(singles)
full = lambda t: np.arange(TASKS[t][0].sum())
best = {}
print(f'\n## each reader, every layer ({POSN}); AUROC, 4-way = balanced accuracy')
for name, cs in READERS:
    print(f'  {name}')
    for c in cs:
        print(f'    {c:22s} ' + '  '.join(f'{t} {metric(t, S[(c, t)], full(t)):.3f}' for t in TASKS))
    best[name] = max(cs, key=lambda c: metric('stuck vs spread', S[(c, 'stuck vs spread')], full('stuck vs spread')))
print('  reader layer used below = best layer for stuck vs spread: ' + ', '.join(best.values()))

combos = [['self', 'conf']] + [[best[n], 'conf'] for n, _ in READERS] + [['self', best[n]] for n, _ in READERS] + \
         [['self', best[n], 'conf'] for n, _ in READERS]
S.update(collect(combos))
rng = np.random.default_rng(0)


def diff(a, b, t):
    nn = TASKS[t][0].sum()
    d = []
    for _ in range(1000):
        bb = rng.integers(0, nn, nn)
        try:
            d.append(metric(t, S[(a, t)], bb) - metric(t, S[(b, t)], bb))
        except ValueError:
            pass
    return f'{np.mean(d):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]'


rows = ['self', 'conf'] + [best[n] for n, _ in READERS] + [' + '.join(c) for c in combos]
print('\n## all inputs; AUROC, 4-way = balanced accuracy (chance 0.25)')
for r in rows:
    print(f'  {r:34s} ' + '  '.join(f'{t} {metric(t, S[(r, t)], full(t)):.3f}' for t in TASKS))
pairs = [('self', best[n]) for n, _ in READERS] + [(f'self + {best[n]}', best[n]) for n, _ in READERS] + \
        [('self + conf', 'conf')] + [(f'self + {best[n]} + conf', f'{best[n]} + conf') for n, _ in READERS]
print('\n## paired bootstrap, A minus B: mean [95% CI]')
for a, b in pairs:
    print(f'  {a:34s} - {b:26s} ' + '  '.join(f'{t} {diff(a, b, t)}' for t in TASKS))
