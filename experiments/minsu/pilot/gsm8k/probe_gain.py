"""Strategy gain: which wrong answers does a strategy (code / calc / cut / line) fix that restarting does not?
(fix_exp.py, 8 + 24 attempts per strategy)

gain = fix rate of the strategy - fix rate of restart on the same wrong answer (32 attempts each).
  reliability   correlation of the gain from attempts 1-16 vs 17-32
  needed        one-sided Fisher exact test strategy > restart, p < 0.05; expected false positives from the per-item null
  traits        what the needed answers have in common (rule type, audited class, difficulty, numbers in the question)
  probe         gain (Spearman) and needed (AUROC; only with >= 20 needed answers) from hidden r_mean L20, question-only q_end L20, confidence +
                length, response TF-IDF, numbers in the question; reference: other-4-solutions accuracy (not usable)
Examples of needed answers -> <fix_dir>/<strategy>_needed_examples.txt
usage: python probe_gain.py <feat_dir> <fix_dir> <gen_dir> <cls_labels.tsv> [strategy, default code]
"""
import collections, glob, json, os, re, sys, warnings
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from scipy.stats import fisher_exact, spearmanr
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
F, FIX, GEN, AUDIT = sys.argv[1:5]
ST = sys.argv[5] if len(sys.argv) > 5 else 'code'
REPS = 5
att = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f'{FIX}/*.jsonl')) + sorted(glob.glob(f'{FIX}/more*/*.jsonl')):  # first 8, then 24 more
    for r in map(json.loads, open(f)):
        if r['cond'] in ('restart', ST):
            att[r['cond']][r['id']] += r['correct']
ids = sorted(i for i in att['restart'] if i in att[ST])
assert all(len(att[c][i]) == 32 for c in ('restart', ST) for i in ids)
R = np.array([att['restart'][i] for i in ids], float)
Cd = np.array([att[ST][i] for i in ids], float)
gain = Cd.mean(1) - R.mean(1)
g1, g2 = Cd[:, :16].mean(1) - R[:, :16].mean(1), Cd[:, 16:].mean(1) - R[:, 16:].mean(1)
print(f'strategy {ST}: items {len(ids)}; fix rate restart {R.mean():.1%} {ST} {Cd.mean():.1%}; mean gain {gain.mean():+.3f}')
print(f'reliability: gain from attempts 1-16 vs 17-32, correlation {np.corrcoef(g1, g2)[0, 1]:.2f}')
p = np.array([fisher_exact([[c.sum(), 32 - c.sum()], [r.sum(), 32 - r.sum()]], alternative='greater')[1] for c, r in zip(Cd, R)])
need = p < 0.05
pr = np.array([fisher_exact([[c.sum(), 32 - c.sum()], [r.sum(), 32 - r.sum()]], alternative='less')[1] for c, r in zip(Cd, R)])
rng = np.random.default_rng(0)
pool = (Cd.sum(1) + R.sum(1)) / 64
null = [np.mean([fisher_exact([[a, 32 - a], [b, 32 - b]], alternative='greater')[1] < .05
                 for a, b in zip(rng.binomial(32, pool), rng.binomial(32, pool))]) for _ in range(20)]
print(f'{ST}-needed ({ST} > restart, Fisher p < 0.05): {need.sum()} ({need.mean():.1%}); expected by chance if equal '
      f'{np.mean(null):.1%}; restart-needed (restart > {ST}): {(pr < .05).sum()} ({(pr < .05).mean():.1%})')

gen = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f))}
sib = collections.defaultdict(list)
for k, r in gen.items():
    sib[r['pid']].append((k, r['correct']))
other_acc = np.array([np.mean([c for k, c in sib[gen[i]['pid']] if k != i]) for i in ids])
audit = {}
for line in open(AUDIT, encoding='utf-8').read().splitlines()[1:]:
    x = line.split('\t')
    audit[x[0]] = x[1]
aud = np.array([audit[i] for i in ids])
TYPES = os.path.join(os.path.dirname(os.path.abspath(F)), 'types', 'auto_types.jsonl')  # next to feat/
types = {f"{t['split']}-{t['pid']}-{t['sample']}": t['cat'] for t in map(json.loads, open(TYPES))} if os.path.exists(TYPES) else {}
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def qnum(q):
    nums = [x.replace(',', '') for x in NUM.findall(q)]
    vals = [float(x) for x in nums]
    return [len(nums), max([len(re.sub(r'\D', '', x)) for x in nums], default=0), np.log10(max(vals, default=0) + 1),
            any('.' in x for x in nums), '%' in q or 'percent' in q.lower(), bool(re.search(r'\d+/\d+|half|third|quarter', q)),
            np.log(len(q))]


QN = np.array([qnum(gen[i]['question']) for i in ids], float)
QN_NAMES = ['numbers', 'max digits', 'log max value', 'decimal', 'percent', 'fraction', 'log question length']
print(f'\n## traits: {ST}-needed vs other wrong answers')
for k, name in enumerate(QN_NAMES):
    print(f'  {name:20s} {QN[need, k].mean():7.2f} vs {QN[~need, k].mean():7.2f}')
print(f"  {'other-4 accuracy':20s} {other_acc[need].mean():7.2f} vs {other_acc[~need].mean():7.2f}")
print(f"  {'restart fix rate':20s} {R[need].mean():7.2f} vs {R[~need].mean():7.2f}")
for c in ('CALC', 'TARGET', 'SETUP', 'OTHER'):
    print(f'  audited {c:12s} {(aud[need] == c).mean():7.1%} vs {(aud[~need] == c).mean():7.1%}')
if types:
    cat = np.array([types[i] for i in ids])
    for c in ('SETUP', 'NOEQ', 'TARGET', 'OTHER', 'ARITH'):
        print(f'  rule {c:15s} {(cat[need] == c).mean():7.1%} vs {(cat[~need] == c).mean():7.1%}')

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
     'confidence+len': np.concatenate([np.load(f'{F}/C_{s}.npy') for s in shards])[keep][J],
     'question numbers': QN}
TEXT = np.array([gen[i]['text'] for i in ids], dtype=object)


def scores(name, target, seed):
    y = need.astype(int) if target == 'code-needed' else gain
    s = np.zeros(len(ids))
    split = StratifiedKFold(5, shuffle=True, random_state=seed).split(s, y) if target == 'code-needed' else \
        KFold(5, shuffle=True, random_state=seed).split(s)
    for tr, te in split:
        if name == 'text':
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True,
                                  token_pattern=r'(?u)\b\w+\b|[+\-*/=]')
            A, B = vec.fit_transform(TEXT[tr]), vec.transform(TEXT[te])
            m = Ridge(alpha=10.0) if target == 'gain' else LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')
        else:
            A, B = X[name][tr], X[name][te]
            big = A.shape[1] > 256
            m = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                              Ridge(alpha=100.0) if target == 'gain' else
                              LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))
        m.fit(A, y[tr])
        s[te] = m.predict(B) if target == 'gain' else m.decision_function(B)
    return name, target, s


with open(f'{FIX}/{ST}_needed_examples.txt', 'w', encoding='utf-8') as f:
    order = np.argsort(-gain)
    for j in [k for k in order if need[k]][:40]:
        i = ids[j]
        f.write(f"\n=== {i} gain {gain[j]:+.2f} (restart {R[j].mean():.2f}, {ST} {Cd[j].mean():.2f}) audit {aud[j]} "
                f"other-4 {other_acc[j]:.2f}\nQ: {gen[i]['question']}\nWRONG ({gen[i]['pred']:g}, gold {gen[i]['gold']:g}):\n"
                f"{gen[i]['text'][:900]}\n")
print(f'\nexamples: {FIX}/{ST}_needed_examples.txt')

if need.sum() < 20:
    sys.exit(f'\nprobe skipped: only {need.sum()} {ST}-needed answers')
NAMES = list(X) + ['text']
with parallel_backend('loky', inner_max_num_threads=1):
    out = Parallel(n_jobs=-1)(delayed(scores)(n, t, s) for n in NAMES for t in ('gain', 'code-needed') for s in range(REPS))
S = collections.defaultdict(list)
for n, t, s in out:
    S[(n, t)].append(s)
S = {k: np.mean(v, 0) for k, v in S.items()}
for t in ('gain', 'code-needed'):
    S[('other-4 accuracy (reference)', t)] = -other_acc
NAMES.append('other-4 accuracy (reference)')
boot = [rng.integers(0, len(ids), len(ids)) for _ in range(500)]
met = lambda t, s, b: spearmanr(s[b], gain[b]).correlation if t == 'gain' else roc_auc_score(need[b], s[b])
print(f'\n## probe: gain (Spearman) and {ST}-needed (AUROC); [95% CI of hidden r_mean minus this input]')
for n in NAMES:
    cells = []
    for t in ('gain', 'code-needed'):
        v = met(t, S[(n, t)], np.arange(len(ids)))
        if n == 'hidden r_mean':
            cells.append(f'{v:.3f}')
        else:
            d = [met(t, S[('hidden r_mean', t)], b) - met(t, S[(n, t)], b) for b in boot]
            cells.append(f'{v:.3f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]')
    print(f'  {n:30s} gain {cells[0]:24s} needed {cells[1]}')

