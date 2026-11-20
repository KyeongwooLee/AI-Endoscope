"""Topic clusters: k-means on the question-only hidden (q_end L20) of every problem, before any solution is written.

Is a topic itself an error type? Per topic: how often the model errs, which rule types its errors get, audited classes
and restart fix rate on test problems. Fit on train problems; test problems are only assigned. 95% CI: bootstrap over
problems. Also: can one topic hold several outcomes? Same-problem wrong answers with a correct sibling / other rule type.
usage: python cluster_topic.py <feat_dir> <auto_types.jsonl> <gen_dir> <cls_labels.tsv> <correct_dir> <k> [<k> ...]
"""
import collections, glob, json, os, re, sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

F, TYPES, GEN, AUDIT, CORR = sys.argv[1:6]
shards = sorted(int(f[5:-5]) for f in os.listdir(F) if f.startswith('meta_'))
metas = [json.load(open(f'{F}/meta_{s}.json')) for s in shards]
meta = [m for m in (m for d in metas for m in d['meta']) if m['format_ok']]  # same row order as H_all.npy
LAYERS, POS = metas[0]['layers'], metas[0]['positions']
Q = np.asarray(np.load(f'{F}/H_all.npy', mmap_mode='r')[:, POS.index('q_end'), LAYERS.index(20)], dtype=np.float32)
typ = {(t['split'], t['pid'], t['sample']): t['cat'] for t in map(json.loads, open(TYPES))}
question = {}
for f in glob.glob(f'{GEN}/*_*.jsonl'):
    for r in map(json.loads, open(f)):
        question[(r['split'], r['pid'])] = r['question']
audit = {}
for line in open(AUDIT, encoding='utf-8').read().splitlines()[1:]:
    x = line.split('\t')
    audit[x[0]] = x[1]
restart = {}
for f in sorted(glob.glob(f'{CORR}/correct_*.jsonl')):
    for r in map(json.loads, open(f)):
        if r['cond'] == 'restart':
            restart[r['id']] = np.mean(r['correct'])

sols, first = collections.defaultdict(list), {}
for i, m in enumerate(meta):
    p = (m['split'], m['pid'])
    sols[p].append(m)
    first.setdefault(p, i)
probs = sorted(sols)
is_tr = np.array([p[0] == 'train' for p in probs])
X = Q[[first[p] for p in probs]]
tok = re.compile(r'[a-z]+')
TYPE_ORDER = ['SETUP', 'NOEQ', 'TARGET', 'OTHER', 'ARITH', 'NEWNUM']
rng = np.random.default_rng(0)

print('## one topic, several outcomes (train problems)')
tr_probs = [p for p in probs if p[0] == 'train']
wrong_of = lambda p: [m for m in sols[p] if not m['correct']]
withw = [p for p in tr_probs if wrong_of(p)]
print(f'problems {len(tr_probs)}, with a wrong answer {len(withw)}, of those also with a correct answer '
      f'{np.mean([len(wrong_of(p)) < len(sols[p]) for p in withw]):.1%}')
cat_of = lambda m: typ[(m['split'], m['pid'], m['sample'])]
multi = [p for p in withw if sum(cat_of(m) != 'NOEQ' for m in wrong_of(p)) >= 2]
print(f'problems with >= 2 wrong answers excluding NOEQ {len(multi)}, of those with >= 2 different rule types '
      f'{np.mean([len({cat_of(m) for m in wrong_of(p) if cat_of(m) != "NOEQ"}) >= 2 for p in multi]):.1%}')

for k in map(int, sys.argv[6:]):
    pre = make_pipeline(StandardScaler(), PCA(50, random_state=0)).fit(X[is_tr])
    Z = pre.transform(X)
    km = KMeans(k, n_init=10, random_state=0).fit(Z[is_tr])
    lab = km.predict(Z)
    Zn = rng.normal(size=Z[is_tr].shape) * Z[is_tr].std(0)
    print(f'\n===== topics k={k}: silhouette {silhouette_score(Z[is_tr], lab[is_tr]):.3f} '
          f'(no-cluster Gaussian {silhouette_score(Zn, KMeans(k, n_init=5, random_state=0).fit(Zn).labels_):.3f}) =====')
    tr = np.where(is_tr)[0]
    err = np.array([len(wrong_of(probs[i])) / len(sols[probs[i]]) for i in range(len(probs))])
    arith = np.array([sum(cat_of(m) == 'ARITH' for m in wrong_of(probs[i])) for i in range(len(probs))])
    nw = np.array([len(wrong_of(probs[i])) for i in range(len(probs))])
    rows = [(j, m) for j in tr for m in wrong_of(probs[j])]
    y = [cat_of(m) for _, m in rows]
    print(f'overall train wrong rate {err[tr].mean():.1%}; AMI(topic, rule type) {adjusted_mutual_info_score(y, lab[[j for j, _ in rows]]):.3f}; '
          f'share of per-problem wrong-rate variance explained by topic '
          f'{sum((lab[tr] == c).sum() * (err[tr][lab[tr] == c].mean() - err[tr].mean()) ** 2 for c in range(k)) / ((err[tr] - err[tr].mean()) ** 2).sum():.1%}')
    print(f"{'c':>2s} {'probs':>5s} {'wrong%':>16s} {'wrong n':>7s} " + ' '.join(f'{t:>6s}' for t in TYPE_ORDER) + f" {'ARITH% 95CI':>16s} | test: {'n':>4s} {'CALC':>6s} {'TARGET':>6s} {'SETUP':>6s} {'restart':>7s}  words")
    for c in range(k):
        idx = tr[lab[tr] == c]
        ws = [m for j in idx for m in wrong_of(probs[j])]
        cats = collections.Counter(cat_of(m) for m in ws)
        be, ba = [], []
        for _ in range(1000):
            s = rng.choice(idx, len(idx))
            be.append(err[s].mean())
            ba.append(arith[s].sum() / max(nw[s].sum(), 1))
        te = np.where(~is_tr & (lab == c))[0]
        ids = [f"{m['split']}-{m['pid']}-{m['sample']}" for j in te for m in wrong_of(probs[j])]
        aud = [audit[i] for i in ids if i in audit]
        fix = [restart[i] for i in ids if i in restart]
        docs = [set(tok.findall(question[p].lower())) for p in probs]
        a, b = collections.Counter(), collections.Counter()
        for j in tr:
            (a if lab[j] == c else b).update(docs[j])
        na, nb = (lab[tr] == c).sum(), (lab[tr] != c).sum()
        lo = {w: np.log((a[w] + .5) / (na - a[w] + .5)) - np.log((b[w] + .5) / (nb - b[w] + .5)) for w in a if a[w] >= 15}
        print(f"{c:2d} {len(idx):5d} {err[idx].mean():6.1%} [{np.percentile(be, 2.5):4.1%},{np.percentile(be, 97.5):4.1%}] {len(ws):7d} " +
              ' '.join(f'{cats[t] / len(ws):6.1%}' for t in TYPE_ORDER) +
              f" {arith[idx].sum() / nw[idx].sum():5.1%} [{np.percentile(ba, 2.5):4.1%},{np.percentile(ba, 97.5):4.1%}] |       {len(aud):4d} " +
              ' '.join(f'{np.mean([x == t for x in aud]):6.1%}' for t in ('CALC', 'TARGET', 'SETUP')) +
              f" {np.mean(fix):7.1%}  " + ' '.join(sorted(lo, key=lo.get, reverse=True)[:10]))
