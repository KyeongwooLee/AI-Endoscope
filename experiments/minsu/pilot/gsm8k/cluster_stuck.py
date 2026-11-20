"""The 'stuck' cluster of contrastive PCA (words: cannot, negative, however, nearest...): real, and does it need hidden states?

Contrastive PCA is fitted with the wrong/correct split, so a cluster free of correct solutions on the fitting data proves
nothing. Check on held-out problems: refit contrastive PCA and k-means(5) on 4 folds, take the cluster with the highest
wrong/correct ratio there (n >= 50), and measure how much of the held-out wrong answers and correct solutions land in it.
Text markers need no hidden state at all: trouble = the model says something is off, rounding = it rounds or approximates.
Compared on the same rows and metric (AUROC wrong vs correct): markers vs held-out cluster membership.
usage: python -I cluster_stuck.py <out_dir> <correct_dir>
"""
import collections, glob, json, pickle, re, sys
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUT, CORR = sys.argv[1:3]
T = pickle.load(open(f'{OUT}/rows.pkl', 'rb'))
prob = np.array([f'{k[0]}-{k[1]}' for k in T['key']])
TROUBLE = re.compile(r"\b(cannot|can't|negative|re-?evaluate|re-?examine|impossible|not possible|no valid|no solution|mistake|incorrect|error|"
                     r"does ?n[o']t make sense|seems? (to be )?(a |an )?(wrong|off|error|mistake|incorrect)|however)\b", re.I)
ROUND = re.compile(r'\b(nearest|rounded|rounding|rounds? (up|down|off|to)|round it|approximately|approx)\b', re.I)  # not 'round trip'
trouble = np.array([bool(TROUBLE.search(t.split('####')[0])) for t in T['text']])
rounding = np.array([bool(ROUND.search(t.split('####')[0])) for t in T['text']])
loop = np.array([t.count('####') >= 3 for t in T['text']])
wrong_all = ~T['correct']
tr_all = np.array([k[0] == 'train' for k in T['key']])
print('## text markers, all train solutions (format ok)')
for name, m in (('trouble', trouble), ('rounding', rounding), ('either', trouble | rounding), ('#### repeated 3+ times', loop)):
    print(f'  {name:24s} wrong {m[tr_all & wrong_all].mean():6.1%}  correct {m[tr_all & ~wrong_all].mean():6.1%}   '
          f'n wrong {m[tr_all & wrong_all].sum():5d} n correct {m[tr_all & ~wrong_all].sum():5d}   '
          f'wrong rate when present {wrong_all[tr_all & m].mean():6.1%} (overall {wrong_all[tr_all].mean():.1%})')


def contrast(Z, Zp, top=10):
    _, V = eigh(np.cov(Z.T), np.cov(Zp.T) + 1e-6 * np.eye(Z.shape[1]))
    return V[:, ::-1][:, :top]


for src in ('qres', 'diff'):
    d = np.load(f'{OUT}/rep_{src}.npz')
    Z, Zp, iz, ip = d['Z'], d['Zp'], d['idx'], d['idx_p']
    gz, gp = prob[iz], prob[ip]
    ups = np.unique(np.r_[gz, gp])
    mem_w, mem_c = np.zeros(len(Z), bool), np.zeros(len(Zp), bool)
    sizes = []
    for a, b in GroupKFold(5).split(ups, groups=ups):
        tr = list(ups[a])
        mz, mp = np.isin(gz, tr), np.isin(gp, tr)
        V = contrast(Z[mz], Zp[mp])
        m0 = Z[mz].mean(0)
        km = KMeans(5, n_init=10, random_state=0).fit((Z[mz] - m0) @ V)
        lw, lc = km.labels_, km.predict((Zp[mp] - m0) @ V)
        ok = [c for c in range(5) if (lw == c).sum() >= 50]
        c = max(ok, key=lambda x: (lw == x).mean() / max((lc == x).mean(), 1e-3))
        sizes.append((lw == c).sum())
        mem_w[~mz] = km.predict((Z[~mz] - m0) @ V) == c
        mem_c[~mp] = km.predict((Zp[~mp] - m0) @ V) == c
    print(f'\n## contrastive from {src}: most wrong-specific cluster, held-out problems (train-fold sizes {sizes})')
    print(f'  wrong answers in it {mem_w.mean():6.1%} ({mem_w.sum()}), correct solutions in it {mem_c.mean():6.1%} ({mem_c.sum()})')
    tw, tc = trouble[iz], trouble[ip]
    rw, rc = rounding[iz], rounding[ip]
    print(f'  markers among wrong: in cluster trouble {tw[mem_w].mean():6.1%} rounding {rw[mem_w].mean():6.1%}; outside trouble {tw[~mem_w].mean():6.1%} rounding {rw[~mem_w].mean():6.1%}')
    y = np.r_[np.ones(len(Z), int), np.zeros(len(Zp), int)]
    for name, s in (('cluster member', np.r_[mem_w, mem_c]), ('trouble marker', np.r_[tw, tc]), ('rounding marker', np.r_[rw, rc]),
                    ('either marker', np.r_[tw | rw, tc | rc])):
        print(f'  AUROC wrong vs correct, {name:16s} {roc_auc_score(y, s):.3f}')

by = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{CORR}/correct_*.jsonl')):
    for r in map(json.loads, open(f)):
        by[r['cond']][r['id']] = np.mean(r['correct'])
sid = lambda k: f'{k[0]}-{k[1]}-{k[2]}'
print('\n## test wrong answers (all non-NOEQ) by marker: audited class and correction')
d = np.load(f'{OUT}/rep_qres_all.npz')
it = d['idx_t']
ids = [sid(T['key'][i]) for i in it]
aud = T['aud'][it]
print(f"{'group':12s} {'n':>5s} {'audited':>7s} {'CALC':>6s} {'TARGET':>6s} {'SETUP':>6s} {'restart':>8s} {'refine':>8s} {'true_plain':>10s} {'true_typed':>10s}")
for name, m in (('trouble', trouble[it]), ('rounding', rounding[it] & ~trouble[it]), ('neither', ~trouble[it] & ~rounding[it])):
    a = aud[m][aud[m] != '']
    sel = [i for i, x in zip(ids, m) if x]
    rate = lambda c: np.mean([by[c][i] for i in sel if i in by[c]])
    print(f"{name:12s} {m.sum():5d} {len(a):7d} " + ' '.join(f'{(a == x).mean():6.1%}' for x in ('CALC', 'TARGET', 'SETUP')) +
          f" {rate('restart'):8.1%} {rate('refine'):8.1%} {rate('true_plain'):10.1%} {rate('true_typed'):10.1%}")
