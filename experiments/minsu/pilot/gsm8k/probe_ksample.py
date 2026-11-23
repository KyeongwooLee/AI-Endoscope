"""Predict the paper's response types (Orgad et al. 2025) from K answers instead of the greedy answer alone.

Labels : types from all 30 rs30 samples (T=1), paper thresholds for K=30:
         B correct >= 15, C one wrong answer >= 15, D |correct - top wrong| <= 5 and both >= 5, E > 10 distinct answers.
Inputs : the greedy answer + the first K-1 sampled answers of gen_train (T=0.7, generated independently of rs30),
         with hidden r_mean L20 and confidence features of every answer (feat_train). Gold is never used as input.
Feature sets per K (K=1 is the greedy answer alone):
  agree           : answer agreement only (top share, 2nd share, distinct/K, share equal to greedy, format ok share)
  agree+conf      : + mean/min/max of the 9 confidence features over the K answers
  agree+pwrong    : + per-answer hidden P(wrong) (nested out-of-fold probe), mean/min/max/std and greedy's
  agree+hidden    : + mean hidden over the K answers (PCA 256)
  agree+all       : agree + conf + pwrong + hidden
  hidden mean     : mean hidden over the K answers only (no agreement features)
  question        : question-only hidden q_end L20 (difficulty baseline, same for every K)
Every number is out-of-fold over questions (5-fold). Types overlap, so each is a one-vs-rest probe.
usage: python probe_ksample.py <rs30_dir> <gen_dir> <feat_dir> <out.txt>
"""
import collections, glob, json, sys, warnings
import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
RS, GEN, F, OUT = sys.argv[1:5]
KS = (1, 3, 5)
TYPES = ['B', 'C', 'D', 'E']
NAME = {'B': 'mostly correct', 'C': 'same wrong (stuck)', 'D': 'two competing', 'E': 'many answers'}
key = lambda x: None if x is None else round(x, 4)
ok = lambda x, g: x is not None and abs(x - g) < 1e-6
log = open(OUT, 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); log.write(s + '\n'); log.flush()


def types(p, g):
    c = sum(ok(x, g) for x in p)
    w = collections.Counter(key(x) for x in p if x is not None and not ok(x, g))
    top = w.most_common(1)[0][1] if w else 0
    return {'B': c >= 15, 'C': top >= 15, 'D': abs(c - top) <= 5 and c >= 5 and top >= 5,
            'E': len({key(x) for x in p if x is not None}) > 10}


# ---- labels
rs = {r['pid']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
# ---- answers of gen_train: pid -> sample -> (pred, format_ok)
ans = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{GEN}/train_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        ans[r['pid']][r['sample']] = (r['pred'], r['format_ok'])
# ---- features
meta = json.load(open(f'{F}/meta_0.json'))
L20, POS = meta['layers'].index(20), meta['positions']
row = {(m['pid'], m['sample']): j for j, m in enumerate(meta['meta'])}
pids = sorted(p for p in rs if all((p, s) in row for s in range(5)) and len(ans[p]) == 5 and ans[p][0][1])
say(f'questions with rs30 labels, 5 answers and features, greedy format ok: {len(pids)}')
H = np.load(f'{F}/H_0.npy', mmap_mode='r')
J = np.array([[row[(p, s)] for s in range(5)] for p in pids])           # (n, 5)
order = np.argsort(J.ravel())
Hr = np.empty((J.size, H.shape[-1]), np.float32)
Hr[order] = np.asarray(H[np.sort(J.ravel()), POS.index('r_mean'), L20], dtype=np.float32)
Hr = Hr.reshape(len(pids), 5, -1)                                        # (n, 5, D)
Hq = np.asarray([H[j, POS.index('q_end'), L20] for j in J[:, 0]], dtype=np.float32)
C = np.load(f'{F}/C_0.npy')[J].astype(np.float32)                       # (n, 5, 9)
n = len(pids)
gold = np.array([rs[p]['gold'] for p in pids])
wrong_g = np.array([not rs[p]['greedy_correct'] for p in pids])
Y = {t: np.array([types(rs[p]['preds'], rs[p]['gold'])[t] for p in pids], dtype=int) for t in TYPES}
samp_wrong = np.array([[not ok(ans[p][s][0], rs[p]['gold']) for s in range(5)] for p in pids], dtype=int)  # target only
say(f'greedy wrong {wrong_g.mean():.1%} (n={wrong_g.sum()}); type share all / greedy-wrong: ' +
    '  '.join(f'{t} {Y[t].mean():.1%}/{Y[t][wrong_g].mean():.1%}' for t in TYPES))
say(f'gen_train sample accuracy (T=0.7): {1 - samp_wrong[:, 1:].mean():.1%}; greedy {1 - samp_wrong[:, 0].mean():.1%}')


def agree(K):
    out = []
    for p in pids:
        a = [key(ans[p][s][0]) for s in range(K)]
        cnt = collections.Counter(x for x in a if x is not None).most_common()
        top = cnt[0][1] if cnt else 0
        sec = cnt[1][1] if len(cnt) > 1 else 0
        out.append([top / K, sec / K, len(cnt) / K, sum(x == a[0] and x is not None for x in a) / K,
                    np.mean([ans[p][s][1] for s in range(K)])])
    return np.array(out, np.float32)


def lr(big, n_tr):
    return make_pipeline(StandardScaler(), *([PCA(min(256, n_tr - 1), random_state=0)] if big else []),
                         LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))


folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(pids, wrong_g * 16 + sum(Y[t] << i for i, t in enumerate(TYPES))))


# ---- per-answer hidden P(wrong), nested out-of-fold: for each outer fold, inner OOF on its train part, full fit for test
def pwrong_fold(tr, te):
    Xs = lambda idx: Hr[idx].reshape(-1, Hr.shape[-1])
    ys = lambda idx: samp_wrong[idx].ravel()
    out = np.zeros((n, 5), np.float32)
    m = lr(True, len(tr) * 5).fit(Xs(tr), ys(tr))
    out[te] = m.predict_proba(Xs(te))[:, 1].reshape(len(te), 5)
    for itr, ite in StratifiedKFold(5, shuffle=True, random_state=1).split(tr, wrong_g[tr]):
        m = lr(True, len(itr) * 5).fit(Xs(tr[itr]), ys(tr[itr]))
        out[tr[ite]] = m.predict_proba(Xs(tr[ite]))[:, 1].reshape(len(ite), 5)
    return out


say('\nnested per-answer P(wrong) probes ...')
PW = Parallel(n_jobs=5)(delayed(pwrong_fold)(tr, te) for tr, te in folds)   # PW[f]: features valid for outer fold f
for f, (tr, te) in enumerate(folds):
    say(f'  fold {f}: per-answer wrong AUROC on test questions {roc_auc_score(samp_wrong[te].ravel(), PW[f][te].ravel()):.3f}')


def feats(K, kind, f):
    """feature matrix for outer fold f (pwrong features depend on the fold)."""
    parts = []
    if kind == 'question':
        return Hq, True
    if kind == 'hidden mean':
        return Hr[:, :K].mean(1), True
    parts.append(agree(K))
    if kind in ('agree+conf', 'agree+all'):
        c = C[:, :K]
        parts.append(np.concatenate([c.mean(1), c.min(1), c.max(1)], 1))
    if kind in ('agree+pwrong', 'agree+all'):
        pw = PW[f][:, :K]
        parts.append(np.stack([pw.mean(1), pw.min(1), pw.max(1), pw.std(1), pw[:, 0]], 1))
    small = np.concatenate(parts, 1)
    if kind in ('agree+hidden', 'agree+all'):
        return np.concatenate([Hr[:, :K].mean(1), small], 1), True
    if kind == 'agree+question':
        return np.concatenate([Hq, small], 1), True
    return small, False


KINDS = ['hidden mean', 'agree', 'agree+question', 'agree+conf', 'agree+pwrong', 'agree+hidden', 'agree+all']
JOBS = [('question', 1)] + [('greedy hidden', 1)] + [(k, K) for K in KS for k in KINDS]


def run(kind, K, t, f):
    tr, te = folds[f]
    if kind == 'greedy hidden':
        X, big = Hr[:, 0], True
    else:
        X, big = feats(K, kind, f)
    m = lr(big, len(tr)).fit(X[tr], Y[t][tr])
    return kind, K, t, te, m.predict_proba(X[te])[:, 1]


say('type probes ...')
res = Parallel(n_jobs=30)(delayed(run)(k, K, t, f) for k, K in JOBS for t in TYPES for f in range(5))
P = {}
for kind, K, t, te, p in res:
    P.setdefault((kind, K, t), np.zeros(n))[te] = p

rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(500)]
for grp, msk in (('greedy-wrong questions', wrong_g), ('all questions', np.ones(n, bool))):
    say(f'\n## one-vs-rest AUROC, {grp} (n={msk.sum()})')
    say(f"{'features':16s} {'K':>2s} " + ''.join(f'{t + " " + NAME[t]:>24s}' for t in TYPES))
    for kind, K in JOBS:
        say(f'{kind:16s} {K:2d} ' + ''.join(f'{roc_auc_score(Y[t][msk], P[(kind, K, t)][msk]):24.3f}' for t in TYPES))

say('\n## does hidden add to answer agreement (and to agreement + question)? greedy-wrong questions, AUROC difference [95% bootstrap CI]')
idx = np.where(wrong_g)[0]
for K in KS:
    for other, base in (('hidden mean', 'question'), ('agree+hidden', 'hidden mean'), ('agree+pwrong', 'agree'), ('agree+hidden', 'agree'), ('agree+conf', 'agree'),
                        ('agree+hidden', 'agree+question'), ('agree+all', 'agree+question')):
        cells = []
        for t in TYPES:
            y, a, b = Y[t][idx], P[(other, K, t)][idx], P[(base, 1 if base == 'question' else K, t)][idx]
            d = []
            for bi in boot:
                i = bi[:len(idx)] % len(idx)
                if y[i].min() == y[i].max():
                    continue
                d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
            cells.append(f'{roc_auc_score(y, a) - roc_auc_score(y, b):+.3f} [{np.percentile(d, 2.5):+.3f},{np.percentile(d, 97.5):+.3f}]')
        say(f'K={K} {other:13s} - {base:14s}: ' + '  '.join(f'{t} {c}' for t, c in zip(TYPES, cells)))
np.savez(OUT.replace('.txt', '.npz'), pids=np.array(pids), wrong=wrong_g, **{f'Y_{t}': Y[t] for t in TYPES},
         **{f'P_{k}_{K}_{t}'.replace(' ', '_').replace('+', 'p'): v for (k, K, t), v in P.items()})
say('done')
