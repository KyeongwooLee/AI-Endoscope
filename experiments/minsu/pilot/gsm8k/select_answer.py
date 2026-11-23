"""Pick the final answer among K answers with a hidden-state probe instead of majority vote (Orgad et al. 2025, sec. 6).

Answers per question: greedy + 4 sampled (T=0.7) from gen_train; hidden r_mean L20 and 9 confidence features of each
answer from feat_train. A per-answer "is this answer wrong" probe is trained out-of-fold over questions (5-fold), so
every question is scored by a probe that never saw it. Gold is used only as the probe target and for scoring.
Selection rules over the first K answers (K = 3, 5):
  greedy          : keep the greedy answer
  majority        : most frequent answer (ties -> earliest, i.e. greedy first)
  probe pick      : answer with the lowest P(wrong)
  probe vote      : sum of (1 - P(wrong)) per distinct answer, take the max
  conf pick/vote  : same with a probe on the 9 confidence features (logit baseline)
  nll pick        : answer with the lowest mean token NLL (no training)
  oracle          : correct if any of the K answers is correct (upper bound)
Reported on all questions, greedy-wrong questions, by paper response type (rs30, K=30), and inside the
detect -> fix pipeline (flag top q% by the greedy answer's P(wrong), apply the rule only there).
usage: python select_answer.py <rs30_dir> <gen_dir> <feat_dir> <out.txt>
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


rs = {r['pid']: r for f in sorted(glob.glob(f'{RS}/rs_*.jsonl')) for r in map(json.loads, open(f))}
ans = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{GEN}/train_*.jsonl')):
    for l in open(f):
        r = json.loads(l)
        ans[r['pid']][r['sample']] = (r['pred'], r['format_ok'])
meta = json.load(open(f'{F}/meta_0.json'))
L20, POS = meta['layers'].index(20), meta['positions']
row = {(m['pid'], m['sample']): j for j, m in enumerate(meta['meta'])}
pids = sorted(p for p in rs if all((p, s) in row for s in range(5)) and len(ans[p]) == 5 and ans[p][0][1])
n = len(pids)
H = np.load(f'{F}/H_0.npy', mmap_mode='r')
J = np.array([[row[(p, s)] for s in range(5)] for p in pids])
srt = np.sort(J.ravel())
Hs = np.asarray(H[srt, POS.index('r_mean'), L20], dtype=np.float32)
Hr = Hs[np.searchsorted(srt, J.ravel())].reshape(n, 5, -1)
C = np.load(f'{F}/C_0.npy')[J].astype(np.float32)                        # (n, 5, 9); C[..., 0] = nll_mean
A = [[key(ans[p][s][0]) for s in range(5)] for p in pids]
gold = [rs[p]['gold'] for p in pids]
R = np.array([[ok(ans[p][s][0], rs[p]['gold']) for s in range(5)] for p in pids], dtype=float)   # answer correct
wrong_g = R[:, 0] == 0
T = {t: np.array([types(rs[p]['preds'], rs[p]['gold'])[t] for p in pids]) for t in 'BCDE'}
say(f'questions {n}; greedy acc {R[:, 0].mean():.1%}; sampled acc {R[:, 1:].mean():.1%}')

folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(pids, wrong_g))


def fit(X, big, tr, te):
    m = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                      LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))
    m.fit(X[tr].reshape(-1, X.shape[-1]), 1 - R[tr].ravel())
    return te, m.predict_proba(X[te].reshape(-1, X.shape[-1]))[:, 1].reshape(len(te), 5)


PW = {}
for name, X, big in (('hidden', Hr, True), ('conf', C, False)):
    PW[name] = np.zeros((n, 5))
    for te, p in Parallel(n_jobs=5)(delayed(fit)(X, big, tr, te) for tr, te in folds):
        PW[name][te] = p
    say(f'per-answer wrong AUROC ({name}): {roc_auc_score(1 - R.ravel(), PW[name].ravel()):.3f}  '
        f'| greedy answers only {roc_auc_score(wrong_g, PW[name][:, 0]):.3f}')


def choose(K, rule):
    """index (0..K-1) of the chosen answer per question."""
    out = np.zeros(n, int)
    for i in range(n):
        a = A[i][:K]
        if rule == 'greedy':
            continue
        if rule == 'majority':
            cnt = collections.Counter(x for x in a if x is not None)
            best = max(cnt.values()) if cnt else 0
            out[i] = next((s for s in range(K) if a[s] is not None and cnt[a[s]] == best), 0)
        elif rule.endswith('pick'):
            sc = {'probe pick': PW['hidden'][i], 'conf pick': PW['conf'][i], 'nll pick': C[i, :, 0]}[rule][:K]
            out[i] = int(np.argmin(sc))
        elif rule.endswith('vote'):
            pw = (PW['hidden'] if rule == 'probe vote' else PW['conf'])[i][:K]
            w = collections.defaultdict(float)
            for s in range(K):
                if a[s] is not None:
                    w[a[s]] += 1 - pw[s]
            best = max(w, key=w.get) if w else None
            out[i] = next((s for s in range(K) if a[s] == best), 0)
    return out


RULES = ['greedy', 'majority', 'probe pick', 'probe vote', 'conf pick', 'conf vote', 'nll pick']
acc = {}
for K in (3, 5):
    for r in RULES:
        acc[(K, r)] = R[np.arange(n), choose(K, r)]
    acc[(K, 'oracle')] = R[:, :K].max(1)

groups = [('all', np.ones(n, bool)), ('greedy-wrong', wrong_g), ('greedy-right', ~wrong_g)] + \
         [(f'wrong & {t}', wrong_g & T[t]) for t in 'BCDE']
for K in (3, 5):
    say(f'\n## K={K} answers (greedy + {K - 1} samples): accuracy of the chosen answer')
    say(f"{'rule':12s}" + ''.join(f'{g:>16s}' for g, _ in groups))
    say(f"{'n':12s}" + ''.join(f'{m.sum():16d}' for _, m in groups))
    for r in RULES + ['oracle']:
        say(f'{r:12s}' + ''.join(f'{acc[(K, r)][m].mean():16.1%}' for _, m in groups))

rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(1000)]
say('\n## paired difference vs majority, all questions, %p [95% bootstrap CI]')
for K in (3, 5):
    for r in ('probe pick', 'probe vote', 'conf pick', 'conf vote', 'nll pick'):
        d = acc[(K, r)] - acc[(K, 'majority')]
        b = [d[i].mean() for i in boot]
        say(f'  K={K} {r:11s} {100 * d.mean():+.2f} [{100 * np.percentile(b, 2.5):+.2f}, {100 * np.percentile(b, 97.5):+.2f}]'
            f'   wins {int((d > 0).sum())} / losses {int((d < 0).sum())}')

say('\n## pipeline: flag top q% by greedy P(wrong) (hidden), apply the rule only to flagged questions')
say(f"{'rule':12s}" + ''.join(f'{f"q={q:.0%} K={K}":>14s}' for q in (0.25, 0.4) for K in (3, 5)))
for r in RULES[1:] + ['oracle']:
    cells = []
    for q in (0.25, 0.4):
        fl = PW['hidden'][:, 0] >= np.quantile(PW['hidden'][:, 0], 1 - q)
        for K in (3, 5):
            cells.append(np.where(fl, acc[(K, r)], R[:, 0]).mean())
    say(f'{r:12s}' + ''.join(f'{c:14.1%}' for c in cells))
say(f'(greedy only {R[:, 0].mean():.1%}; cost per flagged question = K-1 extra answers)')
say('done')
