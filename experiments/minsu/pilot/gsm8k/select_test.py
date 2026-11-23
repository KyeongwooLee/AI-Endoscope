"""Held-out check of detection + probe vote: train every probe on GSM8K train answers, apply once to GSM8K test.

Train: gen_train (greedy + 4 sampled per question, T=0.7) with feat_train. Test: gen_test (same format) with feat_test.
Probes (fit once on all train answers, never on test):
  hidden : per-answer "is this answer wrong", r_mean L20, PCA 256 + logistic regression
  conf   : same target on the 9 confidence features (logit baseline)
Reported on test: greedy-answer wrong AUROC (detection), accuracy of greedy / majority / probe vote / probe pick /
conf vote for K = 3, 5, paired bootstrap vs majority, and the detect -> fix pipeline (flag top q% by greedy P(wrong)).
Only aggregate numbers are written (no per-item test output).
usage: python select_test.py <gen_train> <feat_train> <gen_test> <feat_test> <out.txt>
"""
import collections, glob, json, sys, warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
GTR, FTR, GTE, FTE, OUT = sys.argv[1:6]
key = lambda x: None if x is None else round(x, 4)
log = open(OUT, 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True); log.write(s + '\n'); log.flush()


def load(gen, feat, split):
    ans = collections.defaultdict(dict)
    for f in sorted(glob.glob(f'{gen}/{split}_*.jsonl')):
        for l in open(f):
            r = json.loads(l)
            ans[r['pid']][r['sample']] = (key(r['pred']), r['correct'], r['format_ok'])
    meta = json.load(open(f'{feat}/meta_0.json'))
    L20, POS = meta['layers'].index(20), meta['positions']
    row = {(m['pid'], m['sample']): j for j, m in enumerate(meta['meta']) if m['split'] == split}
    pids = sorted(p for p in ans if len(ans[p]) == 5 and all((p, s) in row for s in range(5)))
    J = np.array([[row[(p, s)] for s in range(5)] for p in pids])
    H = np.load(f'{feat}/H_0.npy', mmap_mode='r')
    srt = np.sort(J.ravel())
    Hs = np.asarray(H[srt, POS.index('r_mean'), L20], dtype=np.float32)
    Hr = Hs[np.searchsorted(srt, J.ravel())].reshape(len(pids), 5, -1)
    C = np.load(f'{feat}/C_0.npy')[J].astype(np.float32)
    A = [[ans[p][s][0] for s in range(5)] for p in pids]
    R = np.array([[ans[p][s][1] for s in range(5)] for p in pids], dtype=float)
    return pids, Hr, C, A, R


_, Htr, Ctr, _, Rtr = load(GTR, FTR, 'train')
pids, Hte, Cte, A, R = load(GTE, FTE, 'test')
n = len(pids)
say(f'train questions {len(Rtr)} (answers {Rtr.size}); test questions {n}; test greedy acc {R[:, 0].mean():.1%}, '
    f'sampled acc {R[:, 1:].mean():.1%}')
PW = {}
for name, Xtr, Xte, big in (('hidden', Htr, Hte, True), ('conf', Ctr, Cte, False)):
    m = make_pipeline(StandardScaler(), *([PCA(256, random_state=0)] if big else []),
                      LogisticRegression(C=0.05 if big else 1.0, max_iter=3000, class_weight='balanced'))
    m.fit(Xtr.reshape(-1, Xtr.shape[-1]), 1 - Rtr.ravel())
    PW[name] = m.predict_proba(Xte.reshape(-1, Xte.shape[-1]))[:, 1].reshape(n, 5)
wrong_g = R[:, 0] == 0
rng = np.random.default_rng(0)
boot = [rng.integers(0, n, n) for _ in range(1000)]
say('\n## detection on test: is the greedy answer wrong? AUROC [95% bootstrap CI]')
for name in PW:
    s = PW[name][:, 0]
    b = [roc_auc_score(wrong_g[i], s[i]) for i in boot]
    say(f'  {name:7s} {roc_auc_score(wrong_g, s):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]'
        f'   (all answers {roc_auc_score(1 - R.ravel(), PW[name].ravel()):.3f})')


def choose(K, rule):
    out = np.zeros(n, int)
    for i in range(n):
        a = A[i][:K]
        if rule == 'greedy':
            continue
        if rule == 'majority':
            cnt = collections.Counter(x for x in a if x is not None)
            best = max(cnt.values()) if cnt else 0
            out[i] = next((s for s in range(K) if a[s] is not None and cnt[a[s]] == best), 0)
        elif rule == 'probe pick':
            out[i] = int(np.argmin(PW['hidden'][i, :K]))
        else:
            pw = (PW['hidden'] if rule == 'probe vote' else PW['conf'])[i, :K]
            w = collections.defaultdict(float)
            for s in range(K):
                if a[s] is not None:
                    w[a[s]] += 1 - pw[s]
            best = max(w, key=w.get) if w else None
            out[i] = next((s for s in range(K) if a[s] == best), 0)
    return out


RULES = ['greedy', 'majority', 'probe vote', 'probe pick', 'conf vote']
acc = {(K, r): R[np.arange(n), choose(K, r)] for K in (3, 5) for r in RULES}
for K in (3, 5):
    acc[(K, 'oracle')] = R[:, :K].max(1)
groups = [('all', np.ones(n, bool)), ('greedy-wrong', wrong_g), ('greedy-right', ~wrong_g)]
for K in (3, 5):
    say(f'\n## test, K={K} answers (greedy + {K - 1} samples): accuracy of the chosen answer')
    say(f"{'rule':12s}" + ''.join(f'{g:>15s}' for g, _ in groups))
    say(f"{'n':12s}" + ''.join(f'{m.sum():15d}' for _, m in groups))
    for r in RULES + ['oracle']:
        say(f'{r:12s}' + ''.join(f'{acc[(K, r)][m].mean():15.1%}' for _, m in groups))
say('\n## test, paired difference vs majority, %p [95% bootstrap CI]')
for K in (3, 5):
    for r in ('probe vote', 'probe pick', 'conf vote'):
        d = acc[(K, r)] - acc[(K, 'majority')]
        b = [d[i].mean() for i in boot]
        say(f'  K={K} {r:11s} {100 * d.mean():+.2f} [{100 * np.percentile(b, 2.5):+.2f}, {100 * np.percentile(b, 97.5):+.2f}]'
            f'   wins {int((d > 0).sum())} / losses {int((d < 0).sum())}')
say('\n## test pipeline: flag top q% by greedy P(wrong), apply the rule (K=5) only there; cost = 1 + 4q')
say(f"{'rule':24s}" + ''.join(f'{f"q={q:.0%}":>10s}' for q in (0.25, 0.4)))
for det in ('hidden', 'conf'):
    for r in ('majority', 'probe vote', 'oracle'):
        cells = []
        for q in (0.25, 0.4):
            fl = PW[det][:, 0] >= np.quantile(PW[det][:, 0], 1 - q)
            cells.append(np.where(fl, acc[(5, r)], R[:, 0]).mean())
        say(f'{"detect(" + det + ") -> " + r:24s}' + ''.join(f'{c:10.1%}' for c in cells))
say(f'(greedy only {R[:, 0].mean():.1%}; all questions majority K=5 {acc[(5, "majority")].mean():.1%}, '
    f'probe vote K=5 {acc[(5, "probe vote")].mean():.1%}, cost 5)')
say('done')
