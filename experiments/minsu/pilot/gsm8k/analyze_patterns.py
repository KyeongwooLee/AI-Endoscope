"""Answer-distribution patterns of the wrong answers and which correction fits each pattern.

Pattern (from restart attempts 1-16, so restarts 17-32 stay unbiased for scoring):
  dominant   correct in at least half
  stuck      one wrong answer in at least half
  competing  correct and one wrong answer each in at least 20%
  spread     more than 5 distinct answers (of 16)
  other      none of these
Corrections, fix rate per pattern (one try = mean over that strategy's attempts; restart = attempts 17-32):
  every single-attempt strategy found in <fix_dir>
  vote k     majority answer of k restarts drawn from 17-32 (ties: random), averaged over 200 draws
  mixed k    majority over k/2 restarts (17-32) + k/2 code answers
Paired bootstrap over items for each strategy minus restart within each pattern.
usage: python -I analyze_patterns.py <fix_dir>
"""
import collections, glob, json, sys
import numpy as np

D = sys.argv[1]
att = collections.defaultdict(lambda: collections.defaultdict(list))
pred = collections.defaultdict(lambda: collections.defaultdict(list))
gold = {}
for f in sorted(glob.glob(f'{D}/*.jsonl')) + sorted(glob.glob(f'{D}/more*/*.jsonl')):
    for r in map(json.loads, open(f)):
        att[r['cond']][r['id']] += r['correct']
        pred[r['cond']][r['id']] += r['preds']
        gold[r['id']] = r['gold']
ids = sorted(att['restart'])
key = lambda x: None if x is None else round(x, 4)
NAMES = ['dominant', 'stuck', 'competing', 'spread', 'other']


def pattern(i):
    p = [key(x) for x in pred['restart'][i][:16]]
    ok = [x is not None and abs(x - gold[i]) < 1e-6 for x in p]
    wrong = collections.Counter(x for x, o in zip(p, ok) if not o)
    c, w = np.mean(ok), (wrong.most_common(1)[0][1] / 16 if wrong else 0)
    return 'dominant' if c >= .5 else 'stuck' if w >= .5 else 'competing' if c >= .2 and w >= .2 else \
        'spread' if len(set(p)) > 5 else 'other'


pat = {i: pattern(i) for i in ids}
rng = np.random.default_rng(0)


def vote(answers, g):
    c = collections.Counter(a for a in answers if a is not None)
    if not c:
        return 0.0
    top = max(c.values())
    winners = [a for a, n in c.items() if n == top]
    return np.mean([abs(a - g) < 1e-6 for a in winners])  # expected correctness under a random tie-break


score = {'restart': {i: np.mean(att['restart'][i][16:]) for i in ids}}
for c in att:
    if c != 'restart':
        score[c] = {i: np.mean(v) for i, v in att[c].items()}
for k in (4, 8, 16):
    score[f'vote {k}'] = {i: np.mean([vote([key(x) for x in rng.choice(pred['restart'][i][16:], k, replace=False)], gold[i])
                                      for _ in range(200)]) for i in ids}
for k in (4, 8, 16):
    score[f'mixed {k}'] = {i: np.mean([vote([key(x) for x in list(rng.choice(pred['restart'][i][16:], k // 2, replace=False)) +
                                             list(rng.choice(pred['code'][i], k // 2, replace=False))], gold[i])
                                       for _ in range(200)]) for i in ids if i in pred['code']}
order = ['restart', 'vote 4', 'vote 8', 'vote 16', 'mixed 4', 'mixed 8', 'mixed 16'] + \
        [c for c in ('code', 'calc', 'plan', 'decompose', 'equation', 'verify', 'notx', 'hot', 'cut', 'line') if c in score]
cnt = collections.Counter(pat.values())
print('patterns (restarts 1-16): ' + '  '.join(f'{n} {cnt[n]}' for n in NAMES))
print(f"\n{'strategy':10s} " + ' '.join(f'{n:>22s}' for n in NAMES + ['all']))
for c in order:
    cells = []
    for n in NAMES + ['all']:
        s = [i for i in ids if (n == 'all' or pat[i] == n) and i in score[c]]
        if len(s) < 5:
            cells.append(f"{'':>22s}")
            continue
        v = np.mean([score[c][i] for i in s])
        if c == 'restart':
            cells.append(f'{v:22.1%}')
            continue
        d = np.array([score[c][i] - score['restart'][i] for i in s])
        b = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
        lo, hi = np.percentile(b, [2.5, 97.5])
        mark = '+' if lo > 0 else '-' if hi < 0 else ' '
        cells.append(f'{v:6.1%} ({d.mean():+5.1%}){mark} n={len(s):<4d}')
    print(f'{c:10s} ' + ' '.join(cells))
print("\n(difference to restart in parentheses; + / - = 95% CI excludes 0; vote/mixed use k answers, i.e. k times the cost)")
