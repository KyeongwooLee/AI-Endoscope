"""Do the correction strategies fix DIFFERENT wrong answers? Multi-label table over all strategies (32 attempts each).

needed(X) = strategy X fixes the wrong answer more often than restarting (one-sided Fisher exact test, p < 0.05).
Chance level of needed(X) from a per-item null with both arms at their pooled rate. Overlap: items needed by both
strategies; unique: needed by X and by no other strategy. Oracle: picking the best strategy per item (upper bound).
usage: python -I analyze_needed.py <fix_dir>   (reads <fix_dir>/*.jsonl and <fix_dir>/more*/*.jsonl)
"""
import collections, glob, json, sys
import numpy as np
from scipy.stats import fisher_exact

D = sys.argv[1]
att = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f'{D}/*.jsonl')) + sorted(glob.glob(f'{D}/more*/*.jsonl')):
    for r in map(json.loads, open(f)):
        att[r['cond']][r['id']] += r['correct']
ids = sorted(att['restart'])
ORDER = ['code', 'calc', 'plan', 'decompose', 'equation', 'verify', 'cut', 'line']
S = [c for c in ORDER if att[c] and all(len(v) == 32 for v in att[c].values())]
rs = {i: sum(att['restart'][i]) for i in ids}
rng = np.random.default_rng(0)
need, rate = {}, {}
print(f"items {len(ids)}; restart fix rate {np.mean([rs[i] / 32 for i in ids]):.1%}")
print(f"\n{'strategy':10s} {'n':>5s} {'fix rate':>8s} {'restart':>8s} {'needed':>8s} {'chance':>7s} {'restart better':>15s}")
for c in S:
    s = [i for i in ids if i in att[c]]
    a = {i: sum(att[c][i]) for i in s}
    need[c] = {i for i in s if fisher_exact([[a[i], 32 - a[i]], [rs[i], 32 - rs[i]]], alternative='greater')[1] < .05}
    worse = sum(fisher_exact([[a[i], 32 - a[i]], [rs[i], 32 - rs[i]]], alternative='less')[1] < .05 for i in s)
    pool = np.array([(a[i] + rs[i]) / 64 for i in s])
    null = np.mean([np.mean([fisher_exact([[x, 32 - x], [y, 32 - y]], alternative='greater')[1] < .05
                             for x, y in zip(rng.binomial(32, pool), rng.binomial(32, pool))]) for _ in range(10)])
    rate[c] = {i: a[i] / 32 for i in s}
    print(f"{c:10s} {len(s):5d} {np.mean(list(rate[c].values())):8.1%} {np.mean([rs[i] / 32 for i in s]):8.1%} "
          f"{len(need[c]) / len(s):8.1%} {null:7.1%} {worse / len(s):15.1%}")

print('\n## overlap of needed sets (count of items needed by both; diagonal = needed)')
print(' ' * 10 + ''.join(f'{c:>10s}' for c in S))
for a in S:
    print(f'{a:10s}' + ''.join(f'{len(need[a] & need[b]):10d}' for b in S))
print('\n## unique: needed by this strategy and no other')
for c in S:
    other = set().union(*(need[o] for o in S if o != c))
    print(f'  {c:10s} {len(need[c] - other):4d} of {len(need[c])}')
anyn = set().union(*need.values())
print(f'\nneeded by at least one strategy: {len(anyn)} ({len(anyn) / len(ids):.1%})')

print('\n## fix rate when each wrong answer gets the best of these strategies (upper bound)')
print('   chosen on attempts 1-16, scored on attempts 17-32, so the noise of picking a maximum is not counted')
half = lambda c, i, p: np.mean(att[c][i][p])
for pool in (['restart'], ['restart', 'code'], ['restart'] + [c for c in S if c not in ('cut', 'line')], ['restart'] + S):
    got = []
    for i in ids:
        cand = [c for c in pool if i in att[c]]
        pick = max(cand, key=lambda c: half(c, i, slice(0, 16)))
        got.append(half(pick, i, slice(16, 32)))
    print(f"  {' + '.join(pool):70s} {np.mean(got):.1%}")
