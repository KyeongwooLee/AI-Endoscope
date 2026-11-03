"""Is the location probe limited by label noise? Split its accuracy by the step-2 AI check of the rule location.

ab_labels.tsv (taxonomy_relabel/setup_ab/out) has loc=y when AI labelers judged the rule-located equation to be the
real first error, loc=n when the real first error was elsewhere. If accuracy is much higher on loc=y, noisy labels
(not the probe) cap the score. Also shows where the probe points relative to the rule location (pred - k).
usage: python -I check_loc_noise.py <preds_loc2.json> <ab_labels.tsv>
"""
import collections, json, sys

P = json.load(open(sys.argv[1]))
loc = {}
for line in open(sys.argv[2], encoding='utf-8').read().splitlines()[1:]:
    i, l = line.split('\t')[:2]
    loc[i] = l
both = [i for i in loc if i in P['k']]
print(f'AI-checked SETUP errors: {len(loc)}; in the location data: {len(both)} '
      f'(loc=y {sum(loc[i] == "y" for i in both)}, loc=n {sum(loc[i] == "n" for i in both)})')
names = ['position (list)', P['best_non_hidden'], 'v1: res L20 + position (indep)', P['selected'],
         'pre+res+mean L12-24 + conf + position (list, centered)']
print(f"\n  {'input':56s} {'loc=y':>7s} {'loc=n':>7s} {'all 5765':>9s}")
for name in dict.fromkeys(names):
    pr = P['pred'][name]
    acc = lambda ids: sum(pr[i] == P['k'][i] for i in ids) / len(ids)
    ys, ns = [i for i in both if loc[i] == 'y'], [i for i in both if loc[i] == 'n']
    print(f'  {name:56s} {acc(ys):7.1%} {acc(ns):7.1%} {acc(list(P["k"])):9.1%}')

pr = P['pred'][P['selected']]
print(f'\n## where the selected probe points, pred - k (rule location)')
for l in 'yn':
    ids = [i for i in both if loc[i] == l]
    c = collections.Counter(max(-2, min(2, pr[i] - P['k'][i])) for i in ids)
    print(f'  loc={l} (n={len(ids)}): ' + '  '.join(f'{d:+d}{"+" if abs(d) == 2 else ""}: {c[d] / len(ids):.0%}'
                                                 for d in (-2, -1, 0, 1, 2)))
    kc = collections.Counter(min(P['k'][i], 3) for i in ids)
    print(f'           rule location k: ' + '  '.join(f'{d}{"+" if d == 3 else ""}: {kc[d] / len(ids):.0%}' for d in range(4)))
