"""Correction types from fix_exp.py: which correction fixes each wrong answer?

fix rate = share of the N attempts that end with the gold answer. A correction 'works' on an item when its fix rate is
at least THR. Type = the first correction that works, in order of how little it needs:
  retry   restart works
  calc    restart does not, the calculator does (only items with an arithmetic slip in an equation)
  code    neither, a Python program does
  line    none of these, a location correction does: continuing from just before the audited first wrong line (cut), or
          quoting that line (line)
  stuck   nothing works
Noise: the same typing from attempts 1-4 vs 5-8 (agreement and kappa). Paired bootstrap over items for each correction
minus restart on the items where it applies.
usage: python -I analyze_fix.py <fix_dir> [<cls_labels.tsv of the audit>]
"""
import collections, glob, json, sys
import numpy as np

D = sys.argv[1]
by = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{D}/fix_*.jsonl')) + sorted(glob.glob(f'{D}/line_*.jsonl')) + sorted(glob.glob(f'{D}/cut_*.jsonl')):
    for r in map(json.loads, open(f)):
        by[r['cond']][r['id']] = r
audit = {}
if len(sys.argv) > 2:
    for line in open(sys.argv[2], encoding='utf-8').read().splitlines()[1:]:
        x = line.split('\t')
        audit[x[0]] = x[1]
ids = sorted(by['restart'])
ORDER = ['restart', 'calc', 'code', 'cut', 'line']
TYPE = dict(restart='retry', calc='calc', code='code', cut='line', line='line')
conds = [c for c in ORDER if by[c]]
rate = lambda c, i, part=slice(None): np.mean(by[c][i]['correct'][part]) if i in by[c] else None
rng = np.random.default_rng(0)
print(f'items {len(ids)}; attempts per correction {len(by["restart"][ids[0]]["correct"])}; corrections {conds}')

print('\n## fix rate per correction (mean over items where it applies)')
print(f"{'correction':10s} {'n':>5s} {'fix rate':>8s} {'any of N':>8s} {'>= half':>8s} {'restart same items':>18s} {'diff [95% CI]':>22s} {'tokens':>7s}")
for c in conds:
    s = [i for i in ids if i in by[c]]
    a = np.array([rate(c, i) for i in s])
    b = np.array([rate('restart', i) for i in s])
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{c:10s} {len(s):5d} {a.mean():8.1%} {np.mean([any(by[c][i]['correct']) for i in s]):8.1%} {(a >= .5).mean():8.1%} "
          f"{b.mean():18.1%} {d.mean():+8.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] "
          f"{np.mean([np.mean(by[c][i]['new_tokens']) for i in s]):7.0f}")


def typing(thr, part=slice(None)):
    out = {}
    for i in ids:
        out[i] = 'stuck'
        for c in conds:
            r = rate(c, i, part)
            if r is not None and r >= thr:
                out[i] = TYPE[c]
                break
    return out


TYPES = ['retry', 'calc', 'code', 'line', 'stuck']
for thr in (0.5, 0.25, 0.125):
    t = typing(thr)
    c = collections.Counter(t.values())
    print(f'\n## types, works = fix rate >= {thr:g}: ' + '  '.join(f'{k} {c[k]} ({c[k] / len(ids):.1%})' for k in TYPES))
t = typing(0.5)
a, b = typing(0.5, slice(0, 4)), typing(0.5, slice(4, 8))
agree = np.mean([a[i] == b[i] for i in ids])
pa = collections.Counter(a.values())
pb = collections.Counter(b.values())
pe = sum(pa[k] * pb[k] for k in TYPES) / len(ids) ** 2
print(f'\n## noise: type from attempts 1-4 vs 5-8 (threshold 0.5): agree {agree:.1%}, kappa {(agree - pe) / (1 - pe):.2f}')
print('  ' + 'first half / second half'.ljust(26) + ' '.join(f'{k:>6s}' for k in TYPES))
for k in TYPES:
    print(f'  {k:26s}' + ' '.join(f'{sum(a[i] == k and b[i] == j for i in ids):6d}' for j in TYPES))

print('\n## which corrections work on the same item (fix rate >= 0.5), items where all apply')
for c in conds[1:]:
    s = [i for i in ids if i in by[c]]
    w = [(rate('restart', i) >= .5, rate(c, i) >= .5) for i in s]
    n = collections.Counter(w)
    print(f'  restart vs {c:5s} (n={len(s)}): both {n[(True, True)]}  only restart {n[(True, False)]}  only {c} {n[(False, True)]}  neither {n[(False, False)]}')

cat = {i: by['restart'][i]['cat'] for i in ids}
print('\n## types by rule category' + (' and by audited class' if audit else ''))
groups = [('rule ' + k, [i for i in ids if cat[i] == k]) for k in ('SETUP', 'NOEQ', 'TARGET', 'OTHER', 'ARITH')]
if audit:
    groups += [('audit ' + k, [i for i in ids if audit.get(i) == k]) for k in ('CALC', 'TARGET', 'SETUP', 'OTHER')]
print(f"  {'group':14s} {'n':>5s} " + ' '.join(f'{k:>6s}' for k in TYPES) + '  ' + ' '.join(f'{c:>8s}' for c in conds))
for name, s in groups + [('all', ids)]:
    if not s:
        continue
    print(f'  {name:14s} {len(s):5d} ' + ' '.join(f"{np.mean([t[i] == k for i in s]):6.1%}" for k in TYPES) + '  ' +
          ' '.join(f"{np.mean([rate(c, i) for i in s if i in by[c]]) if any(i in by[c] for i in s) else float('nan'):8.1%}" for c in conds))

code_none = [p is None for i in ids for p in by['code'][i]['preds']] if by['code'] else []
if code_none:
    print(f'\ncode: programs without a number in the output (error, timeout, nothing printed) {np.mean(code_none):.1%}')
