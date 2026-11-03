"""Summarize the correction experiment (correct_exp.py): fix rate per condition on the model's own wrong answers.

fix rate = share of the N attempts that end with the gold answer (per solution, then averaged); any = at least one of
the N attempts is right; tokens = newly generated tokens per attempt. Paired bootstrap over problems.
usage: python -I analyze_correct.py <correct_dir> [audit loc_labels.tsv]
"""
import collections, glob, json, sys
import numpy as np

R = [json.loads(l) for f in sorted(glob.glob(f'{sys.argv[1]}/correct_*.jsonl')) for l in open(f)]
audit = {}
if len(sys.argv) > 2:
    for line in open(sys.argv[2], encoding='utf-8').read().splitlines()[1:]:
        x = line.split('\t')
        audit[x[0]] = x[7]
by = collections.defaultdict(dict)  # cond -> id -> record
for r in R:
    by[r['cond']][r['id']] = r
fix = lambda r: np.mean(r['correct'])
rng = np.random.default_rng(0)


def table(title, conds, ids, subsets):
    print(f'\n## {title} (solutions={len(ids)})')
    print(f"  {'condition':14s} {'fix rate':>8s} {'any of N':>8s} {'tokens':>7s}  " +
          '  '.join(f'{s:>14s}' for s in subsets))
    for c in conds:
        rs = [by[c][i] for i in ids]
        sub = '  '.join(f'{np.mean([fix(by[c][i]) for i in m]) if m else float("nan"):14.1%}'
                        for m in subsets.values())
        print(f"  {c:14s} {np.mean([fix(r) for r in rs]):8.1%} {np.mean([any(r['correct']) for r in rs]):8.1%} "
              f"{np.mean([np.mean(r['new_tokens']) for r in rs]):7.0f}  {sub}")
    print('  subset sizes: ' + '  '.join(f'{s}={len(m)}' for s, m in subsets.items()))


def boot(ids, a, b):
    prob = np.array(['-'.join(i.split('-')[:2]) for i in ids])
    da = np.array([fix(by[a][i]) - fix(by[b][i]) for i in ids])
    keys = np.unique(prob)
    gi = {k: np.where(prob == k)[0] for k in keys}
    d = [da[np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(2000)]
    print(f'  {a} - {b}: {da.mean():+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]')


loc = sorted(i for i in by['pred_plain'] if all(i in by[c] for c in by if c != 'target_cue'))
L = {i: by['pred_plain'][i] for i in loc}
subsets = {'true k=0': [i for i in loc if L[i]['true_k'] == 0],
           'true k>=1': [i for i in loc if L[i]['true_k'] >= 1],
           'pred = rule': [i for i in loc if L[i]['step'] == L[i]['true_k']],
           'pred != rule': [i for i in loc if L[i]['step'] != L[i]['true_k']],
           'ARITH': [i for i in loc if L[i]['cat'] == 'ARITH']}
conds = ['restart', 'refine', 'pred_plain', 'pred_typed', 'true_plain', 'true_typed']
table('located errors (test split)', conds, loc, subsets)
for c in ('pred_typed', 'true_typed'):
    f = collections.Counter(by[c][i].get('fix') for i in loc)
    rate = {k: np.mean([fix(by[c][i]) for i in loc if by[c][i].get('fix') == k]) for k in f}
    print(f'  {c}: ' + '  '.join(f'{k} n={f[k]} fix {rate[k]:.1%}' for k in f))
print('\n  paired bootstrap over problems (difference in fix rate):')
for a, b in [('refine', 'restart'), ('pred_plain', 'restart'), ('pred_typed', 'restart'), ('pred_typed', 'pred_plain'),
             ('true_plain', 'restart'), ('true_typed', 'restart'), ('true_typed', 'true_plain')]:
    boot(loc, a, b)
k1 = subsets['true k>=1']
print('  only errors at the 2nd equation or later (true k>=1):')
for a, b in [('pred_typed', 'restart'), ('true_typed', 'restart')]:
    boot(k1, a, b)

if audit:
    au = [i for i in loc if i in audit]
    print(f'\n## by audit category (audited solutions in this set: {len(au)})')
    for cat in ('agree', 'agree_prose', 'rule_late', 'rule_early'):
        ids = [i for i in au if audit[i] == cat]
        if ids:
            print(f'  {cat:12s} n={len(ids):3d}  ' + '  '.join(
                f'{c} {np.mean([fix(by[c][i]) for i in ids]):.0%}' for c in conds))

tg = sorted(by['target_cue'])
table('TARGET errors: final answer is an intermediate (test split)', ['restart', 'refine', 'target_cue'], tg, {})
boot(tg, 'target_cue', 'restart')
boot(tg, 'refine', 'restart')
