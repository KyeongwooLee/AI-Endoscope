"""Why does backtracking lose to restarting? Does the model repeat the same wrong step when resampled from the cut?

For the *_plain / *_typed (re-read) attempts of correct_exp.py: the first equation of the continuation is the redone
step. 'repeat' = its result equals the original wrong step's result. Also estimates rejection sampling ('resample
until the redone step differs', as in backtracking with a known mistake location): fix rate among attempts whose
redone step differs (falls back to all attempts of a solution when none differs).
usage: python -I analyze_repeat.py <correct_dir> <gen_dir>
"""
import collections, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations, key

R = [json.loads(l) for f in sorted(glob.glob(f'{sys.argv[1]}/correct_*.jsonl')) for l in open(f)]
gen = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in glob.glob(f'{sys.argv[2]}/test_*.jsonl') for r in map(json.loads, open(f))}
restart = {r['id']: np.mean(r['correct']) for r in R if r['cond'] == 'restart'}
rng = np.random.default_rng(0)

print(f"  {'condition':12s} {'n':>5s} {'repeat':>7s} {'fix':>6s} {'fix|repeat':>10s} {'fix|differ':>10s} "
      f"{'reject-resample':>15s} {'restart':>8s}")
for cond in ('pred_plain', 'true_plain', 'pred_typed', 'true_typed'):
    for sub, cond_k in (('k>=1', lambda r: r['true_k'] >= 1), ('all', lambda r: True)):
        rows = [r for r in R if r['cond'] == cond and r.get('fix') != 'calc' and cond_k(r) and r['step'] >= 1]
        rep_all, fix_rep, fix_dif, rej, base = [], [], [], [], []
        for r in rows:
            orig = key(equations(gen[r['id']]['text'])[r['step']][2])
            rep = []
            for t in r['texts']:
                e = equations(t)
                rep.append(bool(e) and key(e[0][2]) == orig)
            rep_all += rep
            fix_rep += [c for c, x in zip(r['correct'], rep) if x]
            fix_dif += [c for c, x in zip(r['correct'], rep) if not x]
            dif = [c for c, x in zip(r['correct'], rep) if not x]
            rej.append(np.mean(dif) if dif else np.mean(r['correct']))
            base.append(restart[r['id']])
        print(f'  {cond + " " + sub:12s} {len(rows):5d} {np.mean(rep_all):7.1%} '
              f'{np.mean([np.mean(r["correct"]) for r in rows]):6.1%} {np.mean(fix_rep):10.1%} {np.mean(fix_dif):10.1%} '
              f'{np.mean(rej):15.1%} {np.mean(base):8.1%}')
print('  (rows: cut after at least one kept equation, step >= 1; calculator attempts excluded)')

D = [json.loads(l) for f in sorted(glob.glob(f'{sys.argv[1]}/diff_*.jsonl')) for l in open(f)]
if D:
    print(f'\n## resample until the redone step differs (16 samples per cut; 4 accepted attempts = first 4 samples whose '
          f'redone step differs, filled with repeating ones if fewer)')
    print(f"  {'condition':16s} {'n':>5s} {'fix':>6s} {'restart':>8s} {'diff':>23s} {'samples used':>12s} "
          f"{'tokens/attempt':>14s}")
    restart_tok = {r['id']: np.mean(r['new_tokens']) for r in R if r['cond'] == 'restart'}
    for cond in ('pred_plain', 'true_plain'):
        for sub, cond_k in (('all', lambda r: True), ('k>=1', lambda r: r['true_k'] >= 1)):
            rows = [r for r in D if r['cond'] == cond and cond_k(r)]
            fx, used, toks = [], [], []
            for r in rows:
                orig = key(equations(gen[r['id']]['text'])[r['step']][2])
                dif = [not (bool(e) and key(e[0][2]) == orig) for e in map(equations, r['texts'])]
                acc = [j for j, x in enumerate(dif) if x][:4]
                acc += [j for j, x in enumerate(dif) if not x][:4 - len(acc)]
                last = max(acc) + 1 if sum(dif) >= 4 else len(dif)
                fx.append(np.mean([r['correct'][j] for j in acc]))
                used.append(last)
                toks.append(sum(r['new_tokens'][:last]) / 4)
            fx = np.array(fx)
            d = fx - np.array([restart[r['id']] for r in rows])
            prob = np.array(['-'.join(r['id'].split('-')[:2]) for r in rows])
            keys = np.unique(prob)
            gi = {k: np.where(prob == k)[0] for k in keys}
            bs = [d[np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(2000)]
            print(f'  {cond + " " + sub:16s} {len(rows):5d} {fx.mean():6.1%} {fx.mean() - d.mean():8.1%} '
                  f'{d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] '
                  f'{np.mean(used):12.1f} {np.mean(toks):9.0f} (restart '
                  f'{np.mean([restart_tok[r["id"]] for r in rows]):.0f})')

ar = [r for r in R if r['cond'] == 'true_typed' and r['cat'] == 'ARITH']
d = np.array([np.mean(r['correct']) - restart[r['id']] for r in ar])
prob = np.array(['-'.join(r['id'].split('-')[:2]) for r in ar])
keys = np.unique(prob)
gi = {k: np.where(prob == k)[0] for k in keys}
bs = [d[np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(2000)]
print(f'\n  ARITH errors (n={len(ar)}): true_typed (calculator at the true step) - restart: {d.mean():+.3f} '
      f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]')
