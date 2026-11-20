"""What is inside each cluster from cluster_errors.py? One block per representation:k.

Per cluster (train): size, rule type shares, length, confidence, writing style, words that mark the cluster in the
question and in the response (log-odds, document frequency), share of correct solutions assigned to it (control).
Test wrong answers: AI-audited class shares and correction fix rates, paired bootstrap over problems for the difference
to restarting. Near-centroid + random examples go to <out_dir>/examples_<rep>_<k>.txt.
usage: python -I cluster_detail.py <out_dir> <correct_dir> <rep:k> [<rep:k> ...]
"""
import collections, glob, json, pickle, re, sys
import numpy as np

OUT, CORR = sys.argv[1:3]
T = pickle.load(open(f'{OUT}/rows.pkl', 'rb'))
TYPES = ['SETUP', 'ARITH', 'TARGET', 'OTHER', 'NEWNUM']
CONDS = ['restart', 'refine', 'true_plain', 'true_typed', 'pred_plain']
by = collections.defaultdict(dict)
for f in sorted(glob.glob(f'{CORR}/correct_*.jsonl')):
    for r in map(json.loads, open(f)):
        by[r['cond']][r['id']] = np.mean(r['correct'])
sid = lambda k: f'{k[0]}-{k[1]}-{k[2]}'
tok = re.compile(r'[a-z]+|[+\-*/=]')


def words(idx, field, m):
    docs = [set(tok.findall(T[field][i].lower())) for i in idx]
    a, b = collections.Counter(), collections.Counter()
    for d, x in zip(docs, m):
        (a if x else b).update(d)
    na, nb = m.sum(), (~m).sum()
    lo = {w: np.log((a[w] + .5) / (na - a[w] + .5)) - np.log((b[w] + .5) / (nb - b[w] + .5))
          for w in a if a[w] + b[w] >= 20 and a[w] >= 10}
    return ' '.join(sorted(lo, key=lo.get, reverse=True)[:10])


def boot(ids, a, b, rng):
    ids = [i for i in ids if i in by[a] and i in by[b]]
    if len(ids) < 20:
        return f'n={len(ids)}'
    d = np.array([by[a][i] - by[b][i] for i in ids])
    p = np.array(['-'.join(i.split('-')[:2]) for i in ids])
    keys = np.unique(p)
    gi = {k: np.where(p == k)[0] for k in keys}
    s = [d[np.concatenate([gi[k] for k in rng.choice(keys, len(keys))])].mean() for _ in range(1000)]
    return f'{d.mean():+.3f} [{np.percentile(s, 2.5):+.3f},{np.percentile(s, 97.5):+.3f}] n={len(ids)}'


for spec in sys.argv[3:]:
    rep, k = spec.split(':')
    d = np.load(f'{OUT}/rep_{rep}.npz')
    idx, it, ip = d['idx'], d['idx_t'], d['idx_p']
    lab, lt, lp, Z, c = d[f'lab_{k}'], d[f'lab_t_{k}'], d[f'lab_p_{k}'], d['Z'], d[f'c_{k}']
    print(f'\n===== {rep} k={k}: train {len(idx)}, test {len(it)} =====')
    print(f"{'c':>2s} {'n':>5s} {'share':>6s} " + ' '.join(f'{t:>6s}' for t in TYPES) +
          f" {'len':>5s} {'n_eq':>5s} {'nll':>5s} {'step':>5s} {'latex':>5s} {'bold':>5s} {'cor%':>5s}  first words")
    for j in range(int(k)):
        m = lab == j
        i = idx[m]
        first = collections.Counter(T['first'][i]).most_common(3)
        print(f'{j:2d} {m.sum():5d} {m.mean():6.1%} ' + ' '.join(f"{(T['cat'][i] == t).mean():6.1%}" for t in TYPES) +
              f" {np.exp(T['loglen'][i].mean()):5.0f} {T['n_eq'][i].mean():5.1f} {T['nll'][i].mean():5.2f} "
              f"{T['step'][i].mean():5.0%} {T['latex'][i].mean():5.0%} {T['bold'][i].mean():5.0%} "
              f"{(lp == j).mean() if len(lp) else float('nan'):5.0%}  " +
              ' '.join(f'{w}:{n / m.sum():.0%}' for w, n in first))
    print(f"   all {len(idx)}        " + ' '.join(f"{(T['cat'][idx] == t).mean():6.1%}" for t in TYPES))
    print('words:')
    for j in range(int(k)):
        m = lab == j
        print(f'  {j} question: {words(idx, "question", m)}\n    response: {words(idx, "text", m)}')

    print('test: audited class shares and correction (fix rate per attempt)')
    print(f"{'c':>2s} {'n':>5s} {'audited':>7s} {'CALC':>6s} {'TARGET':>6s} {'SETUP':>6s} {'OTHER':>6s} " +
          ' '.join(f'{x:>10s}' for x in CONDS))
    aud = T['aud'][it]
    ids = np.array([sid(T['key'][i]) for i in it])
    for j in list(range(int(k))) + [-1]:
        m = lt == j if j >= 0 else np.ones(len(lt), bool)
        a = aud[m][aud[m] != '']
        print(f"{'all' if j < 0 else j:>2} {m.sum():5d} {len(a):7d} " + ' '.join(f"{(a == x).mean() if len(a) else 0:6.1%}" for x in
              ('CALC', 'TARGET', 'SETUP', 'OTHER')) + ' ' +
              ' '.join(f"{np.mean([by[x][i] for i in ids[m] if i in by[x]]) if any(i in by[x] for i in ids[m]) else float('nan'):10.1%}"
                       for x in CONDS))
    rng = np.random.default_rng(0)
    print('difference to restart, by cluster (mean [95% CI] n):')
    for j in list(range(int(k))) + [-1]:
        m = lt == j if j >= 0 else np.ones(len(lt), bool)
        print(f"  {'all' if j < 0 else j:>3}  true_typed {boot(ids[m].tolist(), 'true_typed', 'restart', rng)}   "
              f"true_plain {boot(ids[m].tolist(), 'true_plain', 'restart', rng)}")

    with open(f'{OUT}/examples_{rep}_{k}.txt', 'w', encoding='utf-8') as out:
        rng = np.random.default_rng(1)
        for j in range(int(k)):
            m = np.where(lab == j)[0]
            near = m[np.argsort(((Z[m] - c[j]) ** 2).sum(1))]
            seen, pick = set(), []
            for x in list(near) + list(rng.permutation(m)):
                p = T['key'][idx[x]][:2]
                if p not in seen and (len(pick) < 5 or x not in near[:50]):
                    seen.add(p)
                    pick.append(x)
                if len(pick) == 10:
                    break
            out.write(f'\n######## cluster {j} (n={len(m)}): first 5 near centroid, last 5 random\n')
            for x in pick:
                i = idx[x]
                key = T['key'][i]
                out.write(f"\n--- {sid(key)} rule={T['cat'][i]} first_wrong={T['line'][key]!r}\nQ: {T['question'][i][:300]}\n"
                          f"A: {T['text'][i][:700]}\n")
