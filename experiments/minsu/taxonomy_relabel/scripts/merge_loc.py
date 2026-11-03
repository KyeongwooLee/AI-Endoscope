"""Merge the location audit (5 AI labelers, 30 shared items) and compare with the rule's first wrong equation.

Per item the labelers chose `first` (first marked equation with a wrong value, or none) without seeing the rule's
pick, plus `words` (a wrong number already appears in text outside the equations) and `prose` (the misunderstanding is
already stated in words before the first wrong number). Categories against the rule (E{k+1}):
  agree        first == rule and words = n          (agree_prose: same, but prose = y -> the thought went wrong earlier)
  rule_late    words = y, or first earlier than rule (the rule missed an earlier error)
  rule_early   first later than rule, or none        (the rule flagged a fine step)
Also re-scores the location probe (preds_loc2.json) against the audited locations.
usage: python -I merge_loc.py <batches_dir> <out_dir> <preds_loc2.json>
writes <out_dir>/loc_labels.tsv and prints the summary
"""
import collections, itertools, json, re, sys
from pathlib import Path

B, OUT, PREDS = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
K = len(list(B.glob('chunk_*_ids.txt')))
shared = set(json.loads((B / 'shared_ids.json').read_text()))
rule = json.loads((B / 'rule_labels.json').read_text())
P = json.load(open(PREDS))

votes = collections.defaultdict(list)  # id -> [(first, words, prose, conf, reason)]
for k in range(K):
    want = set((B / f'chunk_{k}_ids.txt').read_text().split())
    got = {}
    for part in sorted(OUT.glob(f'chunk_{k}_part*.tsv')):
        for line in part.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('id\t'):
                continue
            i, first, words, prose, conf, reason = (line.split('\t') + [''] * 5)[:6]
            first = first.strip()
            assert (first == 'none' or re.fullmatch(r'E\d+', first)) and words in 'yn' and prose in 'yn' \
                and conf in ('h', 'm'), (k, line)
            assert first == 'none' or int(first[1:]) <= rule[i]['n_eq'], (k, 'E out of range', line)
            assert i not in got, (k, 'duplicate', i)
            got[i] = (first, words, prose, conf, reason)
    assert set(got) == want, (k, 'missing', len(want - set(got)), 'extra', len(set(got) - want))
    for i, v in got.items():
        votes[i].append(v)


def fleiss(items):
    n, N = len(items[0]), len(items)
    cats = {c for it in items for c in it}
    p = collections.Counter(c for it in items for c in it)
    pe = sum((p[c] / (N * n)) ** 2 for c in cats)
    pbar = sum((sum(v * v for v in collections.Counter(it).values()) - n) / (n * (n - 1)) for it in items) / N
    return (pbar - pe) / (1 - pe)


def category(i, first, words, prose):
    r = rule[i]['k']
    if words == 'y' or (first != 'none' and int(first[1:]) - 1 < r):
        return 'rule_late'
    if first == 'none' or int(first[1:]) - 1 > r:
        return 'rule_early'
    return 'agree_prose' if prose == 'y' else 'agree'


S = sorted(shared)
F = [[v[0] for v in votes[i]] for i in S]
C = [[category(i, *v[:3]) for v in votes[i]] for i in S]
pair = lambda X: sum(a == b for it in X for a, b in itertools.combinations(it, 2)) / sum(
    1 for it in X for _ in itertools.combinations(it, 2))
print(f'## agreement on {len(S)} shared items ({K} labelers each)')
print(f'  first wrong equation: pairwise {pair(F):.2f}, Fleiss kappa {fleiss(F):.2f}, '
      f'all agree {sum(len(set(it)) == 1 for it in F)}/{len(S)}')
agree_rule = [[c.startswith('agree') for c in it] for it in C]
print(f'  agrees with the rule (y/n): pairwise {pair(agree_rule):.2f}, '
      f'all agree {sum(len(set(it)) == 1 for it in agree_rule)}/{len(S)}')
print(f"  words: pairwise {pair([[v[1] for v in votes[i]] for i in S]):.2f}   "
      f"prose: pairwise {pair([[v[2] for v in votes[i]] for i in S]):.2f}")

final = {}
for i, vs in votes.items():
    c = collections.Counter(v[0] for v in vs).most_common()
    first = c[0][0]
    words = collections.Counter(v[1] for v in vs).most_common(1)[0][0]
    prose = collections.Counter(v[2] for v in vs).most_common(1)[0][0]
    conf = 'm' if len(c) > 1 else vs[0][3]
    final[i] = (first, words, prose, conf, next(v[4] for v in vs if v[0] == first))

CATS = ['agree', 'agree_prose', 'rule_late', 'rule_early']
cat_of = {i: category(i, *final[i][:3]) for i in final}
pool = collections.Counter(c for c in P['cat'].values())  # rule types in the whole location data
print(f'\n## rule location vs audit ({len(final)} items)')
for name, ids in [('all', list(final)), ('ARITH', [i for i in final if rule[i]['cat'] == 'ARITH']),
                  ('SETUP/NEWNUM', [i for i in final if rule[i]['cat'] != 'ARITH']),
                  ('conf=h only', [i for i in final if final[i][3] == 'h'])]:
    c = collections.Counter(cat_of[i] for i in ids)
    print(f'  {name:13s} n={len(ids):4d}  ' + '  '.join(f'{k} {c[k] / len(ids):5.1%}' for k in CATS))
est = collections.Counter()
for grp, members in [('ARITH', ['ARITH']), ('other', ['SETUP', 'NEWNUM'])]:
    ids = [i for i in final if (rule[i]['cat'] == 'ARITH') == (grp == 'ARITH')]
    n_pool = sum(pool[m] for m in members)
    c = collections.Counter(cat_of[i] for i in ids)
    for k in CATS:
        est[k] += c[k] / len(ids) * n_pool
tot = sum(pool.values())
print(f'  estimate for all {tot} located solutions (weighted by type): ' +
      '  '.join(f'{k} ~{est[k]:.0f} ({est[k] / tot:.0%})' for k in CATS))
d = collections.Counter(max(-2, min(2, int(final[i][0][1:]) - 1 - rule[i]['k'])) for i in final
                        if final[i][0] != 'none' and final[i][1] == 'n' and not cat_of[i].startswith('agree'))
print('  disagreements with an audited equation, audited - rule: ' + '  '.join(f'{k:+d}: {d[k]}' for k in sorted(d)))

print('\n## location probe re-scored on the audit (items whose audited first error is a marked equation, words=n)')
ok = [i for i in final if final[i][0] != 'none' and final[i][1] == 'n']
dis = [i for i in ok if not cat_of[i].startswith('agree')]
print(f'  items: {len(ok)} (rule disagrees on {len(dis)})')
print(f"  {'input':56s} {'vs rule':>8s} {'vs audit':>9s} {'audit k/k-1':>11s} {'disagree: =audit':>17s} {'=rule':>6s}")
for name in ['position (list)', P['best_non_hidden'], 'v1: res L20 + position (indep)', P['selected']]:
    pr = P['pred'][name]
    a = {i: int(final[i][0][1:]) - 1 for i in ok}
    acc = lambda ids, lab: sum(pr[i] == lab(i) for i in ids) / max(1, len(ids))
    print(f'  {name:56s} {acc(ok, lambda i: rule[i]["k"]):8.1%} {acc(ok, lambda i: a[i]):9.1%} '
          f'{sum(pr[i] in (a[i], a[i] - 1) for i in ok) / len(ok):11.1%} {acc(dis, lambda i: a[i]):17.1%} '
          f'{acc(dis, lambda i: rule[i]["k"]):6.1%}')

with open(OUT / 'loc_labels.tsv', 'w', encoding='utf-8') as f:
    f.write('id\tfirst\twords\tprose\tconf\treason\trule\tcategory\tcat\n')
    for i in sorted(final):
        first, words, prose, conf, reason = final[i]
        f.write(f'{i}\t{first}\t{words}\t{prose}\t{conf}\t{reason}\tE{rule[i]["k"] + 1}\t{cat_of[i]}\t{rule[i]["cat"]}\n')
