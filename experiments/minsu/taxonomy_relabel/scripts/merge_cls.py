"""Merge the class + location audit of the test-split wrong answers and compare with the rule labels.

Per item the labelers chose (without seeing the rule): `class` (CALC / TARGET / SETUP / OTHER, by the first error),
`num` (line of the first wrong value), `thought` (line where the misunderstanding first shows in words, <= num).
Shared items (in the first chunks) give the agreement; for them the final label is the majority (class first, then
the most common lines among the labelers that chose that class).
The audited first wrong line is also mapped to an equation index (audit_k = first checkable equation on or after it)
so it can be compared with the rule's k and the location probe.
usage: python -I merge_cls.py <batches_dir> <out_dir>
writes <out_dir>/cls_labels.tsv and prints the summary
"""
import collections, itertools, json, re, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
K = len(list(B.glob('chunk_*_ids.txt')))
shared = set(json.loads((B / 'shared_ids.json').read_text()))
rule = json.loads((B / 'rule_labels.json').read_text())
CLASSES = ['CALC', 'TARGET', 'SETUP', 'OTHER']
L = lambda s: int(s[1:]) if s != '-' else None

votes = collections.defaultdict(list)  # id -> [(class, num, thought, conf, reason)]
for k in range(K):
    want = set((B / f'chunk_{k}_ids.txt').read_text().split())
    got = {}
    for part in sorted(OUT.glob(f'chunk_{k}_part*.tsv')):
        for line in part.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('id\t'):
                continue
            i, cls, num, th, conf, reason = (line.split('\t') + [''] * 5)[:6]
            num, th = num.strip(), th.strip()
            assert cls in CLASSES and conf in ('h', 'm'), (k, line)
            assert re.fullmatch(r'L\d+|-', num) and re.fullmatch(r'L\d+|-', th), (k, line)
            assert num != '-' or cls == 'OTHER', (k, 'num - only for OTHER', line)
            n = rule[i]['n_lines']
            assert all(x == '-' or 1 <= L(x) <= n for x in (num, th)), (k, 'line out of range', line)
            if num != '-' and th != '-' and L(th) > L(num):
                th = num  # rubric: thought <= num
            assert i not in got, (k, 'duplicate', i)
            got[i] = (cls, num, th, conf, reason)
    assert set(got) == want, (k, 'missing', len(want - set(got)), 'extra', len(set(got) - want))
    for i, v in got.items():
        votes[i].append(v)


def fleiss(items):
    n, N = len(items[0]), len(items)
    p = collections.Counter(c for it in items for c in it)
    pe = sum((v / (N * n)) ** 2 for v in p.values())
    pbar = sum((sum(v * v for v in collections.Counter(it).values()) - n) / (n * (n - 1)) for it in items) / N
    return (pbar - pe) / (1 - pe)


pair = lambda X: sum(a == b for it in X for a, b in itertools.combinations(it, 2)) / sum(
    1 for it in X for _ in itertools.combinations(it, 2))
S = sorted(shared)
nl = len(votes[S[0]])
print(f'## agreement on {len(S)} shared items ({nl} labelers each)')
for name, f in [('class', lambda v: v[0]), ('first wrong line (num)', lambda v: v[1]),
                ('thought line', lambda v: v[2]), ('thought earlier than num (y/n)', lambda v: v[2] != v[1])]:
    X = [[f(v) for v in votes[i]] for i in S]
    print(f'  {name:32s} pairwise {pair(X):.2f}  Fleiss {fleiss(X):.2f}  all agree {sum(len(set(x)) == 1 for x in X)}/{len(S)}')
X = [[v[1] for v in votes[i] if v[0] == 'SETUP'] for i in S]
X = [x for x in X if len(x) == nl]
if X:
    print(f'  num line, items all labelers call SETUP ({len(X)}): pairwise {pair(X):.2f}, all agree '
          f'{sum(len(set(x)) == 1 for x in X)}/{len(X)}')

final = {}
for i, vs in votes.items():
    cls = collections.Counter(v[0] for v in vs).most_common(1)[0][0]
    same = [v for v in vs if v[0] == cls]
    num = collections.Counter(v[1] for v in same).most_common(1)[0][0]
    th = collections.Counter(v[2] for v in same if v[1] == num).most_common(1)[0][0]
    conf = 'm' if len({v[0] for v in vs}) > 1 or len({v[1] for v in vs}) > 1 else vs[0][3]
    final[i] = (cls, num, th, conf, next(v[4] for v in same if v[1] == num))


def audit_k(i):
    """equation index of the audited first wrong line: the first checkable equation on or after it"""
    num = final[i][1]
    if num == '-':
        return None
    after = [j for j, l in enumerate(rule[i]['eq_lines']) if l >= L(num)]
    return after[0] if after else None


ids = sorted(final)
print(f'\n## rule class vs audit class ({len(ids)} items)')
print('  ' + 'rule / audit'.ljust(14) + ' ' + ' '.join(f'{c:>7s}' for c in CLASSES) + '   n   agree')
for rc in ['CALC', 'TARGET', 'SETUP']:
    row = [i for i in ids if rule[i]['cls'] == rc]
    c = collections.Counter(final[i][0] for i in row)
    print(f"  {rc:14s} " + ' '.join(f'{c[a]:7d}' for a in CLASSES) + f' {len(row):4d}  {c[rc] / len(row):5.1%}')
agree = sum(final[i][0] == rule[i]['cls'] for i in ids)
print(f'  overall agreement {agree / len(ids):.1%}; audit class shares: ' + '  '.join(
    f'{c} {sum(final[i][0] == c for i in ids) / len(ids):.1%}' for c in CLASSES))
conf_m = [i for i in ids if final[i][3] == 'm']
print(f'  conf=m {len(conf_m) / len(ids):.1%}')

st = [i for i in ids if rule[i]['cls'] == 'SETUP' and final[i][0] == 'SETUP']
print(f'\n## location, rule SETUP & audit SETUP ({len(st)} items)')
rel = collections.Counter('same' if L(final[i][1]) == rule[i]['rule_line'] else
                          'audit earlier' if L(final[i][1]) < rule[i]['rule_line'] else 'audit later' for i in st)
print('  first wrong line vs rule line: ' + '  '.join(f'{k} {rel[k] / len(st):.1%}' for k in
                                                       ('same', 'audit earlier', 'audit later')))
noeq = sum(L(final[i][1]) not in rule[i]['eq_lines'] for i in st)
print(f'  first wrong line has no checkable equation (wrong number in plain text): {noeq / len(st):.1%}')
first = lambda i: L(final[i][1]) <= rule[i]['eq_lines'][0]
print(f"  'wrong from the first equation': rule (k=0) {sum(rule[i]['k'] == 0 for i in st) / len(st):.1%}, "
      f"audit {sum(map(first, st)) / len(st):.1%}, agree {sum(first(i) == (rule[i]['k'] == 0) for i in st) / len(st):.1%}")
th = sum(final[i][2] != final[i][1] for i in st)
print(f'  thought earlier than the first wrong number: {th / len(st):.1%}')

with open(OUT / 'cls_labels.tsv', 'w', encoding='utf-8') as f:
    f.write('id\tclass\tnum\tthought\tconf\treason\trule_cls\trule_cat\trule_k\trule_line\taudit_k\tn_votes\n')
    for i in ids:
        cls, num, th, conf, reason = final[i]
        r = rule[i]
        f.write(f"{i}\t{cls}\t{num}\t{th}\t{conf}\t{reason}\t{r['cls']}\t{r['cat']}\t{r['k']}\t{r['rule_line']}\t"
                f"{audit_k(i)}\t{len(votes[i])}\n")
