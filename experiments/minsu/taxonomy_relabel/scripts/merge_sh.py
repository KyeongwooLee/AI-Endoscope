"""Merge the re-labeling of Sunghun's 7-type error labels: how ambiguous is his codebook?

Every labeler labeled the same items (blind to his label). Reports agreement among the labelers (pairwise, Fleiss,
all agree), self-reported borderline rate (conf=m), the most confused type pairs, and each labeler / the majority vs
his original label.
usage: python -I merge_sh.py <batch_dir> <out_dir>
writes <out_dir>/sh_labels.tsv and prints the summary
"""
import collections, itertools, json, sys
from pathlib import Path

B, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TYPES = ['modeling_relation', 'arithmetic_execution', 'unit_conversion', 'reasoning_state', 'answer_reporting',
         'undetermined', 'none_observed']
ids = (B / 'ids.txt').read_text().split()
sh = json.loads((B / 'sunghun_labels.json').read_text())

lab = {}  # labeler -> id -> (type, conf, reason)
for n in sorted({p.name.split('_part')[0] for p in OUT.glob('labeler_*_part*.tsv')}):
    got = {}
    for part in sorted(OUT.glob(f'{n}_part*.tsv')):
        for line in part.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.startswith('id\t'):
                continue
            i, t, conf, reason = (line.split('\t') + [''] * 3)[:4]
            assert t in TYPES and conf in ('h', 'm'), (n, line)
            assert i not in got, (n, 'duplicate', i)
            got[i] = (t, conf, reason)
    assert set(got) == set(ids), (n, 'missing', len(set(ids) - set(got)), 'extra', len(set(got) - set(ids)))
    lab[n] = got
names = sorted(lab)


def fleiss(items):
    n, N = len(items[0]), len(items)
    p = collections.Counter(c for it in items for c in it)
    pe = sum((v / (N * n)) ** 2 for v in p.values())
    pbar = sum((sum(v * v for v in collections.Counter(it).values()) - n) / (n * (n - 1)) for it in items) / N
    return (pbar - pe) / (1 - pe)


def cohen(a, b):
    po = sum(x == y for x, y in zip(a, b)) / len(a)
    ca, cb = collections.Counter(a), collections.Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / len(a) ** 2
    return (po - pe) / (1 - pe)


pair = lambda X: sum(a == b for it in X for a, b in itertools.combinations(it, 2)) / sum(
    1 for it in X for _ in itertools.combinations(it, 2))
X = [[lab[n][i][0] for n in names] for i in ids]
print(f'## {len(ids)} items, {len(names)} labelers (all label every item)')
print(f'  type: pairwise {pair(X):.2f}  Fleiss {fleiss(X):.2f}  all agree {sum(len(set(x)) == 1 for x in X)}/{len(ids)}'
      f' ({sum(len(set(x)) == 1 for x in X) / len(ids):.0%})')
m = [lab[n][i][1] == 'm' for n in names for i in ids]
print(f'  self-reported borderline (conf=m): {sum(m) / len(m):.1%} of labels; per labeler ' +
      ' '.join(f'{sum(lab[n][i][1] == "m" for i in ids)}' for n in names))
print(f'  items where >=1 labeler says m or labelers disagree: '
      f'{sum(len(set(x)) > 1 or any(lab[n][i][1] == "m" for n in names) for x, i in zip(X, ids)) / len(ids):.0%}')

print('\n## type counts per labeler (and Sunghun)')
print('  ' + ' ' * 21 + ' '.join(f'{n[-1]:>4s}' for n in names) + '  Sung')
for t in TYPES:
    print(f'  {t:21s}' + ' '.join(f'{sum(lab[n][i][0] == t for i in ids):4d}' for n in names) +
          f'  {sum(sh[i]["type"] == t for i in ids):4d}')

conf_pairs = collections.Counter()
for x in X:
    for a, b in itertools.combinations(x, 2):
        if a != b:
            conf_pairs[tuple(sorted((a, b)))] += 1
alts = collections.Counter()
for n in names:
    for i in ids:
        t, c, r = lab[n][i]
        if c == 'm' and 'alt=' in r:
            alts[(t, r.split('alt=')[1].split()[0].strip('.,;)'))] += 1
print(f'\n## most confused pairs (disagreeing labeler pairs, out of {sum(conf_pairs.values())})')
for (a, b), c in conf_pairs.most_common(6):
    print(f'  {a} vs {b}: {c}')
print('## conf=m alternatives (chosen -> alt)')
for (a, b), c in alts.most_common(6):
    print(f'  {a} -> {b}: {c}')

maj = {}
for i, x in zip(ids, X):
    (t, c), = collections.Counter(x).most_common(1)
    maj[i] = t if c > len(names) / 2 else None  # None: no majority
print(f'\n## vs Sunghun\'s original label')
for n in names:
    a, b = [lab[n][i][0] for i in ids], [sh[i]['type'] for i in ids]
    print(f'  {n}: agree {sum(x == y for x, y in zip(a, b)) / len(ids):.0%}, Cohen kappa {cohen(a, b):.2f}')
mi = [i for i in ids if maj[i]]
print(f'  majority (> half, {len(mi)} items have one): agree with Sunghun {sum(maj[i] == sh[i]["type"] for i in mi) / len(mi):.0%}')
cm = collections.Counter((sh[i]['type'], maj[i] or 'no majority') for i in ids)
print('  Sunghun -> majority (only changed ones):')
for (a, b), c in cm.most_common():
    if a != b:
        print(f'    {a} -> {b}: {c}')

with open(OUT / 'sh_labels.tsv', 'w', encoding='utf-8') as f:
    f.write('id\tsunghun\tmajority\t' + '\t'.join(names) + '\tn_conf_m\n')
    for i, x in zip(ids, X):
        f.write(f"{i}\t{sh[i]['type']}\t{maj[i] or '-'}\t" + '\t'.join(x) +
                f"\t{sum(lab[n][i][1] == 'm' for n in names)}\n")
