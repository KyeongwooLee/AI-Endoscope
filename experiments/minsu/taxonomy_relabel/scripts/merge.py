"""Merge the 4 chunk labelings, check agreement on the 30 shared items, and tabulate group balance vs Sunghun's types.

usage: python -I merge.py <labels.jsonl> <chunks_dir> <out_dir>
writes <out_dir>/relabeled.tsv and prints the summary
"""
import json, sys, collections, itertools
from pathlib import Path

src, chunks, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
GROUP = dict(CV='A', CO='A', HA='A', OP='B', FC='B', MS='B', TG='B', CA='C', PS='C', UC='D', UN='UN')
ORDER = ['CV', 'CO', 'HA', 'OP', 'FC', 'MS', 'TG', 'CA', 'PS', 'UC', 'UN']
SUNG = ['modeling_relation', 'arithmetic_execution', 'reasoning_state', 'unit_conversion']

rows = {r['id']: r for r in (json.loads(l) for l in src.read_text(encoding='utf-8').split('\n') if l.strip())
        if r['first_observable_error_span']}
shared = set(json.loads((chunks / 'shared_ids.json').read_text()))

votes = collections.defaultdict(list)          # id -> [(chunk, code, conf, reason)]
for k in range(4):
    want = {json.loads(l)['id'] for l in (chunks / f'chunk_{k}.jsonl').read_text(encoding='utf-8').split('\n') if l.strip()}
    got = {}
    for line in (out / f'chunk_{k}.tsv').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        i, code, conf, reason = (line.split('\t') + ['', '', ''])[:4]
        assert code in GROUP, (k, line)
        assert i not in got, (k, 'duplicate', i)
        got[i] = (k, code, conf, reason)
    assert set(got) == want, (k, 'missing', len(want - set(got)), 'extra', len(set(got) - want))
    for i, v in got.items():
        votes[i].append(v)
assert set(votes) == set(rows)

# agreement on shared items
pairs_code = pairs_group = n_pairs = 0
all4_code = all4_group = 0
for i in shared:
    codes = [v[1] for v in votes[i]]
    for a, b in itertools.combinations(codes, 2):
        n_pairs += 1; pairs_code += a == b; pairs_group += GROUP[a] == GROUP[b]
    all4_code += len(set(codes)) == 1
    all4_group += len({GROUP[c] for c in codes}) == 1
print(f'## agreement on {len(shared)} shared items (4 labelers each)')
print(f'  pairwise: code {pairs_code / n_pairs:.2f}, group {pairs_group / n_pairs:.2f}')
print(f'  all four agree: code {all4_code}/{len(shared)}, group {all4_group}/{len(shared)}')
dis = collections.Counter()
for i in shared:
    codes = sorted({v[1] for v in votes[i]})
    if len(codes) > 1:
        dis['/'.join(codes)] += 1
print('  disagreement patterns:', dict(dis.most_common(10)))

# final label: single vote, or majority for shared (ties -> 'm' conf, first most common)
final = {}
for i, vs in votes.items():
    c = collections.Counter(v[1] for v in vs).most_common()
    code = c[0][0]
    tie = len(c) > 1 and c[1][1] == c[0][1]
    conf = 'm' if tie or len(c) > 1 else vs[0][2]
    final[i] = (code, conf, next(v[3] for v in vs if v[1] == code))
with open(out / 'relabeled.tsv', 'w', encoding='utf-8') as f:
    f.write('id\tdataset\tsunghun_type\tcode\tgroup\tconf\treason\n')
    for i in sorted(final):
        code, conf, reason = final[i]
        f.write(f"{i}\t{rows[i]['dataset']}\t{rows[i]['math_error_type']}\t{code}\t{GROUP[code]}\t{conf}\t{reason}\n")

def table(ids, title):
    n = len(ids)
    c = collections.Counter(final[i][0] for i in ids)
    g = collections.Counter(GROUP[final[i][0]] for i in ids)
    print(f'\n## {title} (n={n})')
    print('  groups: ' + '  '.join(f'{k}={g[k]} ({g[k] / n:.0%})' for k in ['A', 'B', 'C', 'D', 'UN']))
    print('  codes : ' + '  '.join(f'{k}={c[k]}' for k in ORDER if c[k]))
    conf_m = sum(final[i][1] == 'm' for i in ids)
    print(f'  borderline (conf m): {conf_m} ({conf_m / n:.0%})')

table(list(final), 'all 885')
for ds in ['gsm8k', 'svamp']:
    table([i for i in final if rows[i]['dataset'] == ds], ds)

print("\n## Sunghun type (rows) x new code (cols)")
print('  ' + ' ' * 21 + ''.join(f'{k:>5}' for k in ORDER))
for t in SUNG:
    c = collections.Counter(final[i][0] for i in final if rows[i]['math_error_type'] == t)
    print(f'  {t:21s}' + ''.join(f'{c[k]:>5}' for k in ORDER) + f'   | n={sum(c.values())}')
