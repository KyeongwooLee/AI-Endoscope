"""Merge the step-2 A/B labelings (6 chunks + 30 shared items), report agreement and balance.

usage: python -I merge_ab.py <batches_dir> <out_dir>
reads <out_dir>/chunk_k.tsv, writes <out_dir>/ab_labels.tsv and <out_dir>/ab_types.jsonl (input for probe_types.py)
"""
import collections, itertools, json, sys
from pathlib import Path

B, out = Path(sys.argv[1]), Path(sys.argv[2])
GROUP = dict(CV='A', CO='A', HA='A', OP='B', FC='B', MS='B', TG='B', PS='C', UC='D', UN='UN')
ORDER = ['CV', 'CO', 'HA', 'OP', 'FC', 'MS', 'TG', 'PS', 'UC', 'UN']
K = len(list(B.glob('chunk_*_ids.txt')))
shared = set(json.loads((B / 'shared_ids.json').read_text()))

votes = collections.defaultdict(list)  # id -> [(code, loc, conf, reason)]
for k in range(K):
    want = set((B / f'chunk_{k}_ids.txt').read_text().split())
    got = {}
    for line in (out / f'chunk_{k}.tsv').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        i, loc, code, conf, reason = (line.split('\t') + [''] * 4)[:5]
        assert code in GROUP and loc in ('y', 'n') and conf in ('h', 'm'), (k, line)
        assert i not in got, (k, 'duplicate', i)
        got[i] = (code, loc, conf, reason)
    assert set(got) == want, (k, 'missing', len(want - set(got)), 'extra', len(set(got) - want))
    for i, v in got.items():
        votes[i].append(v)


def fleiss(items, cats):
    """Fleiss' kappa for items rated by the same number of raters"""
    n = len(items[0]); N = len(items)
    p = collections.Counter(c for it in items for c in it)
    pe = sum((p[c] / (N * n)) ** 2 for c in cats)
    pbar = sum((sum(v * v for v in collections.Counter(it).values()) - n) / (n * (n - 1)) for it in items) / N
    return (pbar - pe) / (1 - pe)


S = sorted(shared)
codes = [[v[0] for v in votes[i]] for i in S]
groups = [[GROUP[c] for c in it] for it in codes]
pc = [a == b for it in codes for a, b in itertools.combinations(it, 2)]
pg = [a == b for it in groups for a, b in itertools.combinations(it, 2)]
pab = [a == b for it in groups for a, b in itertools.combinations(it, 2) if {a, b} <= {'A', 'B'}]
print(f'## agreement on {len(S)} shared items ({K} labelers each)')
print(f'  pairwise: code {sum(pc) / len(pc):.2f}, group {sum(pg) / len(pg):.2f}, A-vs-B (pairs where both A/B) '
      f'{sum(pab) / len(pab):.2f}')
print(f'  Fleiss kappa: code {fleiss(codes, ORDER):.2f}, group {fleiss(groups, set(GROUP.values())):.2f}')
print(f'  all {K} agree: code {sum(len(set(it)) == 1 for it in codes)}/{len(S)}, '
      f'group {sum(len(set(it)) == 1 for it in groups)}/{len(S)}')
print(f'  majority size (group): {dict(collections.Counter(collections.Counter(it).most_common(1)[0][1] for it in groups))}')
dis = collections.Counter('/'.join(sorted(set(it))) for it in codes if len(set(it)) > 1)
print('  code disagreement patterns:', dict(dis.most_common(12)))

final = {}
for i, vs in votes.items():
    c = collections.Counter(v[0] for v in vs).most_common()
    code = c[0][0]
    conf = 'm' if len(c) > 1 else vs[0][2]
    loc = collections.Counter(v[1] for v in vs).most_common(1)[0][0]
    final[i] = (code, loc, conf, next(v[3] for v in vs if v[0] == code))

n = len(final)
g = collections.Counter(GROUP[v[0]] for v in final.values())
c = collections.Counter(v[0] for v in final.values())
m = sum(v[2] == 'm' for v in final.values())
locn = sum(v[1] == 'n' for v in final.values())
print(f'\n## {n} SETUP errors')
print('  groups: ' + '  '.join(f'{k}={g[k]} ({g[k] / n:.0%})' for k in ['A', 'B', 'C', 'D', 'UN']))
print('  codes : ' + '  '.join(f'{k}={c[k]}' for k in ORDER if c[k]))
print(f'  borderline (conf m): {m} ({m / n:.0%})  by group: ' + '  '.join(
    f'{k} {sum(v[2] == "m" for v in final.values() if GROUP[v[0]] == k) / max(1, g[k]):.0%}' for k in ['A', 'B']))
print(f'  auto-located step was NOT the first error (loc=n): {locn} ({locn / n:.0%})')
for k in range(K):
    kc = collections.Counter(GROUP[v[0]] for i, vs in votes.items() for j, v in enumerate(vs)
                             if i not in shared and (B / f'chunk_{k}_ids.txt').read_text().count(i + '\n'))
    if k == 0:
        print('  per-labeler A share on own unique items:', end='')
    print(f' {kc["A"] / max(1, sum(kc.values())):.0%}', end='')
print()

with open(out / 'ab_labels.tsv', 'w', encoding='utf-8') as f, open(out / 'ab_types.jsonl', 'w') as h:
    f.write('id\tloc\tcode\tgroup\tconf\treason\n')
    for i in sorted(final):
        code, loc, conf, reason = final[i]
        f.write(f'{i}\t{loc}\t{code}\t{GROUP[code]}\t{conf}\t{reason}\n')
        split, pid, sample = i.split('-')
        h.write(json.dumps(dict(split=split, pid=int(pid), sample=int(sample), cat=GROUP[code], code=code,
                                conf=conf, loc=loc)) + '\n')
