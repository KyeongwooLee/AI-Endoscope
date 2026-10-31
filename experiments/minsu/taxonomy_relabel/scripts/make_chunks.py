"""Split the 885 located errors into 4 labeling chunks + 30 shared calibration items (in every chunk).

Sunghun's main type is withheld from the chunks to reduce anchoring; his explanation/subtypes are kept as evidence.
usage: python -I make_chunks.py <labels.jsonl> <out_dir>
"""
import json, random, sys, collections
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rows = [json.loads(l) for l in src.read_text(encoding='utf-8').split('\n') if l.strip()]
rows = [r for r in rows if r['first_observable_error_span']]
assert len(rows) == 885

rng = random.Random(20261030)
by_type = collections.defaultdict(list)
for r in rows:
    by_type[r['math_error_type']].append(r)
quota = dict(modeling_relation=18, arithmetic_execution=5, reasoning_state=4, unit_conversion=3)
shared = [r for t, n in quota.items() for r in rng.sample(by_type[t], n)]
shared_ids = {r['id'] for r in shared}
rest = [r for r in rows if r['id'] not in shared_ids]
rng.shuffle(rest)

keep = ['id', 'dataset', 'question', 'evidence_quote', 'subtypes', 'explanation', 'corrected_step',
        'predicted_number', 'gold_number']
for k in range(4):
    chunk = rest[k::4] + shared
    rng.shuffle(chunk)
    with open(out / f'chunk_{k}.jsonl', 'w', encoding='utf-8') as f:
        for r in chunk:
            f.write(json.dumps({x: r[x] for x in keep}, ensure_ascii=False) + '\n')
    print(f'chunk_{k}: {len(chunk)} items')
(out / 'shared_ids.json').write_text(json.dumps(sorted(shared_ids)))
