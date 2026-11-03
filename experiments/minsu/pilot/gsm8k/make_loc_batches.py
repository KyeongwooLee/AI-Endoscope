"""Build labeling batches for the location audit: where does each wrong solution FIRST go wrong?

Pool = the solutions of the location experiment (keys of preds_loc2.json). One solution per problem: N_ARITH problems
with an ARITH error, the rest SETUP/NEWNUM. Every equation the rule can check is marked [E1], [E2], ... right after
its result; the rule's own pick (E{k+1}) is NOT shown, so labelers judge independently.
N_LABELERS chunks of N_UNIQUE items + N_SHARED items in every chunk (agreement check).
usage: python -I make_loc_batches.py <gen_dir> <gsm8k_dir> <auto_types.jsonl> <preds_loc2.json> <out_dir>
"""
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equation_spans

GEN, DATA, TYPES, PREDS, OUT = sys.argv[1:6]
N_LABELERS, N_UNIQUE, N_SHARED, N_ARITH, MAX_CHARS = 5, 120, 30, 60, 3000
rng = random.Random(20261103)
rows = {(r['split'], r['pid'], r['sample']): r for f in glob.glob(f'{GEN}/*_*.jsonl') for r in map(json.loads, open(f))}
ref = {s: [json.loads(l)['answer'] for l in open(f'{DATA}/{s}.jsonl')] for s in ('train', 'test')}
pool = set(json.load(open(PREDS))['k'])
types = [t for t in map(json.loads, open(TYPES)) if f"{t['split']}-{t['pid']}-{t['sample']}" in pool]


def marked(t):
    """wrong solution with [E#] after each equation's result; None if it cannot be marked or is too long"""
    text = rows[(t['split'], t['pid'], t['sample'])]['text']
    sp = equation_spans(text)
    body = text.split('####')[0].rstrip()
    if sp is None or len(sp) != t['n_eq'] or len(body) > MAX_CHARS:
        return None
    for i, (_, _, _, end) in reversed(list(enumerate(sp))):
        body = body[:end] + f' [E{i + 1}]' + body[end:]
    return body


byq = {}
for t in types:
    byq.setdefault((t['split'], t['pid']), []).append(t)
probs = sorted(byq)
rng.shuffle(probs)
arith, other = [], []
for p in probs:
    a = [t for t in byq[p] if t['cat'] == 'ARITH' and marked(t)]
    o = [t for t in byq[p] if t['cat'] != 'ARITH' and marked(t)]
    if a and len(arith) < N_ARITH:
        arith.append(rng.choice(a))
    elif o and len(other) < N_LABELERS * N_UNIQUE + N_SHARED - N_ARITH:
        other.append(rng.choice(o))
picked = arith + other
rng.shuffle(picked)
assert len(picked) == N_LABELERS * N_UNIQUE + N_SHARED, len(picked)
shared, rest = picked[:N_SHARED], picked[N_SHARED:]


def item(t):
    r = rows[(t['split'], t['pid'], t['sample'])]
    right = [x for (s, p, _), x in rows.items() if s == t['split'] and p == t['pid'] and x['correct'] and x['format_ok']]
    gold_ref = re.sub(r'<<[^>]*>>', '', ref[t['split']][int(t['pid'])]).replace('\n', ' | ')
    if right:
        corr = rng.choice(right)['text'].split('####')[0].strip()
        corr = corr if len(corr) <= 1000 else corr[:1000] + ' ...'
    else:
        corr = '(none: the model never solved this problem correctly)'
    iid = f"{t['split']}-{t['pid']}-{t['sample']}"
    return iid, (f"=== ITEM id={iid} ===\nQUESTION: {r['question']}\n"
                 f"GOLD ANSWER: {t['gold']:g} | WRONG FINAL ANSWER: {t['pred']:g}\n"
                 f"GSM8K REFERENCE SOLUTION: {gold_ref}\n"
                 f"--- CORRECT SOLUTION (same model, another sample) ---\n{corr}\n"
                 f"--- WRONG SOLUTION (equations marked E1..E{t['n_eq']}) ---\n{marked(t)}\n\n")


os.makedirs(OUT, exist_ok=True)
for k in range(N_LABELERS):
    chunk = rest[k * N_UNIQUE:(k + 1) * N_UNIQUE] + shared
    rng.shuffle(chunk)
    with open(f'{OUT}/chunk_{k}.txt', 'w') as f, open(f'{OUT}/chunk_{k}_ids.txt', 'w') as g:
        for t in chunk:
            iid, txt = item(t)
            f.write(txt); g.write(iid + '\n')
json.dump(sorted(f"{t['split']}-{t['pid']}-{t['sample']}" for t in shared), open(f'{OUT}/shared_ids.json', 'w'))
json.dump({f"{t['split']}-{t['pid']}-{t['sample']}": dict(k=t['k'], n_eq=t['n_eq'], cat=t['cat']) for t in picked},
          open(f'{OUT}/rule_labels.json', 'w'))
print(f'pool solutions: {len(types)}, problems: {len(byq)}; picked {len(picked)} '
      f'(ARITH {len(arith)}, other {len(other)}); chunks: {N_LABELERS} x ({N_UNIQUE} + {N_SHARED} shared)')
