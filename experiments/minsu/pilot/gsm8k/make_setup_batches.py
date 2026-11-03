"""Build labeling batches for step 2: SETUP errors (first wrong step located by auto_types.py) -> A/B judgment.

One SETUP error per problem, only problems where the same model also solved it correctly (that solution is shown
next to the wrong one). N_LABELERS chunks of N_UNIQUE items + N_SHARED items in every chunk (agreement check).
Items are written as plain text so a labeler can read them in parts.
usage: python -I make_setup_batches.py <gen_dir> <gsm8k_dir> <auto_types.jsonl> <out_dir>
"""
import glob, json, os, random, re, sys

GEN, DATA, TYPES, OUT = sys.argv[1:5]
N_LABELERS, N_UNIQUE, N_SHARED = 6, 160, 30
rng = random.Random(20261101)
rows = {(r['split'], r['pid'], r['sample']): r for f in glob.glob(f'{GEN}/*_*.jsonl') for r in map(json.loads, open(f))}
ref = {s: [json.loads(l)['answer'] for l in open(f'{DATA}/{s}.jsonl')] for s in ('train', 'test')}
types = [json.loads(l) for l in open(TYPES)]

byq = {}
for t in types:
    if t['cat'] == 'SETUP' and t['has_right']:
        byq.setdefault((t['split'], t['pid']), []).append(t)
picked = [rng.choice(v) for _, v in sorted(byq.items())]
rng.shuffle(picked)
need = N_LABELERS * N_UNIQUE + N_SHARED
assert len(picked) >= need, len(picked)
picked = picked[:need]
shared, rest = picked[:N_SHARED], picked[N_SHARED:]


def item(t):
    r = rows[(t['split'], t['pid'], t['sample'])]
    right = [x for (s, p, _), x in rows.items() if s == t['split'] and p == t['pid'] and x['correct'] and x['format_ok']]
    c = rng.choice(right)
    gold_ref = re.sub(r'<<[^>]*>>', '', ref[t['split']][int(t['pid'])]).replace('\n', ' | ')
    wrong = r['text'].split('####')[0].strip()
    wrong = wrong if len(wrong) <= 1600 else '...' + wrong[-1600:]
    corr = c['text'].split('####')[0].strip()
    corr = corr if len(corr) <= 1000 else corr[:1000] + ' ...'
    iid = f"{t['split']}-{t['pid']}-{t['sample']}"
    return iid, (f"=== ITEM id={iid} ===\nQUESTION: {r['question']}\n"
                 f"GOLD ANSWER: {t['gold']:g} | WRONG ANSWER: {t['pred']:g}\n"
                 f"GSM8K REFERENCE SOLUTION: {gold_ref}\n"
                 f"--- CORRECT SOLUTION (same model, another sample) ---\n{corr}\n"
                 f"--- WRONG SOLUTION ---\n{wrong}\n"
                 f">>> AUTO-DETECTED FIRST WRONG STEP: {t['line']}\n\n")


os.makedirs(OUT, exist_ok=True)
for k in range(N_LABELERS):
    chunk = rest[k * N_UNIQUE:(k + 1) * N_UNIQUE] + shared
    rng.shuffle(chunk)
    with open(f'{OUT}/chunk_{k}.txt', 'w') as f, open(f'{OUT}/chunk_{k}_ids.txt', 'w') as g:
        for t in chunk:
            iid, txt = item(t)
            f.write(txt); g.write(iid + '\n')
json.dump(sorted(f"{t['split']}-{t['pid']}-{t['sample']}" for t in shared), open(f'{OUT}/shared_ids.json', 'w'))
print(f'problems with a SETUP error and a correct sibling: {len(byq)}; chunks: {N_LABELERS} x '
      f'({N_UNIQUE} + {N_SHARED} shared)')
