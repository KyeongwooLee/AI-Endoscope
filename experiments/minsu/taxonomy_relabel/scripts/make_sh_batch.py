"""Build one shared batch to measure how ambiguous Sunghun's 7-type error labels are.

N items drawn at random from his 1,100 labeled wrong answers (GSM8K + SVAMP), at most one per question. Every labeler
labels all of them with his codebook; his own label is NOT shown (saved separately for the comparison).
usage: python -I make_sh_batch.py <labels.jsonl> <out_dir> [N]
writes <out_dir>/items.txt, ids.txt, sunghun_labels.json
"""
import json, os, random, re, sys

SRC, OUT = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = random.Random(20261105)
rows = [json.loads(l) for l in open(SRC, encoding='utf-8')]
rng.shuffle(rows)
seen, pick = set(), []
for r in rows:
    if r['question_id'] not in seen:
        seen.add(r['question_id'])
        pick.append(r)
    if len(pick) == N:
        break


def item(r):
    if r['dataset'] == 'svamp':
        ref = f"{r['reference_equation']} = {r['gold_number']}"
    else:
        ref = re.sub(r'<<[^>]*>>', '', r['reference']).replace('\n', ' | ')
    return (f"=== ITEM id={r['id']} ===\nDATASET: {r['dataset']}\nQUESTION: {r['question']}\n"
            f"GOLD ANSWER: {r['gold_number']} | MODEL'S FINAL ANSWER: {r['predicted_number']}\n"
            f"REFERENCE: {ref}\n--- MODEL RESPONSE ---\n{r['response'].rstrip()}\n\n")


os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/items.txt', 'w', encoding='utf-8') as f:
    f.write(''.join(map(item, pick)))
with open(f'{OUT}/ids.txt', 'w') as f:
    f.write(''.join(r['id'] + '\n' for r in pick))
json.dump({r['id']: dict(type=r['math_error_type'], certainty=r['certainty'], dataset=r['dataset'],
                         reference_issue=r['reference_issue']) for r in pick},
          open(f'{OUT}/sunghun_labels.json', 'w'), ensure_ascii=False, indent=0)
c = {}
for r in pick:
    c[r['math_error_type']] = c.get(r['math_error_type'], 0) + 1
print(f"items {len(pick)} (svamp {sum(r['dataset'] == 'svamp' for r in pick)}); Sunghun types {c}; "
      f"{os.path.getsize(f'{OUT}/items.txt') / 1e3:.0f} KB")
