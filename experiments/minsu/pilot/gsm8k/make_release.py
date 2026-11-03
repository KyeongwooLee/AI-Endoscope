"""Build the GitHub release files for our own-generated GSM8K wrong answers (Qwen2.5-1.5B-Instruct).

Only the GSM8K train split (the official test split stays unreleased, as in the team's other releases) and only
answers with a valid '####' line (format failures are excluded, as in the team's other releases).
errors.jsonl    : question, GSM8K reference, response, rule labels (auto_types.py)
token_ids.jsonl : exact prompt / output token ids (for replaying the model and extracting hidden states)
usage: python -I make_release.py <gen_dir> <gsm8k_dir> <auto_types.jsonl> <out_dir>
"""
import glob, json, os, sys

GEN, DATA, TYPES, OUT = sys.argv[1:5]
rows = [r for f in sorted(glob.glob(f'{GEN}/train_*.jsonl')) for r in map(json.loads, open(f))]
ref = [json.loads(l)['answer'] for l in open(f'{DATA}/train.jsonl')]
iid = lambda r: f"{r['split']}-{r['pid']}-{r['sample']}"
rule = {iid(t): t for t in map(json.loads, open(TYPES))}
solved = {r['pid'] for r in rows if r['correct'] and r['format_ok']}
wrong = sorted((r for r in rows if not r['correct'] and r['format_ok']), key=lambda r: (r['pid'], r['sample']))
assert all(r['split'] == 'train' for r in wrong) and all(iid(r) in rule for r in wrong)
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/errors.jsonl', 'w', encoding='utf-8') as f, open(f'{OUT}/token_ids.jsonl', 'w') as g:
    for r in wrong:
        t = rule[iid(r)]
        f.write(json.dumps(dict(
            id=iid(r), split=r['split'], pid=r['pid'], sample=r['sample'], mode=r['mode'],
            question=r['question'], reference=ref[r['pid']], gold=r['gold'], pred=r['pred'],
            response=r['text'], finish=r['finish'], has_correct_sibling=r['pid'] in solved,
            rule_type=t['cat'], rule_first_wrong_eq=t.get('k'), rule_n_eq=t.get('n_eq')), ensure_ascii=False) + '\n')
        g.write(json.dumps(dict(id=iid(r), prompt_token_ids=r['prompt_ids'], output_token_ids=r['output_ids'])) + '\n')
c = {}
for r in wrong:
    c[rule[iid(r)]['cat']] = c.get(rule[iid(r)]['cat'], 0) + 1
print(f"wrong {len(wrong)}, problems {len({r['pid'] for r in wrong})}, solutions in split {len(rows)}; "
      f"rule types {dict(sorted(c.items(), key=lambda x: -x[1]))}; truncated {sum(r['finish'] == 'length' for r in wrong)}")
