"""Turn Sunghun's 1,100 type-labeled wrong answers into the gen-file format extract_gen.py reads.

Uses the exact prompt / generated token ids from his release records (GSM8K sampled posthoc + SVAMP inference), so
extract_gen.py replays the same token path and saves our usual features (hidden q_end / r_end / r_mean / ans at
layers 4..28 + confidence). split = dataset, pid = question_id, sample = response id.
usage: python -I make_sh_gen.py <release_dir> <labels.jsonl> <out_dir>   (writes <out_dir>/sh_0.jsonl)
"""
import json, os, sys

REL, LABELS, OUT = sys.argv[1:4]
want = {json.loads(l)['id'] for l in open(LABELS, encoding='utf-8')}
os.makedirs(OUT, exist_ok=True)
n = 0
with open(f'{OUT}/sh_0.jsonl', 'w') as f:
    for d in ('gsm8k-sampled-posthoc-v1', 'svamp-inference-v1'):
        for line in open(f'{REL}/{d}/records.jsonl', encoding='utf-8'):
            r = json.loads(line)
            if r['id'] not in want:
                continue
            g = r['generation']
            assert all(isinstance(t, int) for t in g['prompt_token_ids'] + g['generated_token_ids'])
            f.write(json.dumps(dict(split=r['dataset'], pid=r['question_id'], sample=r['id'], mode='sample',
                                    correct=r['answer_correct'] == 'True', format_ok=True, finish=g['stop_reason'],
                                    prompt_ids=g['prompt_token_ids'], output_ids=g['generated_token_ids'])) + '\n')
            n += 1
assert n == len(want), (n, len(want))
print('wrote', n)
