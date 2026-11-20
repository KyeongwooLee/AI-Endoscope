"""Paper replication (Orgad et al., ICLR 2025, Sec. 5.1): K = 30 responses at temperature 1 for EVERY train question
(not only ones answered wrongly), same prompt as generate.py. Only the parsed answers are kept.
usage: python resample_all.py <gen_dir> <out_dir> <shard> <n_shards>   -> <out_dir>/rs_<shard>.jsonl
"""
import glob, json, os, sys, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
GEN, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
K = int(os.environ.get('K', 30))
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
SYSTEM = ('You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer '
          'with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write '
          'nothing after this final line. Do not omit this line. The final answer must contain only a number, without '
          'units, currency symbols, Markdown emphasis, or LaTeX.')
USER = ('Problem:\n{question}\n\nSolve the problem step by step. After your reasoning, write the answer on the LAST '
        'line using exactly this format:\n#### <number>\nReplace <number> with the numerical answer. Do not write any '
        'text after that line.')
import re
def to_num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None
def parse_answer(text):  # as in generate.py
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    if m:
        return to_num(m[-1]), True
    nums = re.findall(r'-?[\d,]*\.?\d+', text)
    return (to_num(nums[-1]) if nums else None), False
os.makedirs(OUT, exist_ok=True)
rows = sorted((r for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f)) if r['mode'] == 'greedy'),
              key=lambda r: r['pid'])
rows = [r for j, r in enumerate(rows) if j % NSH == SHARD]
tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype='bfloat16', gpu_memory_utilization=0.85, seed=SHARD)
prompts = [{'prompt_token_ids': tok.apply_chat_template(
    [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(question=r['question'])}],
    add_generation_prompt=True, tokenize=True)} for r in rows]
t0 = time.time()
outs = llm.generate(prompts, SamplingParams(n=K, temperature=1.0, top_p=1.0, max_tokens=1024))
with open(f'{OUT}/rs_{SHARD}.jsonl.tmp', 'w') as f:
    for r, o in zip(rows, outs):
        pa = [parse_answer(c.text) for c in o.outputs]
        f.write(json.dumps(dict(pid=r['pid'], gold=r['gold'], greedy_id=f"train-{r['pid']}-{r['sample']}",
                                greedy_correct=r['correct'], preds=[p for p, _ in pa], format_ok=[ok for _, ok in pa],
                                finish=[c.finish_reason for c in o.outputs])) + '\n')
os.replace(f'{OUT}/rs_{SHARD}.jsonl.tmp', f'{OUT}/rs_{SHARD}.jsonl')
print('done', len(rows), f'{time.time() - t0:.0f}s', flush=True)
