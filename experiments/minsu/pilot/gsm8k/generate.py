"""Qwen2.5-1.5B-Instruct solves GSM8K itself (vLLM). Per problem: 1 greedy + K sampled solutions.

Saves exact prompt/output token ids so features can be extracted by replaying the model's own path.
Prompt = the one the teammate used, for comparability.
usage: python generate.py <gsm8k_dir> <out_dir> <K> <shard> <n_shards> [limit]
"""
import json, os, re, sys, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SRC, OUT = sys.argv[1], sys.argv[2]
K, SHARD, NSH = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
LIMIT = int(sys.argv[6]) if len(sys.argv) > 6 else None
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
SYSTEM = ('You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer '
          'with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write '
          'nothing after this final line. Do not omit this line. The final answer must contain only a number, without '
          'units, currency symbols, Markdown emphasis, or LaTeX.')
USER = ('Problem:\n{question}\n\nSolve the problem step by step. After your reasoning, write the answer on the LAST '
        'line using exactly this format:\n#### <number>\nReplace <number> with the numerical answer. Do not write any '
        'text after that line.')
os.makedirs(OUT, exist_ok=True)


def to_num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def parse_answer(text):
    """number after the last '####'; falls back to the last number in the text (format_ok=False)."""
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    if m:
        return to_num(m[-1]), True
    nums = re.findall(r'-?[\d,]*\.?\d+', text)
    return (to_num(nums[-1]) if nums else None), False


tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(MODEL, dtype='bfloat16', gpu_memory_utilization=0.85, seed=0)
greedy = SamplingParams(temperature=0.0, max_tokens=1024)
sampled = SamplingParams(n=K, temperature=0.7, top_p=0.95, max_tokens=1024)  # engine seed=0 (per-request seed is slow)

for split in ['test', 'train']:
    rows = [json.loads(l) for l in open(f'{SRC}/{split}.jsonl')]
    rows = rows[:LIMIT] if LIMIT else rows
    ids = [i for i in range(len(rows)) if i % NSH == SHARD]
    rows = [rows[i] for i in ids]
    prompts = [{'prompt_token_ids': tok.apply_chat_template(
        [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(question=r['question'])}],
        add_generation_prompt=True, tokenize=True)} for r in rows]
    t0 = time.time()
    outs = {'greedy': llm.generate(prompts, greedy), 'sampled': llm.generate(prompts, sampled) if K else None}
    stat = {m: [0, 0, 0] for m in ('greedy', 'sampled')}  # n, correct, format_ok
    with open(f'{OUT}/{split}_{SHARD}.jsonl', 'w') as f:
        for i, r in enumerate(rows):
            gold = to_num(r['answer'].split('####')[-1])
            cands = [('greedy', 0, outs['greedy'][i].outputs[0])]
            if K:
                cands += [('sampled', j + 1, o) for j, o in enumerate(outs['sampled'][i].outputs)]
            for mode, j, o in cands:
                pred, fmt = parse_answer(o.text)
                ok = pred is not None and gold is not None and abs(pred - gold) < 1e-6
                stat[mode][0] += 1; stat[mode][1] += ok; stat[mode][2] += fmt
                f.write(json.dumps(dict(split=split, pid=ids[i], sample=j, mode=mode, question=r['question'], gold=gold,
                                        text=o.text, prompt_ids=prompts[i]['prompt_token_ids'],
                                        output_ids=list(o.token_ids), finish=o.finish_reason, pred=pred,
                                        format_ok=fmt, correct=bool(ok))) + '\n')
    print(f'{split}: {len(rows)} problems, {time.time() - t0:.0f}s | ' + ' | '.join(
        f'{m}: n={v[0]} acc={v[1] / v[0]:.3f} format_ok={v[2] / v[0]:.3f}' for m, v in stat.items() if v[0]), flush=True)
