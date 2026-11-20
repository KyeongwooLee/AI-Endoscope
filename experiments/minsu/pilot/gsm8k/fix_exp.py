"""Correction-type experiment: which correction fixes each of the model's own wrong answers? (vLLM, sampling as generate.py)

Sample = sample_ids.json from make_fix_batches.py (train-split wrong answers, one per problem). Conditions, N samples each:
  restart  solve again from scratch (same prompt)
  calc     calculator, no location needed: the first equation (no variable, no LaTeX on its line) whose written result differs from
           its left side is cut at its '=' and the computed value is inserted, then the model continues
           (not applicable when no equation has wrong arithmetic)
  code     a new chat asks for a Python program that prints the answer; fix_code.py runs it without network, with a
           5 s time limit and 2 GB of memory, and the last number it prints (or its final bare expression) is the answer
  notx     original prompt plus a note that the earlier answer (its value only) is wrong
  hot      restart at temperature 1.0 instead of 0.7
  plan, decompose, equation, verify   prompt strategies (env CONDS): a new chat with the original system prompt and
           a user turn asking to solve in that way (STYLES)
  line     (only with an audit labels file) the wrong solution is shown, then a user turn quotes the audited first wrong
           line and asks to solve again fixing it (not applicable when the audit gives no line)
  cut      (only with an audit labels file) the wrong solution is kept up to the start of the audited first wrong line and
           the model continues from there (first line wrong: nothing is kept, i.e. a restart)
env: N attempts (8), SEED (0), CONDS (restart,code,calc), AUDIT_CONDS (line)
usage: python fix_exp.py <gen_dir> <auto_types.jsonl> <sample_ids.json> <out_dir> <shard> <n_shards> [<audit labels.tsv>]
writes <out_dir>/fix_<shard>.jsonl, or with an audit file only the conditions in AUDIT_CONDS (default line) to
<out_dir>/<conds joined by _>_<shard>.jsonl
"""
import bisect, glob, itertools, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations, equation_spans, close
from fix_code import run_code

GEN, TYPES, SAMPLE, OUT = sys.argv[1:5]
SHARD, NSH = int(sys.argv[5]), int(sys.argv[6])
AUDIT = sys.argv[7] if len(sys.argv) > 7 else None
AUDIT_CONDS = os.environ.get('AUDIT_CONDS', 'line').split(',')
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
N, SEED = int(os.environ.get('N', 8)), int(os.environ.get('SEED', 0))  # more attempts: new SEED, new out_dir
CONDS = os.environ.get('CONDS', 'restart,code,calc').split(',')  # without an audit file
FEEDBACK = ('This line of your solution is wrong:\n{line}\n'
            'Solve the problem again from the beginning and fix this mistake. '
            'End with a separate final line formatted exactly as #### followed by the numerical answer.')
CODE_SYSTEM = 'You are a helpful assistant that solves math problems by writing Python code.'
CODE_USER = ('Write a Python program that solves the following problem and prints only the final numerical answer.\n\n'
             'Problem:\n{question}\n\nReply with a single ```python code block and nothing else.')
SYSTEM = ('You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer '
          'with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write '
          'nothing after this final line. Do not omit this line. The final answer must contain only a number, without '
          'units, currency symbols, Markdown emphasis, or LaTeX.')  # as in generate.py
FORMAT = ('After your reasoning, write the answer on the LAST line using exactly this format:\n#### <number>\n'
          'Replace <number> with the numerical answer. Do not write any text after that line.')
USER = ('Problem:\n{question}\n\nSolve the problem step by step. ' + FORMAT)  # as in generate.py
NOTX = '\n\nNote: an earlier attempt at this problem gave the answer {pred:g}, which is wrong.'
STYLES = {  # prompt strategies: same system prompt, a different way of solving
    'plan': 'Before solving, first write what the question asks for and list every given quantity. Then solve step by step.',
    'decompose': 'First break the problem into a numbered list of smaller sub-questions. Then answer them one by one.',
    'equation': 'Define variables for the unknown quantities, write equations that express the relations in the problem, '
                'and solve them.',
    'verify': 'Solve step by step. Then check your answer by putting it back into the conditions of the problem. If it '
              'does not fit, find the mistake and solve again.'}
VAR = re.compile(r'(?<![A-Za-z])(?!a\b)[A-Za-z](?![A-Za-z])|\\\(|\\\[|\\frac')  # single-letter variable or LaTeX
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
special = set(tok.all_special_ids)


def to_num(s):  # as in generate.py
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


def char_spans(out):  # as in extract_loc.py
    n = len(out)
    while n and out[n - 1] in special:
        n -= 1
    pieces = [tok.decode([t]) for t in out[:n]]
    text = ''.join(pieces)
    if text != tok.decode(out[:n]):
        ends = [len(tok.decode(out[:i + 1])) for i in range(n)]
        text = tok.decode(out[:n])
    else:
        ends = list(itertools.accumulate(map(len, pieces)))
    return text, ends, n


def fmt(v):  # as in correct_exp.py
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f'{v:.2f}'.rstrip('0').rstrip('.')


types = {f"{t['split']}-{t['pid']}-{t['sample']}": t for t in map(json.loads, open(TYPES))}
rows = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in sorted(glob.glob(f'{GEN}/train_*.jsonl'))
        for r in map(json.loads, open(f))}
ids = [i for j, i in enumerate(json.load(open(SAMPLE))) if j % NSH == SHARD]
audit = {}
if AUDIT:
    for line in open(AUDIT, encoding='utf-8').read().splitlines()[1:]:
        x = line.split('\t')
        audit[x[0]] = (x[1], x[2])
end = tok.encode('<|im_end|>\n<|im_start|>user\n')
asst = tok.encode('<|im_end|>\n<|im_start|>assistant\n')

reqs, na = [], {'calc': 0, 'line': 0, 'cut': 0}
for i in ids:
    r, t = rows[i], types[i]
    text, ends, n = char_spans(r['output_ids'])
    out, P0 = r['output_ids'][:n], r['prompt_ids']
    base = dict(id=i, gold=r['gold'], cat=t['cat'], n_eq=t['n_eq'], rule_k=t['k'])
    if AUDIT:
        cls, num = audit[i]
        lines = [l for l in r['text'].rstrip().split('\n') if l.strip()]
        if num == '-':
            na['line'] += 1
            continue
        line = lines[int(num[1:]) - 1].strip()
        meta = dict(base, audit_cls=cls, audit_num=num, line=line)
        if 'line' in AUDIT_CONDS:
            reqs.append((dict(meta, cond='line'), P0 + out + end + tok.encode(FEEDBACK.format(line=line)) + asst, ''))
        if 'cut' in AUDIT_CONDS:
            starts = [m.start() for m in re.finditer(r'(?m)^(?=.*\S)', text)]  # start of each non-empty line
            if len(starts) != len(lines) or text[starts[int(num[1:]) - 1]:].split('\n')[0].strip() != line:
                na['cut'] += 1
                continue
            s = starts[int(num[1:]) - 1]
            ts = bisect.bisect_right(ends, s - 1) + 1 if s > 0 else 0  # keep the token holding char s-1
            reqs.append((dict(meta, cond='cut', cut=ts), P0 + out[:ts], tok.decode(out[:ts])))
        continue
    if 'restart' in CONDS:
        reqs.append((dict(base, cond='restart'), P0, ''))
    q = tok.apply_chat_template([{'role': 'system', 'content': CODE_SYSTEM},
                                 {'role': 'user', 'content': CODE_USER.format(question=r['question'])}],
                                add_generation_prompt=True)
    if 'code' in CONDS:
        reqs.append((dict(base, cond='code'), q, ''))
    if 'notx' in CONDS:  # told only that its earlier answer is wrong; the earlier solution is not shown
        reqs.append((dict(base, cond='notx'), tok.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM},
             {'role': 'user', 'content': USER.format(question=r['question']) + NOTX.format(pred=r['pred'])}],
            add_generation_prompt=True), ''))
    if 'hot' in CONDS:  # restart at temperature 1.0
        reqs.append((dict(base, cond='hot'), P0, ''))
    for name, how in STYLES.items():
        if name in CONDS:
            reqs.append((dict(base, cond=name), tok.apply_chat_template(
                [{'role': 'system', 'content': SYSTEM},
                 {'role': 'user', 'content': f"Problem:\n{r['question']}\n\n{how} {FORMAT}"}], add_generation_prompt=True), ''))
    sp = equation_spans(text) if t['n_eq'] else None
    eqs = equations(text)
    calc = None
    if sp is not None and len(sp) == len(eqs):
        ti = lambda c: bisect.bisect_right(ends, c)
        for j, ((lhs, lv, rv, *_), (ls, eqc, ns, ne)) in enumerate(zip(eqs, sp)):
            te = ti(eqc)
            if VAR.search(text[ls:ne]):  # algebra with a variable: the parser drops it and the 'error' is spurious
                continue
            if not close(lv, rv) and ends[te] <= ns:
                ins = ' ' + fmt(lv)
                calc = (dict(base, cond='calc', step=j, wrote=rv, true=lv), P0 + out[:te + 1] + tok.encode(ins),
                        tok.decode(out[:te + 1]) + ins)
                break
    if calc and 'calc' in CONDS:
        reqs.append(calc)
    else:
        na['calc'] += 1
print(f'shard {SHARD}: solutions {len(ids)}, requests {len(reqs)}, not applicable {na}', flush=True)
if os.environ.get('DRY'):
    for meta, p, prefix in [x for x in reqs if x[0]['cond'] in ('calc', 'line', 'cut')][:3]:
        print('-----', {k: v for k, v in meta.items() if k != 'gold'}, '\n...' + (prefix[-300:] or tok.decode(p)[-500:]))
    print('----- code prompt\n' + tok.decode(next(p for m, p, _ in reqs if m['cond'] == 'code')) if not AUDIT else '')
    sys.exit()

llm = LLM(MODEL, dtype='bfloat16', gpu_memory_utilization=0.85, seed=SEED)
outs = llm.generate([{'prompt_token_ids': p} for _, p, _ in reqs],
                    [SamplingParams(n=N, temperature=1.0 if m['cond'] == 'hot' else 0.7, top_p=0.95, max_tokens=1024)
                     for m, _, _ in reqs])
t0 = time.time()
with ThreadPoolExecutor(32) as ex:
    code_preds = {k: list(ex.map(lambda t: run_code(t, OUT), [c.text for c in o.outputs]))
                  for k, ((meta, _, _), o) in enumerate(zip(reqs, outs)) if meta['cond'] == 'code'}
print(f'ran {sum(map(len, code_preds.values()))} programs, {time.time() - t0:.0f}s', flush=True)
with open(f"{OUT}/{'_'.join(AUDIT_CONDS) if AUDIT else 'fix'}_{SHARD}.jsonl", 'w') as f:
    for k, ((meta, _, prefix), o) in enumerate(zip(reqs, outs)):
        preds = code_preds[k] if k in code_preds else [parse_answer(prefix + c.text)[0] for c in o.outputs]
        ok = [p is not None and abs(p - meta['gold']) < 1e-6 for p in preds]
        f.write(json.dumps(dict(meta, prefix=prefix[-300:], preds=preds, correct=ok,
                                new_tokens=[len(c.token_ids) for c in o.outputs], texts=[c.text for c in o.outputs])) + '\n')
print('done', flush=True)
