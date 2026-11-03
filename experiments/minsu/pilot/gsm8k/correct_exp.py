"""Correction experiment on the model's own WRONG test-split GSM8K solutions (vLLM, same sampling as generate.py).

Located errors (first wrong equation known by the rule; location predicted out-of-sample by probe_loc2.py):
  restart      : solve again from scratch
  refine       : show the wrong solution, ask the model to review and correct it (no location given)
  pred_plain   : keep the solution up to the end of the line of the last equation before the PREDICTED first wrong
                 equation (all text about the wrong step is removed), continue from there
  pred_typed   : type-specific: if that step's arithmetic is wrong (calculator check on its text), keep up to its '='
                 and insert the calculator result; otherwise the same cut as pred_plain plus a re-read cue
  true_plain   : like pred_plain at the rule location (upper bound for the location)
  true_typed   : like pred_typed at the rule location
TARGET errors (all equations valid, final answer is an intermediate; type taken from the rule):
  restart, refine, target_cue: keep everything up to the end of the last equation's line, insert a cue to check what
  the question asks, continue.
N samples per condition. Output: one jsonl line per (condition, solution) with the N predictions.
DIFF=1: located errors, *_plain conditions only, N=16 (to estimate 'resample until the redone step differs'),
written to diff_<shard>.jsonl.
usage: python correct_exp.py <gen_dir> <auto_types.jsonl> <preds_loc2.json> <out_dir> <shard> <n_shards>
"""
import bisect, glob, itertools, json, os, re, sys, time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations, equation_spans, close

GEN, TYPES, PREDS, OUT, SHARD, NSH = sys.argv[1:5] + [int(sys.argv[5]), int(sys.argv[6])]
DIFF = bool(os.environ.get('DIFF'))
MODEL, N = 'Qwen/Qwen2.5-1.5B-Instruct', 16 if DIFF else 4
CUE_SETUP = 'Wait, let me re-read the problem and check what the next step should compute.\n'
CUE_TARGET = '\nLet me check what the question is asking for.\n'
REFINE = ('Review your solution above step by step. If you find a mistake, correct it and solve the problem again. '
          'End with a separate final line formatted exactly as #### followed by the numerical answer.')
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


def fmt(v):
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f'{v:.2f}'.rstrip('0').rstrip('.')


P = json.load(open(PREDS))
pred_loc = P['pred'][P['selected']]
types = {f"{t['split']}-{t['pid']}-{t['sample']}": t for t in map(json.loads, open(TYPES))}
rows = {f"{r['split']}-{r['pid']}-{r['sample']}": r for f in sorted(glob.glob(f'{GEN}/test_*.jsonl'))
        for r in map(json.loads, open(f))}
located = sorted(i for i in P['k'] if i.startswith('test-'))
target = sorted(i for i, t in types.items() if t['split'] == 'test' and t['cat'] == 'TARGET')
items = [(i, 'loc') for i in located] + ([] if DIFF else [(i, 'target') for i in target])
items = [x for j, x in enumerate(items) if j % NSH == SHARD]
end = tok.encode('<|im_end|>\n<|im_start|>user\n' + REFINE + '<|im_end|>\n<|im_start|>assistant\n')
assert tok.convert_tokens_to_ids('<|im_end|>') in end and tok.convert_tokens_to_ids('<|im_start|>') in end

reqs, skipped = [], 0
for i, kind in items:
    r, t = rows[i], types[i]
    text, ends, n = char_spans(r['output_ids'])
    out, P0 = r['output_ids'][:n], r['prompt_ids']
    sp = equation_spans(text)
    eqs = equations(text)
    if sp is None or len(sp) != t['n_eq']:
        skipped += 1
        continue
    ti = lambda c: bisect.bisect_right(ends, c)
    base = dict(id=i, kind=kind, gold=r['gold'], true_k=t['k'], n_eq=t['n_eq'], cat=t['cat'])
    if not DIFF:
        reqs.append((dict(base, cond='restart', cut=None), P0, ''))
        reqs.append((dict(base, cond='refine', cut=None), P0 + out + end, ''))
    if kind == 'target':
        c = text.find('\n', sp[-1][3])
        c = len(text) if c < 0 else c + 1
        kt = min(ti(c - 1) + 1, n)
        pre = tok.decode(out[:kt]) + CUE_TARGET
        reqs.append((dict(base, cond='target_cue', cut=kt), P0 + out[:kt] + tok.encode(CUE_TARGET), pre))
        continue
    for name, j in (('pred', pred_loc[i]), ('true', t['k'])):
        ls, eqc, ns, ne = sp[j]
        if j == 0:
            s = 0
        elif sp[j - 1][0] == ls:                                        # 2nd equation on the same line
            s = sp[j - 1][3]
        else:                                                           # end of the previous equation's line
            c = text.find('\n', sp[j - 1][3])
            s = min(ls, len(text) if c < 0 else c + 1)
        ts = ti(s - 1) + 1 if s > 0 else 0                              # keep the token holding char s-1
        reqs.append((dict(base, cond=f'{name}_plain', step=j, cut=ts), P0 + out[:ts], tok.decode(out[:ts])))
        if DIFF:
            continue
        lhs, lv, rv = eqs[j][:3]
        te = ti(eqc)
        if not close(lv, rv) and ends[te] <= ns:                        # calculator: the step's arithmetic is wrong
            kt, ins = te + 1, ' ' + fmt(lv)
            reqs.append((dict(base, cond=f'{name}_typed', step=j, cut=kt, fix='calc'),
                         P0 + out[:kt] + tok.encode(ins), tok.decode(out[:kt]) + ins))
        else:
            kept = tok.decode(out[:ts])
            cue = CUE_SETUP if kept == '' or kept[-1].isspace() else '\n' + CUE_SETUP
            reqs.append((dict(base, cond=f'{name}_typed', step=j, cut=ts, fix='reread'),
                         P0 + out[:ts] + tok.encode(cue), kept + cue))
print(f'shard {SHARD}: solutions {len(items)} (skipped {skipped}), requests {len(reqs)}', flush=True)
if os.environ.get('DRY'):  # show a few prefixes and stop
    for meta, _, prefix in [x for x in reqs if x[0].get('fix') == 'calc'][:3] + \
            [x for x in reqs if x[0]['cond'] == 'target_cue'][:3]:
        print('-----', {k: v for k, v in meta.items() if k != 'gold'}, '\n...' + prefix[-300:])
    sys.exit()

llm = LLM(MODEL, dtype='bfloat16', gpu_memory_utilization=0.85, seed=0)
sp_ = SamplingParams(n=N, temperature=0.7, top_p=0.95, max_tokens=1024)
t0 = time.time()
outs = llm.generate([{'prompt_token_ids': p} for _, p, _ in reqs], sp_)
with open(f"{OUT}/{'diff' if DIFF else 'correct'}_{SHARD}.jsonl", 'w') as f:
    for (meta, _, prefix), o in zip(reqs, outs):
        preds, oks, ntok, texts = [], [], [], []
        for c in o.outputs:
            pred, _ = parse_answer(prefix + c.text)
            ok = pred is not None and meta['gold'] is not None and abs(pred - meta['gold']) < 1e-6
            preds.append(pred); oks.append(bool(ok)); ntok.append(len(c.token_ids)); texts.append(c.text)
        f.write(json.dumps(dict(meta, prefix=prefix[-400:], preds=preds, correct=oks, new_tokens=ntok,
                                texts=texts)) + '\n')
print(f'done {len(reqs)} requests, {time.time() - t0:.0f}s', flush=True)
