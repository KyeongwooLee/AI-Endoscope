"""Let ANOTHER model read the same question + solution text and save its hidden states (external-reader control).

If a reader that did not write the solution predicts the answer-distribution pattern as well as the writer's own
hidden state, the signal is in the text, not self-knowledge.
Input to the reader: same system + user prompt as generate.py (its own chat template), the writer's solution as the
assistant turn. Saved per solution: q_end (last prompt token), r_mean (mean over reasoning tokens, i.e. the text
before the final '####' line, as in extract_gen.py) and ans (last token of the final answer number; last token when
there is none) at 7 evenly spaced layers (k/7 of the depth, k = 1..7).
usage: python extract_reader.py <model> <gen_dir> <out_dir> <fix_dir> [<fix_dir> ...]
   ids = every wrong answer with restart rows in the fix dirs; a single fix_dir 'greedy' = every train greedy answer
"""
import glob, json, os, re, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL, GEN, OUT, FIXES = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
BS = int(os.environ.get('BS', 8))
SYSTEM = ('You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer '
          'with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write '
          'nothing after this final line. Do not omit this line. The final answer must contain only a number, without '
          'units, currency symbols, Markdown emphasis, or LaTeX.')
USER = ('Problem:\n{question}\n\nSolve the problem step by step. After your reasoning, write the answer on the LAST '
        'line using exactly this format:\n#### <number>\nReplace <number> with the numerical answer. Do not write any '
        'text after that line.')
os.makedirs(OUT, exist_ok=True)

ids = set()
for D in ([] if FIXES == ['greedy'] else FIXES):
    for f in sorted(glob.glob(f'{D}/fix_*.jsonl')):
        ids |= {r['id'] for r in map(json.loads, open(f)) if r['cond'] == 'restart'}
gen = {}
for f in glob.glob(f'{GEN}/train_*.jsonl'):
    for r in map(json.loads, open(f)):
        k = f"{r['split']}-{r['pid']}-{r['sample']}"
        if k in ids or (FIXES == ['greedy'] and r['mode'] == 'greedy'):
            gen[k] = r
ids = ids or set(gen)
ids = sorted(ids)
print('solutions', len(ids), 'missing', len([i for i in ids if i not in gen]), flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).cuda().eval()
NL = model.config.num_hidden_layers
LAYERS = [round(NL * k / 7) for k in range(1, 8)]
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def encode(r):
    m = None
    for m in re.finditer(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', r['text']):
        pass
    reason = r['text'][:m.start()].rstrip() if m else r['text']
    if not reason:  # ~5% answer first ('#### 20' then the reasoning): extract_gen.py then pools the answer tokens
        reason = r['text'][:m.end()]
    upto = r['text'][:m.end()] if m else r['text']
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER.format(question=r['question'])}]
    prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p = tok(prefix, add_special_tokens=False)['input_ids']
    R = len(tok(prefix + reason, add_special_tokens=False)['input_ids'])
    full = tok(prefix + (upto if len(upto) >= len(reason) else reason), add_special_tokens=False)['input_ids']
    return full, len(p), max(R, len(p) + 1)


enc = [encode(gen[i]) for i in ids]
order = sorted(range(len(ids)), key=lambda j: len(enc[j][0]))
H = np.zeros((len(ids), 3, len(LAYERS), model.config.hidden_size), dtype=np.float16)
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), BS):
        js = order[b:b + BS]
        L = max(len(enc[j][0]) for j in js)
        x = torch.full((len(js), L), pad, dtype=torch.long)
        a = torch.zeros((len(js), L), dtype=torch.long)
        for k, j in enumerate(js):
            s = enc[j][0]
            x[k, :len(s)] = torch.tensor(s); a[k, :len(s)] = 1
        hs = model.model(x.cuda(), attention_mask=a.cuda(), output_hidden_states=True).hidden_states
        for k, j in enumerate(js):
            s, P, R = enc[j]
            for li, l in enumerate(LAYERS):
                H[j, 0, li] = hs[l][k, P - 1].float().cpu().numpy()
                H[j, 1, li] = hs[l][k, P:R].float().mean(0).cpu().numpy()
                H[j, 2, li] = hs[l][k, len(s) - 1].float().cpu().numpy()
        if (b // BS) % 50 == 0:
            print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)
np.save(f'{OUT}/H.npy', H)
json.dump(dict(model=MODEL, ids=ids, layers=LAYERS, positions=['q_end', 'r_mean', 'ans']), open(f'{OUT}/meta.json', 'w'))
print('done', len(ids), f'{time.time() - t0:.0f}s', flush=True)
