"""Pilot 2: step-level features for every step of every solution (no oracle position).

For each clean EIC pair, both the original and transformed solutions are teacher-forced and,
for EVERY step, we save
  X_end  : hidden state of the step's last token      (all layers)
  X_mean : mean hidden state over the step's tokens    (all layers)
  nll_max / nll_mean : token-level confidence over the step
Step labels: 0 = correct (all original steps, transformed steps before the error),
             1 = first wrong step, 2 = after the first error (propagated; excluded from training)
"""
import json, glob, sys, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

EIC, OUT, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()


def load(f):
    return [json.loads(l) for l in open(f) if l.strip()]


def step_spans(sol):
    spans, pos = [], 0
    for line in sol.split('\n'):
        if line.strip() and not line.strip().startswith('####'):
            spans.append((pos, pos + len(line)))
        pos += len(line) + 1
    return spans


def first_diff(a, b):
    A = [a[s:e].strip() for s, e in step_spans(a)]
    B = [b[s:e].strip() for s, e in step_spans(b)]
    for i, (x, y) in enumerate(zip(A, B)):
        if x != y:
            return i
    return None


rows = []
for ds in ['GSM8K', 'MathQA']:
    for f in sorted(glob.glob(f'{EIC}/generated_cases_{ds}/*/*_100/*.jsonl')):
        t = f.split('/')[-3]
        for r in load(f):
            d = first_diff(r['original_solution'], r['transformed_solution'])
            if d is not None and d + 1 == r['wrong_step']:
                rows.append(dict(ds=ds, type=t, q=r['question'], orig=r['original_solution'],
                                 trans=r['transformed_solution'], err=d))
rows = rows[:int(os.environ['LIMIT'])] if os.environ.get('LIMIT') else rows
print('pairs', len(rows), flush=True)


@torch.no_grad()
def features(q, sol):
    prefix = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
    enc = tok(prefix + sol, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs, plen = enc['input_ids'], enc['offset_mapping'], len(prefix)
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0].float()                    # (L+1, T, D)
    lp = torch.log_softmax(out.logits[0].float(), -1)
    tok_lp = lp[:-1].gather(1, torch.tensor(ids[1:], device=dev)[:, None])[:, 0]  # logprob of token i+1
    res = []
    for s, e in step_spans(sol):
        idx = [i for i, (a, b) in enumerate(offs) if a < e + plen and b > s + plen]
        nll = -tok_lp[torch.tensor(idx, device=dev) - 1]
        res.append((hs[:, idx[-1]].half().cpu().numpy(), hs[:, idx].mean(1).half().cpu().numpy(),
                    nll.max().item(), nll.mean().item(), len(idx)))
    return res


X_end, X_mean, meta = [], [], []
t0 = time.time()
for pid, r in enumerate(rows):
    for sol_name, sol in [('orig', r['orig']), ('trans', r['trans'])]:
        for k, (he, hm, nmax, nmean, ntok) in enumerate(features(r['q'], sol)):
            label = 0 if sol_name == 'orig' or k < r['err'] else (1 if k == r['err'] else 2)
            X_end.append(he); X_mean.append(hm)
            meta.append(dict(pid=pid, sol=sol_name, k=k, label=label, err=r['err'], ds=r['ds'], type=r['type'],
                             q=r['q'], nll_max=nmax, nll_mean=nmean, ntok=ntok))
    if pid % 200 == 0:
        print(f'{pid}/{len(rows)} steps={len(meta)} {time.time() - t0:.0f}s', flush=True)

np.save(f'{OUT}/X_end.npy', np.stack(X_end)); np.save(f'{OUT}/X_mean.npy', np.stack(X_mean))
json.dump(meta, open(f'{OUT}/steps_meta.json', 'w'))
print('done steps', len(meta), f'{time.time() - t0:.0f}s', flush=True)
