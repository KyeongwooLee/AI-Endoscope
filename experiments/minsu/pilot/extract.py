"""Pilot dry-run: teacher-force EIC (original, transformed) solution pairs and save features.

Positions per solution:
  div : first token where original/transformed token sequences diverge
  end : last token of the wrong step line
Saved per pair: hidden states (all layers) at div/end for both solutions, plus logit-based scalars.
"""
import json, glob, re, sys, os, time
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
    """char spans of non-empty, non-#### lines"""
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
                                 trans=r['transformed_solution'], step=d))
rows = rows[:int(os.environ['LIMIT'])] if os.environ.get('LIMIT') else rows
print('pairs', len(rows), flush=True)


def encode(q, sol):
    prefix = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
    enc = tok(prefix + sol, return_offsets_mapping=True, add_special_tokens=False)
    return enc['input_ids'], enc['offset_mapping'], len(prefix)


@torch.no_grad()
def run(ids):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0].float()          # (L+1, T, D)
    logp = torch.log_softmax(out.logits[0].float(), -1)             # (T, V)
    return hs, logp


def step_end_tok(offs, plen, sol, k):
    s, e = step_spans(sol)[k]
    s, e = s + plen, e + plen
    idx = [i for i, (a, b) in enumerate(offs) if a < e and b > s]
    return idx[-1]


H = {k: [] for k in ['o_div', 't_div', 'o_end', 't_end']}
meta = []
t0 = time.time()
for n, r in enumerate(rows):
    io, oo, po = encode(r['q'], r['orig'])
    it, ot, pt = encode(r['q'], r['trans'])
    div = next(i for i, (a, b) in enumerate(zip(io, it)) if a != b)
    eo, et = step_end_tok(oo, po, r['orig'], r['step']), step_end_tok(ot, pt, r['trans'], r['step'])
    eo, et = max(eo, div), max(et, div)
    hs_o, lp_o = run(io)
    hs_t, lp_t = run(it)
    if not (torch.isfinite(hs_o).all() and torch.isfinite(hs_t).all()):
        print('non-finite at', n, flush=True)
    H['o_div'].append(hs_o[:, div].half().cpu().numpy()); H['t_div'].append(hs_t[:, div].half().cpu().numpy())
    H['o_end'].append(hs_o[:, eo].half().cpu().numpy()); H['t_end'].append(hs_t[:, et].half().cpu().numpy())
    pre = lp_o[div - 1]                                    # identical context for both: predicts the divergent token
    top2 = pre.topk(2).values
    p_ent = -(pre.exp() * pre).sum().item()

    def step_nll(lp, ids, offs, plen, sol, k):
        s, e = step_spans(sol)[k]
        idx = [i for i, (a, b) in enumerate(offs) if a < e + plen and b > s + plen and i > 0]
        v = [-lp[i - 1, ids[i]].item() for i in idx]
        return float(np.mean(v)), float(np.max(v))

    sol_start = lambda offs, plen: next(i for i, (a, b) in enumerate(offs) if a >= plen)
    so, st = sol_start(oo, po), sol_start(ot, pt)
    # teacher-forced top-1 agreement over the whole original solution (memorization indicator)
    pred = lp_o[so - 1:len(io) - 1].argmax(-1).cpu().numpy()
    top1_acc = float((pred == np.array(io[so:])).mean())
    meta.append(dict(ds=r['ds'], type=r['type'], q=r['q'], step=r['step'],
                     div=div, lp_orig_tok=pre[io[div]].item(), lp_trans_tok=pre[it[div]].item(),
                     div_entropy=p_ent, div_margin=(top2[0] - top2[1]).item(),
                     t_step_nll_mean=step_nll(lp_t, it, ot, pt, r['trans'], r['step'])[0],
                     t_step_nll_max=step_nll(lp_t, it, ot, pt, r['trans'], r['step'])[1],
                     o_step_nll_mean=step_nll(lp_o, io, oo, po, r['orig'], r['step'])[0],
                     o_step_nll_max=step_nll(lp_o, io, oo, po, r['orig'], r['step'])[1],
                     orig_top1_acc=top1_acc, orig_tok=tok.decode([io[div]]), trans_tok=tok.decode([it[div]])))
    if n % 100 == 0:
        print(f'{n}/{len(rows)} {time.time() - t0:.0f}s', flush=True)

for k, v in H.items():
    np.save(f'{OUT}/{k}.npy', np.stack(v))
json.dump(meta, open(f'{OUT}/meta.json', 'w'))
print('done', time.time() - t0, flush=True)
