"""Replay the model's own WRONG GSM8K solutions and save features at every checkable equation (step).

Location task: which equation is the first wrong one? Steps = equations(text) from auto_types.py in the same order,
so auto_types' k (first equation whose result appears in no correct solution of the problem) is the label.
A step's text starts at its line start, or right after the previous equation's result if both are on one line.
Hidden states (LAYERS) per step:
  pre  : token just before the step's text (the state that chooses what to write next)
  eq   : the '=' token (left side written, result not yet)
  res  : last token of the result number
  mean : mean over the step's tokens up to the result
Confidence per step (final-layer logprobs): min / mean logprob over the step's tokens, logprob of the result tokens
(min / mean), entropy at the result tokens (mean), top-2 margin at the result tokens (min), entropy when choosing the
step's first token, entropy over the step's tokens (mean / max), top-2 margin over the step's tokens (min).
A solution is skipped if any of its equations cannot be aligned to the tokens.
usage: python extract_loc.py <gen_dir> <auto_types.jsonl> <out_dir> <shard> <n_shards>
"""
import bisect, glob, json, os, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equations, equation_spans

GEN, TYPES, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
MODEL, LAYERS, BS = 'Qwen/Qwen2.5-1.5B-Instruct', [8, 12, 16, 20, 24, 28], 16
CONF = ['line_lp_min', 'line_lp_mean', 'res_lp_min', 'res_lp_mean', 'res_ent_mean', 'res_margin_min',
        'start_ent', 'line_ent_mean', 'line_ent_max', 'line_margin_min']
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
special = set(tok.all_special_ids)

types = {(t['split'], t['pid'], t['sample']): t for t in map(json.loads, open(TYPES)) if t['k'] is not None}
rows = [r for f in sorted(glob.glob(f'{GEN}/*_*.jsonl')) for r in map(json.loads, open(f))
        if (r['split'], r['pid'], r['sample']) in types]
rows = [r for i, r in enumerate(rows) if i % NSH == SHARD]
rows = rows[:int(os.environ['LIMIT'])] if os.environ.get('LIMIT') else rows
print('solutions', len(rows), flush=True)


def char_spans(out):
    n = len(out)
    while n and out[n - 1] in special:
        n -= 1
    pieces = [tok.decode([t]) for t in out[:n]]
    text = ''.join(pieces)
    if text != tok.decode(out[:n]):                     # non-ASCII merge: fall back to prefix decoding
        ends = [len(tok.decode(out[:i + 1])) for i in range(n)]
        text = tok.decode(out[:n])
    else:
        ends = np.cumsum([len(p) for p in pieces]).tolist()
    return text, ends


H, META = [], []
skipped = 0
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['prompt_ids']) + len(rows[i]['output_ids']))
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), BS):
        batch = []
        for i in order[b:b + BS]:
            r = rows[i]
            text, ends = char_spans(r['output_ids'])
            st = equation_spans(text) if text == r['text'] or [e[1:3] for e in equations(text)] == \
                [e[1:3] for e in equations(r['text'])] else None
            t = types[(r['split'], r['pid'], r['sample'])]
            if st is None or len(st) != t['n_eq']:
                skipped += 1
                continue
            batch.append((r, text, ends, st, t))
        if not batch:
            continue
        seqs = [x[0]['prompt_ids'] + x[0]['output_ids'] for x in batch]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        out = model(ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
        hs = torch.stack([out.hidden_states[l] for l in LAYERS], 1)          # (B, nL, T, D)
        for k, (r, text, ends, st, t) in enumerate(batch):
            P, n = len(r['prompt_ids']), len(ends)
            ti = lambda c: bisect.bisect_right(ends, c)                        # token containing char c
            lp = torch.log_softmax(out.logits[k, P - 1:P + n - 1].float(), -1)  # row t predicts output token t
            tok_lp = lp.gather(1, torch.tensor(r['output_ids'][:n], device=dev)[:, None])[:, 0]
            ent = -(lp.exp() * lp).sum(-1)
            top2 = lp.topk(2, -1).values.exp()
            marg = top2[:, 0] - top2[:, 1]
            prev = (None, None)
            for e, (ls, eqc, ns, ne) in enumerate(st):
                s = prev[1] if prev[0] == ls else ls                            # 2nd equation on the same line
                prev = (ls, ne)
                te, tr = ti(eqc), ti(ne - 1)
                ts = min(ti(s), tr)
                line, res = slice(ts, tr + 1), slice(ti(ns), tr + 1)
                H.append(torch.stack([hs[k, :, P + ts - 1], hs[k, :, P + te], hs[k, :, P + tr],
                                      hs[k, :, P + ts:P + tr + 1].mean(1)], 0).half().cpu().numpy())
                META.append(dict(split=r['split'], pid=r['pid'], sample=r['sample'], cat=t['cat'], k=t['k'],
                                 n_eq=t['n_eq'], e=e, line=text[s:ne][-300:], eq_tok=tok.decode([r['output_ids'][te]]),
                                 res_tok=tok.decode(r['output_ids'][res]),
                                 conf=[tok_lp[line].min().item(), tok_lp[line].mean().item(), tok_lp[res].min().item(),
                                       tok_lp[res].mean().item(), ent[res].mean().item(), marg[res].min().item(),
                                       ent[ts].item(), ent[line].mean().item(), ent[line].max().item(),
                                       marg[line].min().item()]))
        if (b // BS) % 50 == 0:
            print(f'{b}/{len(order)} steps={len(META)} skipped={skipped} {time.time() - t0:.0f}s', flush=True)

np.save(f'{OUT}/H_loc_{SHARD}.npy', np.stack(H))
json.dump(dict(meta=META, layers=LAYERS, positions=['pre', 'eq', 'res', 'mean'], conf=CONF, skipped=skipped),
          open(f'{OUT}/meta_loc_{SHARD}.json', 'w'))
print('done', len(META), 'steps, skipped solutions', skipped, f'{time.time() - t0:.0f}s', flush=True)
