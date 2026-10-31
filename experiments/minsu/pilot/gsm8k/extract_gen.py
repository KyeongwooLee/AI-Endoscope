"""Replay the model's own GSM8K solutions (exact token ids) and save solution-level features.

Hidden states (layers LAYERS) at 4 positions:
  q_end  : last prompt token, before answering (question-only / difficulty baseline)
  r_end  : last reasoning token, before the final '####' line
  r_mean : mean over reasoning tokens
  ans    : last token of the final answer number
Confidence (final layer, over reasoning tokens): NLL mean/max, lower-tail logprob, entropy mean/max,
margin mean/min, answer-token logprob, length.
usage: python extract_gen.py <gen_dir> <out_dir> <shard> <n_shards>
"""
import json, os, re, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

GEN, OUT, SHARD, NSH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
MODEL, LAYERS, BS = 'Qwen/Qwen2.5-1.5B-Instruct', [4, 8, 12, 16, 20, 24, 28], 16
CONF = ['nll_mean', 'nll_max', 'lowtail_lp', 'ent_mean', 'ent_max', 'margin_mean', 'margin_min', 'ans_lp', 'length']
os.makedirs(OUT, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == 'cuda' else torch.float32).to(dev).eval()
special = set(tok.all_special_ids)

import glob
rows = [json.loads(l) for f in sorted(glob.glob(f'{GEN}/*_*.jsonl')) for l in open(f)]
rows = [r for i, r in enumerate(rows) if i % NSH == SHARD]
rows = rows[:int(os.environ['LIMIT'])] if os.environ.get('LIMIT') else rows
print('solutions', len(rows), flush=True)


def positions(r):
    out = r['output_ids']
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
    starts = [0] + ends[:-1]
    m = None
    for m in re.finditer(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text):
        pass
    if m:
        r_idx = [i for i in range(n) if starts[i] < m.start()]
        a_idx = [i for i in range(n) if starts[i] < m.end() and ends[i] > m.start(1)]
    else:
        r_idx, a_idx = list(range(n)), [n - 1]
    if not r_idx:
        r_idx = a_idx
    return r_idx, a_idx


H, C, META = [], [], []
order = sorted(range(len(rows)), key=lambda i: len(rows[i]['prompt_ids']) + len(rows[i]['output_ids']))
t0 = time.time()
with torch.no_grad():
    for b in range(0, len(order), BS):
        batch = [rows[i] for i in order[b:b + BS]]
        seqs = [r['prompt_ids'] + r['output_ids'] for r in batch]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :len(s)] = torch.tensor(s); att[k, :len(s)] = 1
        out = model(ids.to(dev), attention_mask=att.to(dev), output_hidden_states=True)
        hs = torch.stack([out.hidden_states[l] for l in LAYERS], 1)          # (B, nL, T, D)
        for k, r in enumerate(batch):
            P = len(r['prompt_ids'])
            r_idx, a_idx = positions(r)
            ri = torch.tensor(r_idx, device=dev) + P
            h = torch.stack([hs[k, :, P - 1], hs[k, :, P + r_idx[-1]], hs[k, :, ri].mean(1), hs[k, :, P + a_idx[-1]]], 0)
            H.append(h.half().cpu().numpy())
            lp = torch.log_softmax(out.logits[k, P - 1:P + max(r_idx[-1], a_idx[-1])].float(), -1)  # predicts output tokens
            tgt = torch.tensor(r['output_ids'][:lp.shape[0]], device=dev)
            tok_lp = lp.gather(1, tgt[:, None])[:, 0]
            ent = -(lp.exp() * lp).sum(-1)
            top2 = lp.topk(2, -1).values.exp()
            marg = top2[:, 0] - top2[:, 1]
            R = torch.tensor(r_idx, device=dev)
            rl = tok_lp[R]
            k10 = max(1, int(0.1 * len(r_idx)))
            C.append([(-rl).mean().item(), (-rl).max().item(), rl.sort().values[:k10].mean().item(),
                      ent[R].mean().item(), ent[R].max().item(), marg[R].mean().item(), marg[R].min().item(),
                      tok_lp[torch.tensor(a_idx, device=dev)].mean().item(), float(len(r_idx))])
            META.append(dict(split=r['split'], pid=r['pid'], sample=r['sample'], mode=r['mode'], correct=r['correct'],
                             format_ok=r['format_ok'], finish=r['finish']))
        if (b // BS) % 200 == 0:
            print(f'{b}/{len(order)} {time.time() - t0:.0f}s', flush=True)

np.save(f'{OUT}/H_{SHARD}.npy', np.stack(H)); np.save(f'{OUT}/C_{SHARD}.npy', np.array(C, dtype=np.float32))
json.dump(dict(meta=META, layers=LAYERS, positions=['q_end', 'r_end', 'r_mean', 'ans'], conf=CONF),
          open(f'{OUT}/meta_{SHARD}.json', 'w'))
print('done', len(META), f'{time.time() - t0:.0f}s', flush=True)
