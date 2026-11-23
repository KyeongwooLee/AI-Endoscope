"""Activation steering as a correction: re-solve with a 'correct-minus-wrong' direction added to the residual stream.

Direction (layer 20 output = hidden_states[20], the layer the probes use): for every train question with both correct and
wrong solutions among its 5 own solutions (gen_train, feat_train r_mean L20), diff_q = mean(correct) - mean(wrong);
v = mean_q diff_q. Within-question, so difficulty/topic cancel. Questions are split by pid parity and a question is
only ever steered with the vector built from the OTHER half (no leakage).
Strength alpha is in units of sigma = std of the r_mean projection onto v/|v| (per half).
Conditions: base (no steering, greedy), +v at each alpha, random direction with the same norm at each alpha.
Greedy decoding throughout, so any change in the answer comes from the intervention.
Eval set: every greedy-wrong train question + N_OK greedy-correct ones (harm check).
usage: python steer.py <feat_train_dir> <gen_greedy_dir> <out_dir> [limit_per_group]
env: CONDS=base,steer_2 (subset)  ALPHAS="1,2,4,8"  RAND_ALPHAS="2,4,8"  N_OK=400  BS=48  MAXTOK=768
"""
import glob, json, os, re, sys, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FT, GG, OUT = sys.argv[1:4]
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
ALPHAS = [float(a) for a in os.environ.get('ALPHAS', '1,2,4,8').split(',')]
RALPHAS = [float(a) for a in os.environ.get('RAND_ALPHAS', '2,4,8').split(',') if a]
N_OK, BS, MAXTOK = int(os.environ.get('N_OK', 400)), int(os.environ.get('BS', 48)), int(os.environ.get('MAXTOK', 768))
MODEL, LAYER = 'Qwen/Qwen2.5-1.5B-Instruct', 20
os.makedirs(OUT, exist_ok=True)


def to_num(s):
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def parse_answer(text):
    m = re.findall(r'####\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    if m:
        return to_num(m[-1])
    nums = re.findall(r'-?[\d,]*\.?\d+', text)
    return to_num(nums[-1]) if nums else None


# ---- steering vectors (one per half) ----
meta = json.load(open(f'{FT}/meta_0.json'))
li, pi = meta['layers'].index(LAYER), meta['positions'].index('r_mean')
H = np.load(f'{FT}/H_0.npy', mmap_mode='r')
R = np.asarray(H[:, pi, li], dtype=np.float32)
by = {}
for j, m in enumerate(meta['meta']):
    if m['split'] == 'train' and m['format_ok']:
        by.setdefault(m['pid'], {True: [], False: []})[m['correct']].append(j)
V, SIG = {}, {}
for h in (0, 1):
    qs = [q for q, d in by.items() if q % 2 == h and d[True] and d[False]]
    diff = np.stack([R[by[q][True]].mean(0) - R[by[q][False]].mean(0) for q in qs])
    v = diff.mean(0)
    u = v / np.linalg.norm(v)
    allj = [j for q, d in by.items() if q % 2 == h for j in d[True] + d[False]]
    SIG[h] = float((R[allj] @ u).std())
    V[h] = u
    print(f'half {h}: {len(qs)} mixed questions, |v| {np.linalg.norm(v):.2f}, sigma {SIG[h]:.2f}, '
          f'|v|/sigma {np.linalg.norm(v) / SIG[h]:.2f}, mean |h| {np.linalg.norm(R[allj], axis=1).mean():.1f}', flush=True)
cos = float(V[0] @ V[1])
print(f'cosine between the two halves\' directions: {cos:.3f}', flush=True)
rng = np.random.default_rng(0)
RND = {}
for h in (0, 1):
    r = rng.standard_normal(V[h].shape).astype(np.float32)
    r -= (r @ V[h]) * V[h]
    RND[h] = r / np.linalg.norm(r)

# ---- eval set ----
rows = sorted((r for f in glob.glob(f'{GG}/*.jsonl') for r in map(json.loads, open(f))), key=lambda r: r['pid'])
wrong = [r for r in rows if not r['correct']]
okr = [r for r in rows if r['correct']]
okr = [okr[i] for i in sorted(np.random.default_rng(1).choice(len(okr), N_OK, replace=False))]
if LIMIT:
    wrong, okr = wrong[:LIMIT], okr[:LIMIT]
evals = [('wrong', r) for r in wrong] + [('ok', r) for r in okr]
print(f'eval: {len(wrong)} greedy-wrong + {len(okr)} greedy-correct questions', flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
layer = model.model.layers[LAYER - 1]          # its output is hidden_states[LAYER]
ADD = {'vec': None}


def hook(mod, inp, out):
    if ADD['vec'] is None:
        return out
    if isinstance(out, tuple):
        return (out[0] + ADD['vec'],) + out[1:]
    return out + ADD['vec']


layer.register_forward_hook(hook)
eos = model.generation_config.eos_token_id


def solve(items, vec_of):
    """items: list of (group, row); vec_of(row) -> np vector or None. Returns list of dicts."""
    res = [None] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i][1]['prompt_ids']))
    # group by identical vector (half), so one vector per batch
    for h in (0, 1):
        idx = [i for i in order if items[i][1]['pid'] % 2 == h]
        vec = vec_of(h)
        ADD['vec'] = None if vec is None else torch.tensor(vec, dtype=torch.bfloat16, device='cuda')
        for b in range(0, len(idx), BS):
            bi = idx[b:b + BS]
            seqs = [items[i][1]['prompt_ids'] for i in bi]
            L = max(map(len, seqs))
            ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros_like(ids)
            for k, s in enumerate(seqs):
                ids[k, L - len(s):] = torch.tensor(s); att[k, L - len(s):] = 1
            with torch.no_grad():
                out = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAXTOK,
                                     do_sample=False, eos_token_id=eos, pad_token_id=tok.pad_token_id)
            for k, i in enumerate(bi):
                text = tok.decode(out[k, L:], skip_special_tokens=True)
                g, r = items[i]
                pred = parse_answer(text)
                res[i] = dict(pid=r['pid'], group=g, gold=r['gold'], orig_pred=r['pred'], pred=pred,
                              correct=pred is not None and abs(pred - r['gold']) < 1e-6,
                              same_as_orig=pred is not None and r['pred'] is not None and abs(pred - r['pred']) < 1e-6,
                              text=text)
    ADD['vec'] = None
    return res


conds = [('base', None, 0.0)] + [(f'steer_{a:g}', 'v', a) for a in ALPHAS] + [(f'random_{a:g}', 'r', a) for a in RALPHAS]
ONLY = set(os.environ['CONDS'].split(',')) if os.environ.get('CONDS') else None
for name, kind, a in conds:
    if ONLY and name not in ONLY:
        continue
    path = f'{OUT}/{name}.jsonl'
    if os.path.exists(path):
        print('skip', name, flush=True); continue
    t0 = time.time()
    vec_of = (lambda h: None) if kind is None else \
        (lambda h, a=a, kind=kind: (V[h] if kind == 'v' else RND[h]) * a * SIG[h])
    # steer question of half h with the OTHER half's direction
    res = solve(evals, (lambda h, f=vec_of: f(1 - h)))
    with open(path + '.tmp', 'w') as f:
        for r in res:
            f.write(json.dumps(r) + '\n')
    os.replace(path + '.tmp', path)
    w = [r for r in res if r['group'] == 'wrong']; o = [r for r in res if r['group'] == 'ok']
    print(f'{name:12s} fixed {np.mean([r["correct"] for r in w]):.1%} of wrong (same old answer '
          f'{np.mean([r["same_as_orig"] for r in w]):.1%}) | kept {np.mean([r["correct"] for r in o]):.1%} of correct | '
          f'no answer {np.mean([r["pred"] is None for r in res]):.1%} | {time.time() - t0:.0f}s', flush=True)
