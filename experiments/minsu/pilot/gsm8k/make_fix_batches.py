"""Sample train-split wrong answers for the correction-type experiment and build their class + location audit batches.

Sample: N_ITEMS train problems with a wrong answer (format ok), one wrong answer each, all rule types (NOEQ too).
Batches as in make_cls_batches.py (same rubric, RUBRIC_CLS.md): every non-empty line numbered [L1], [L2], ...; rule class
and location NOT shown. N_CHUNKS chunks; the N_SHARED shared items go into the first N_SHARED_CHUNKS chunks.
usage: python -I make_fix_batches.py <gen_dir> <errors.jsonl from the release> <auto_types.jsonl> <out_dir>
writes <out_dir>/sample_ids.json, chunk_k.txt, chunk_k_ids.txt, shared_ids.json, rule_labels.json
"""
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equation_spans


def numbered(text):  # as in make_cls_batches.py
    """text with [L#] at the start of each non-empty line, and the L# of each raw line index (None if empty)"""
    out, lab, n = [], [], 0
    for line in text.rstrip().split('\n'):
        if line.strip():
            n += 1
            out.append(f'[L{n}] {line}')
            lab.append(n)
        else:
            out.append(line)
            lab.append(None)
    return '\n'.join(out), lab


GEN, ERRORS, TYPES, OUT = sys.argv[1:5]
N_ITEMS, N_CHUNKS, N_SHARED_CHUNKS, N_SHARED = 1000, 8, 4, 30
CLS = {'ARITH': 'CALC', 'SETUP': 'SETUP', 'NEWNUM': 'SETUP', 'TARGET': 'TARGET', 'OTHER': 'OTHER', 'NOEQ': 'NOEQ'}
rng = random.Random(20261112)
rows = {(r['split'], r['pid'], r['sample']): r for f in glob.glob(f'{GEN}/train_*.jsonl') for r in map(json.loads, open(f))}
ref = {e['pid']: e['reference'] for e in map(json.loads, open(ERRORS))}
iid = lambda t: f"{t['split']}-{t['pid']}-{t['sample']}"
types = {(t['split'], t['pid'], t['sample']): t for t in map(json.loads, open(TYPES)) if t['split'] == 'train'}
wrong = sorted(k for k, r in rows.items() if not r['correct'] and r['format_ok'])
assert all(k in types for k in wrong)
by_pid = {}
for k in wrong:
    by_pid.setdefault(k[1], []).append(k)
pids = rng.sample(sorted(by_pid), N_ITEMS)
sample = [types[rng.choice(by_pid[p])] for p in pids]


def item(t):
    r = rows[(t['split'], t['pid'], t['sample'])]
    right = [x for (s, p, _), x in rows.items() if s == t['split'] and p == t['pid'] and x['correct'] and x['format_ok']]
    gold_ref = re.sub(r'<<[^>]*>>', '', ref[t['pid']]).replace('\n', ' | ')
    if right:
        corr = rng.choice(right)['text'].split('####')[0].strip()
        corr = corr if len(corr) <= 1000 else corr[:1000] + ' ...'
    else:
        corr = '(none: the model never solved this problem correctly)'
    body, _ = numbered(r['text'])
    return (f"=== ITEM id={iid(t)} ===\nQUESTION: {r['question']}\n"
            f"GOLD ANSWER: {t['gold']:g} | WRONG FINAL ANSWER: {t['pred']:g}\n"
            f"GSM8K REFERENCE SOLUTION: {gold_ref}\n"
            f"--- CORRECT SOLUTION (same model, another sample) ---\n{corr}\n"
            f"--- WRONG SOLUTION (lines numbered) ---\n{body}\n\n")


def rule_info(t):
    """rule class, the L# of each checkable equation, the rule's first wrong line (TARGET / OTHER / NOEQ: the answer line)"""
    text = rows[(t['split'], t['pid'], t['sample'])]['text']
    _, lab = numbered(text)
    sp = equation_spans(text)
    eq_lines = [lab[text.count('\n', 0, eqc)] for _, eqc, _, _ in sp] if sp else []
    n_lines = max(x for x in lab if x)
    ans = [lab[j] for j, l in enumerate(text.rstrip().split('\n')) if '####' in l]
    line = eq_lines[t['k']] if t['k'] is not None and t['k'] < len(eq_lines) else (ans[-1] if ans else n_lines)
    return dict(cls=CLS[t['cat']], cat=t['cat'], k=t['k'], n_eq=t['n_eq'], eq_lines=eq_lines, rule_line=line,
                n_lines=n_lines)


order = sample[:]
rng.shuffle(order)
shared, rest = order[:N_SHARED], order[N_SHARED:]
size = (len(rest) + N_SHARED * N_SHARED_CHUNKS) / N_CHUNKS
n_first = round(size) - N_SHARED
cuts = [n_first] * N_SHARED_CHUNKS
left = len(rest) - sum(cuts)
cuts += [left // (N_CHUNKS - N_SHARED_CHUNKS) + (j < left % (N_CHUNKS - N_SHARED_CHUNKS))
         for j in range(N_CHUNKS - N_SHARED_CHUNKS)]
os.makedirs(OUT, exist_ok=True)
pos = 0
for k, c in enumerate(cuts):
    chunk = rest[pos:pos + c] + (shared if k < N_SHARED_CHUNKS else [])
    pos += c
    rng.shuffle(chunk)
    with open(f'{OUT}/chunk_{k}.txt', 'w') as f, open(f'{OUT}/chunk_{k}_ids.txt', 'w') as g:
        for t in chunk:
            f.write(item(t)); g.write(iid(t) + '\n')
assert pos == len(rest)
json.dump(sorted(iid(t) for t in sample), open(f'{OUT}/sample_ids.json', 'w'))
json.dump(sorted(iid(t) for t in shared), open(f'{OUT}/shared_ids.json', 'w'))
json.dump({iid(t): rule_info(t) for t in sample}, open(f'{OUT}/rule_labels.json', 'w'))
cnt = {c: sum(t['cat'] == c for t in sample) for c in ('SETUP', 'NOEQ', 'TARGET', 'OTHER', 'ARITH', 'NEWNUM')}
print(f'items {len(sample)} {cnt}; chunks: {[c + (N_SHARED if k < N_SHARED_CHUNKS else 0) for k, c in enumerate(cuts)]}')
