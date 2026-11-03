"""Build labeling batches for the class + location audit of the test-split wrong answers.

Pool = test-split solutions of the location experiment (keys of preds_loc2.json: rule class ARITH or SETUP/NEWNUM) +
test-split TARGET solutions (auto_types.jsonl). Every non-empty line of the wrong solution is numbered [L1], [L2], ...;
the rule's class and location are NOT shown, so labelers judge independently.
N_CHUNKS chunks; the N_SHARED shared items go into the first N_SHARED_CHUNKS chunks (agreement check).
usage: python -I make_cls_batches.py <gen_dir> <gsm8k_dir> <auto_types.jsonl> <preds_loc2.json> <out_dir>
"""
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auto_types import equation_spans

GEN, DATA, TYPES, PREDS, OUT = sys.argv[1:6]
N_CHUNKS, N_SHARED_CHUNKS, N_SHARED = 10, 5, 30
CLS = {'ARITH': 'CALC', 'SETUP': 'SETUP', 'NEWNUM': 'SETUP', 'TARGET': 'TARGET'}
rng = random.Random(20261104)
rows = {(r['split'], r['pid'], r['sample']): r for f in glob.glob(f'{GEN}/test_*.jsonl') for r in map(json.loads, open(f))}
ref = [json.loads(l)['answer'] for l in open(f'{DATA}/test.jsonl')]
located = {i for i in json.load(open(PREDS))['k'] if i.startswith('test-')}
iid = lambda t: f"{t['split']}-{t['pid']}-{t['sample']}"
types = [t for t in map(json.loads, open(TYPES))
         if t['split'] == 'test' and (iid(t) in located or t['cat'] == 'TARGET')]


def numbered(text):
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
    """rule class, the L# of each checkable equation, the rule's first wrong line (TARGET: the last line)"""
    text = rows[(t['split'], t['pid'], t['sample'])]['text']
    _, lab = numbered(text)
    sp = equation_spans(text)
    eq_lines = [lab[text.count('\n', 0, eqc)] for _, eqc, _, _ in sp] if sp else []
    n_lines = max(x for x in lab if x)
    ans = [lab[j] for j, l in enumerate(text.rstrip().split('\n')) if '####' in l]
    line = (ans[-1] if ans else n_lines) if t['cat'] == 'TARGET' else eq_lines[t['k']]
    return dict(cls=CLS[t['cat']], cat=t['cat'], k=t['k'], n_eq=t['n_eq'], eq_lines=eq_lines, rule_line=line,
                n_lines=n_lines)


rng.shuffle(types)
shared, rest = types[:N_SHARED], types[N_SHARED:]
size = (len(rest) + N_SHARED * N_SHARED_CHUNKS) / N_CHUNKS  # items per chunk, shared copies included
n_first = round(size) - N_SHARED                             # unique items in each chunk that has the shared ones
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
json.dump(sorted(iid(t) for t in shared), open(f'{OUT}/shared_ids.json', 'w'))
json.dump({iid(t): rule_info(t) for t in types}, open(f'{OUT}/rule_labels.json', 'w'))
cnt = {c: sum(CLS[t['cat']] == c for t in types) for c in ('CALC', 'TARGET', 'SETUP')}
print(f'items {len(types)} {cnt}; chunks: {[c + (N_SHARED if k < N_SHARED_CHUNKS else 0) for k, c in enumerate(cuts)]}')
