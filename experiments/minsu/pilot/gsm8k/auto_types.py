"""Rule-based error types for the model's own wrong GSM8K solutions (no human/LLM judgment).

Valid values V = numbers in the question + gold intermediates (<<expr=val>>) + values computed in the model's own
correct solutions of the same problem. Walk the wrong solution's equations in order; the first equation whose result
is not in V is the first observable error ("divergence").
  NOEQ     no checkable equation (answer only / prose)                           -> undecidable
  ARITH    divergent equation is arithmetically wrong                            -> calculation (C)
  NEWNUM   divergent equation uses a number found nowhere (question, own steps)  -> misread / invented number (A part)
  SETUP    divergent equation is a correct computation of some other quantity    -> setup, A/B undecided
  TARGET   no divergence, final answer equals a computed valid intermediate      -> stopped early / wrong target (B part)
  OTHER    no divergence, final answer not explained by the equations
Operator-swap / missing-term / extra-term variants were tried and dropped: on 8-sample checks most matches were
coincidences (e.g. 'x - x', '/ 1'), so those cases stay in SETUP.
usage: python -I auto_types.py <gen_dir> <gsm8k_dir> <out.jsonl>   (also writes <out>.examples.txt)
"""
import ast, glob, json, math, os, random, re, sys, collections

WORDS = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9, ten=10, eleven=11, twelve=12,
             thirteen=13, fourteen=14, fifteen=15, sixteen=16, seventeen=17, eighteen=18, nineteen=19, twenty=20,
             thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90, hundred=100, thousand=1000,
             dozen=12, half=0.5, twice=2, double=2, triple=3, thrice=3, quarter=0.25, third=1 / 3,
             dozens=12, halves=0.5, halved=2, doubles=2, doubled=2, triples=3, tripled=3, quadruple=4,
             quadrupled=4, quarters=0.25, fourth=0.25, fourths=0.25, thirds=1 / 3, fifth=0.2, fifths=0.2, tenth=0.1)
CONST = {0.5, 1, 2, 3, 4, 7, 8, 10, 12, 16, 24, 30, 31, 52, 60, 100, 365, 1000, 3600, 5280}  # incl. unit factors
OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a / b}


def ev(n):
    if isinstance(n, ast.Expression): return ev(n.body)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
    if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left), ev(n.right))
    raise ValueError


def evaluate(e):
    try:
        return ev(ast.parse(e, mode='eval'))
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError, OverflowError):
        return None


def close(a, b):
    """equal up to the model's rounding (incl. rounding up/down to whole units)"""
    if abs(a - b) <= 1e-3 * max(1, abs(b)): return True
    return abs(a) < 1e12 and b in (math.floor(a), math.ceil(a))


key = lambda v: round(v, 2)


def normalize(text):
    t = re.sub(r'<<[^>]*>>', '', text.split('####')[0])
    t = re.sub(r'\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', t)
    t = re.sub(r'\\text\{[^{}]*\}', ' ', t)
    for a, b in [('\\times', '*'), ('\\cdot', '*'), ('\\div', '/'), ('×', '*'), ('÷', '/'), ('−', '-')]:
        t = t.replace(a, b)
    t = re.sub(r'\*\*|\$|\\', ' ', t)
    t = re.sub(r'(?<=\d),(?=\d{3})', '', t)
    t = re.sub(r'(\d+(?:\.\d+)?)\s*%\s*of\b', r'(\1/100) *', t)
    t = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', t)
    t = re.sub(r'(?<=\d)\s*[xX]\s*(?=[\d(])', '*', t)
    return re.sub(r'\n\s*=', ' =', t)  # continuation lines '= 12 + 6'


def trim_parens(e):
    while e.count('(') > e.count(')') and e.startswith('('): e = e[1:].strip()
    while e.count(')') > e.count('(') and e.endswith(')'): e = e[:-1].strip()
    return e


# number | unit word (incl. 'chores/day', 'first-day') | per-unit slash | operator/paren | anything else (stops the run)
TOK = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z'’]*(?:[/\-][A-Za-z]+)*|/(?=\s*[A-Za-z])|[+\-*/()]|\S")
WORD_OP = {'times': '*', 'plus': '+', 'minus': '-'}


def run(seg, backward):
    """arithmetic expression at the end (backward) or start of seg; unit words between numbers are skipped:
    '100 customers on the first day + 50 customers' -> '100 + 50'"""
    toks = TOK.findall(seg)
    if backward: toks.reverse()
    elif toks and re.match(r'[A-Za-z]', toks[0]) and toks[0].lower() not in WORD_OP: return ''
    out, last = [], None
    for t in toks:
        t = WORD_OP.get(t.lower(), t)
        if len(t) == 1 and t.isalpha() and t not in 'aAI': return ''  # algebra with a variable (2x, P): not checkable
        if re.fullmatch(r'\d+(?:\.\d+)?', t):
            if last == 'num': break
            out.append(t); last = 'num'
        elif t in ('+', '-', '*', '/', '(', ')'):
            out.append(t); last = 'op' if t not in '()' else last
        elif t == '/' or re.match(r'[A-Za-z]', t):
            continue
        else:
            break
    if backward:
        out.reverse()
        return trim_parens(re.sub(r'^[\s+\-*/]+|[\s+\-*/]+$', '', ' '.join(out)))  # leading '-' is a list bullet
    return trim_parens(re.sub(r'^[\s+*/]+|[\s+\-*/]+$', '', ' '.join(out)))  # keep a negative result


tail = lambda s: run(s, True)
head = lambda s: run(s, False)
HAS_OP = re.compile(r'[\d)]\s*[+\-*/]\s*[\d(]')


def equations(text):
    """[(lhs, lhs_val, rhs_val, line, line_no, pair_no)] for each 'expr = number' pair whose left side contains an operator.
    Right sides that are expressions ('= 12 + 6', '= (20/100)') are not compared: they are mostly restatements."""
    out = []
    for i, line in enumerate(normalize(text).split('\n')):
        segs = line.split('=')
        for j, (a, b) in enumerate(zip(segs, segs[1:])):
            lhs, rhs = tail(a), head(b)
            if not HAS_OP.search(lhs) or HAS_OP.search(rhs): continue
            lv, rv = evaluate(lhs), evaluate(rhs)
            if lv is not None and rv is not None:
                out.append((lhs, lv, rv, line.strip()[:200], i, j))
    return out


NUM = re.compile(r'-?\d[\d,]*(?:\.\d+)?')


def equation_spans(text):
    """[(line_start, eq_char, num_start, num_end)] in the raw text for each equation, in the order of equations(text);
    None if any equation cannot be found"""
    eqs = equations(text)
    body = text.split('####')[0]
    masked = re.sub(r'<<[^>]*>>', lambda m: ' ' * len(m.group()), body)
    # raw lines, joined like normalize() joins continuation lines ('\n   = 12'); '$', '\', '**' are blanked there first
    starts = [0] + [m.start() + 1 for m in re.finditer(r'\n', masked) if not re.match(r'\n[\s$\\*]*=', masked[m.start():])]
    if len(starts) != len(normalize(text).split('\n')):
        return None
    bounds = list(zip(starts, starts[1:] + [len(masked)]))
    out = []
    for lhs, lv, rv, line, li, j in eqs:
        a, b = bounds[li]
        eq_pos = [m.start() + a for m in re.finditer('=', masked[a:b])]
        if j >= len(eq_pos):
            return None
        m = NUM.search(masked, eq_pos[j] + 1, b)
        if not m or key(float(m.group().replace(',', ''))) != key(rv):
            return None
        out.append((a, eq_pos[j], m.start(), m.end()))
    return out


def question_numbers(q):
    q = q.replace(',', '')
    vals = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', q)]
    vals += [float(a) / float(b) for a, b in re.findall(r'(\d+)/(\d+)', q) if float(b)]
    pct = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)\s*(?:%|percent)', q)]
    vals += [v for p in pct for v in (p / 100, 100 - p, 1 - p / 100, 1 + p / 100)]  # '30% off' -> 0.3, 70, 0.7, 1.3
    vals += [WORDS[w] for w in re.findall(r'[a-z]+', q.lower()) if w in WORDS]
    if re.search(r'and a half', q.lower()):
        vals += [v + 0.5 for v in vals] + [1.5]  # 'an hour and a half'
    return vals


def operands(e):
    return [float(x) for x in re.findall(r'\d+(?:\.\d+)?', e)]


def main(GEN, DATA, OUT):
    rows = [json.loads(l) for f in sorted(glob.glob(f'{GEN}/*.jsonl')) for l in open(f) if l.strip()]
    rows = [r for r in rows if r['format_ok']]
    ref = {s: [json.loads(l)['answer'] for l in open(f'{DATA}/{s}.jsonl')] for s in ('train', 'test')}
    byq = collections.defaultdict(list)
    for r in rows:
        byq[(r['split'], r['pid'])].append(r)

    res, cons = [], collections.Counter()
    for (split, pid), sols in byq.items():
        wrong = [r for r in sols if not r['correct']]
        if not wrong: continue
        right = [r for r in sols if r['correct']]
        preds = collections.Counter(r['pred'] for r in wrong)
        cons['mixed' if right else 'always_wrong_same' if preds.most_common(1)[0][1] >= len(sols) - 1 else
             'always_wrong_varied'] += 1
        gold_ref = ref[split][int(pid)]
        q = sols[0]['question']
        qn = question_numbers(q)
        inter = [float(v) for v in re.findall(r'<<[^=>]*=\s*(-?[\d.]+)\s*>>', gold_ref) if evaluate(v) is not None]
        for r in right:
            inter += [x for lhs, lv, rv, *_ in equations(r['text']) if close(lv, rv) for x in (lv, rv)]
        V = {key(v) for v in qn + inter} | {key(sols[0]['gold'])}
        inV = lambda v: key(v) in V
        computed = {key(v) for v in inter}
        for r in wrong:
            eqs = equations(r['text'])
            k = next((i for i, e in enumerate(eqs) if not inV(e[2])), None)
            if not eqs:
                cat = 'NOEQ'
            elif k is None:
                p = r['pred']
                cat = 'TARGET' if p is not None and abs(p) > 2 and key(p) in computed else 'OTHER'
            else:
                lhs, lv, rv, line, li, j = eqs[k]
                earlier = '\n'.join(normalize(r['text']).split('\n')[:li])  # numbers the model already wrote down
                known = V | {key(v) for e in eqs[:k] for v in e[1:3]} | {key(c) for c in CONST} | \
                        {key(float(x)) for x in re.findall(r'\d+(?:\.\d+)?', earlier)}
                if not close(lv, rv):
                    cat = 'ARITH'
                elif j == 0 and any(key(n) not in known for n in operands(lhs)):  # later pairs of a chain reuse derived values
                    cat = 'NEWNUM'
                else:
                    cat = 'SETUP'
            res.append(dict(split=split, pid=pid, sample=r['sample'], mode=r['mode'], cat=cat, has_right=bool(right),
                            n_eq=len(eqs), k=k, line=eqs[k][3] if k is not None else None, pred=r['pred'],
                            gold=r['gold']))

    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'w') as f:
        for x in res: f.write(json.dumps(x) + '\n')

    ORDER = ['ARITH', 'NEWNUM', 'SETUP', 'TARGET', 'OTHER', 'NOEQ']
    print(f'solutions={len(rows)} problems={len(byq)} wrong={len(res)}')
    print('problems with >=1 wrong, by resampling behaviour:', dict(cons))
    for name, sub in [('all', res), ('has correct sibling', [x for x in res if x['has_right']]),
                      ('always wrong', [x for x in res if not x['has_right']])]:
        c = collections.Counter(x['cat'] for x in sub)
        print(f'\n## {name} (n={len(sub)})')
        print('  ' + '  '.join(f'{k}={c[k]} ({c[k] / len(sub):.0%})' for k in ORDER))

    text = {(r['split'], r['pid'], r['sample']): r for r in rows}
    rng = random.Random(0)
    with open(OUT + '.examples.txt', 'w') as f:
        for cat in ORDER:
            xs = [x for x in res if x['cat'] == cat]
            for x in rng.sample(xs, min(8, len(xs))):
                r = text[(x['split'], x['pid'], x['sample'])]
                f.write(f"\n######## {cat} | {x['split']}-{x['pid']}-{x['sample']} | pred={x['pred']} gold={x['gold']} "
                        f"| has_right={x['has_right']}\nQ: {r['question']}\n")
                f.write('GOLD: ' + ref[x['split']][int(x['pid'])].replace('\n', ' | ') + '\n')
                if x['line']: f.write(f'DIVERGENT LINE: {x["line"]}\n')
                f.write('RESPONSE:\n' + r['text'][:1200] + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:4])
