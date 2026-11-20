"""Run the model's Python answers for fix_exp.py: no network, 5 s, 2 GB; the answer is the last number printed.

Many programs end with a bare expression (notebook style, e.g. `total_minutes`) instead of print(); that last
expression is printed, as a REPL would.
usage (re-run the code condition of saved results): python fix_code.py <fix_dir>  -> rewrites preds/correct of code rows
"""
import ast, glob, json, os, re, resource, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor


def to_num(s):  # as in generate.py
    s = s.replace(',', '').replace('$', '').strip().rstrip('.')
    try:
        return float(s)
    except ValueError:
        return None


def as_program(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    last = tree.body[-1] if tree.body else None
    if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call) and getattr(last.value.func, 'id', '') == 'print'):
        tree.body[-1] = ast.Expr(ast.Call(ast.Name('print', ast.Load()), [last.value], []))
        return ast.unparse(ast.fix_missing_locations(tree))
    return code


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_code(text, workdir):
    """last number printed by the model's program, or None"""
    m = re.findall(r'```(?:python)?\s*\n(.*?)```', text, re.S)
    code = as_program(m[0] if m else text)
    with tempfile.TemporaryDirectory(dir=workdir) as d:
        try:
            p = subprocess.run(['unshare', '-rn', 'timeout', '5', '/usr/bin/python3', '-I', '-c', code], cwd=d,
                               capture_output=True, text=True, timeout=10, env={'PATH': '/usr/bin:/bin'},
                               preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return None
    nums = re.findall(r'-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?', p.stdout)
    return to_num(nums[-1]) if nums else None


if __name__ == '__main__':
    D = sys.argv[1]
    for f in sorted(glob.glob(f'{D}/fix_*.jsonl')):
        rows = [json.loads(l) for l in open(f)]
        with ThreadPoolExecutor(32) as ex:
            for r in rows:
                if r['cond'] == 'code':
                    r['preds'] = list(ex.map(lambda t: run_code(t, D), r['texts']))
                    r['correct'] = [p is not None and abs(p - r['gold']) < 1e-6 for p in r['preds']]
        with open(f, 'w') as g:
            for r in rows:
                g.write(json.dumps(r) + '\n')
        print('rewrote', f, flush=True)
