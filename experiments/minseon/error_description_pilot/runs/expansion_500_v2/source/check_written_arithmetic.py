"""Conservative numeric-only equation checks. Flags, not automatic semantic labels."""
from pathlib import Path
import re,json,ast,operator
from fractions import Fraction
out=Path(__file__).resolve().parents[1]
number=r'[+-]?\d+(?:,\d{3})*(?:\.\d+)?'
pattern=re.compile(r'(?<![\w.])([+-]?\d[\d,. ()+*/×÷−-]*[+*/×÷−-][\d,. ()+*/×÷−-]*?)\s*=\s*('+number+r')(?![\w.])')
def calc(s):
 if len(s)>160:raise ValueError('too long')
 s=s.replace(',','').replace('×','*').replace('÷','/').replace('−','-').strip()
 def v(n):
  if isinstance(n,ast.Constant) and type(n.value) in [int,float]:return Fraction(str(n.value))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):return v(n.operand) if isinstance(n.op,ast.UAdd) else -v(n.operand)
  if isinstance(n,ast.BinOp) and type(n.op) in [ast.Add,ast.Sub,ast.Mult,ast.Div]:return {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}[type(n.op)](v(n.left),v(n.right))
  raise ValueError('unsupported')
 return v(ast.parse(s,mode='eval').body)
def check(text):
 results=[];offset=0
 for line in text.splitlines(keepends=True):
  for m in pattern.finditer(line):
   prefix=line[:m.start()].rstrip();suffix=line[m.end():].lstrip()
   # Reject numeric suffixes of symbolic expressions or incomplete right sides.
   if prefix and prefix[-1] in '+-*/×÷−(' :continue
   if re.search(r'(\\(?:times|cdot|div)|\b(?:times|plus|minus|over)|divided\s+by|multiplied\s+by)$',prefix):continue
   if suffix and suffix[0] in '+-*/×÷−.%':continue
   lhs,rhs=m.group(1),m.group(2)
   try:a,b=calc(lhs),calc(rhs)
   except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):continue
   results.append({'quote':m.group(0),'start':offset+m.start(),'end':offset+m.end(),'lhs':lhs,'rhs':rhs,'computed_exact':str(a),'reported_exact':str(b),'consistent':abs(float(a-b))<1e-8})
  offset+=len(line)
 return results
if __name__=='__main__':
 assert check('80 - 16 = 40')[0]['consistent'] is False
 assert check('32 × 12 = 384')[0]['consistent'] is True
 assert check('x + 3 * 4 = 24')==[]
 assert check('(2 + 3) * 4 = 20')==[] # conservative skip, not a full equation parser
 assert check('3*4 = 12 + x')==[]
 assert check(r'\(2 \times 4 - 1 = 7\)')==[]
 text='앞부분\n80 - 16 = 40 ants\n';r=check(text)[0];assert text[r['start']:r['end']]==r['quote']
 rows=[]
 for s in map(json.loads,(out/'sample.jsonl').read_text().splitlines()):
  for r in check(s['response']):
   assert s['response'][r['start']:r['end']]==r['quote'];rows.append({'source_id':s['id'],**r})
 (out/'written_arithmetic_checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));summary={'checked_equations':len(rows),'mismatched_equations':sum(not r['consistent'] for r in rows),'flagged_sources':sorted({r['source_id'] for r in rows if not r['consistent']}),'limitations':'Only conservatively parsed numeric equations; rounding, quoted counterexamples, units, omitted multipliers and story interpretation still require review. Consistent equations do not prove a correct solution.'};(out/'written_arithmetic_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
