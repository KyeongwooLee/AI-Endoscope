from pathlib import Path
import json,re,ast,operator
from fractions import Fraction
out=Path(__file__).resolve().parents[1]
def calc(s):
 if len(s)>160:raise ValueError('expression too long')
 def visit(n):
  if isinstance(n,ast.Constant) and type(n.value) in [int,float]:return Fraction(str(n.value))
  if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)):
   v=visit(n.operand);return v if isinstance(n.op,ast.UAdd) else -v
  if isinstance(n,ast.BinOp) and type(n.op) in [ast.Add,ast.Sub,ast.Mult,ast.Div]:return {ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}[type(n.op)](visit(n.left),visit(n.right))
  raise ValueError('unsupported arithmetic syntax')
 return visit(ast.parse(s.replace(',','').strip(),mode='eval').body)
rows=[]
for r in map(json.loads,(out/'sample.jsonl').read_text().splitlines()):
 for fragment in re.findall(r'<<([^<>]+)>>',r['reference']):
  if '=' not in fragment:continue
  lhs,rhs=fragment.rsplit('=',1)
  try:
   actual,reported=calc(lhs),calc(rhs);delta=abs(float(actual-reported));status='consistent' if delta<1e-8 else 'rounding_or_mismatch'
   rows.append({'source_id':r['id'],'fragment':fragment,'status':status,'exact_value':str(actual),'reported_value':str(reported),'absolute_difference':delta})
  except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):rows.append({'source_id':r['id'],'fragment':fragment,'status':'not_checked'})
(out/'reference_arithmetic_checks.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
print(json.dumps([r for r in rows if r['status']=='rounding_or_mismatch'],ensure_ascii=False,indent=2));print('equations',len(rows))
