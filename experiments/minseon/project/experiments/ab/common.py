"""Reference-free parsing and safe, conservative calculation extraction."""
import ast, operator, re
from decimal import Decimal, InvalidOperation
NUM = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
def normalize(value):
    try:
        d=Decimal(str(value).replace(',','').replace('$','').strip())
        return str(d.normalize()) if d.is_finite() else None
    except (InvalidOperation, ValueError): return None

def answer(text, truncated=False):
    # Predeclared GSM8K last-number fallback; never consult a reference.
    tagged=list(re.finditer(r'(?:Final answer\s*:|####)\s*(?:\$\s*)?('+NUM+r')',text,re.I))
    boxed=list(re.finditer(r'\\boxed\{\s*('+NUM+r')\s*\}',text))
    if tagged: return normalize(tagged[-1][1]), 'final_marker'
    if boxed: return normalize(boxed[-1][1]), 'boxed'
    if truncated: return None, 'truncated_without_final'
    tail=next((line.strip() for line in reversed(text.splitlines()) if line.strip()),'')
    # A concluding verb can identify the answer before trailing incidental quantities.
    clauses=list(re.finditer(r'\b(?:is|are|was|were|equals?|needs?|takes?|paid|pay|costs?|owes?)\s+(?:(?:approximately|about|exactly|a total of)\s+)?[*$\\( ]*('+NUM+r')',tail,re.I))
    if clauses: return normalize(clauses[-1][1]), 'concluding_clause'
    nums=list(re.finditer(NUM,text))
    return (normalize(nums[-1][0]),'last_number') if nums else (None,'missing')

def strict_answer(text):
    m=list(re.finditer(r'^Final answer:\s*\$?\s*('+NUM+r')\s*\.?\s*$',text,re.M|re.I))
    return normalize(m[0][1]) if len(m)==1 and not text[m[0].end():].strip() else None

def correct(pred,gold):
    return pred is not None and Decimal(pred)==Decimal(gold)

def verdict(text):
    s=text.strip().upper()
    return s if s in ('CORRECT','INCORRECT') else 'INVALID'

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
def calculate(expr):
    if len(expr)>250: raise ValueError('expression too long')
    tree=ast.parse(expr,mode='eval')
    if sum(1 for _ in ast.walk(tree))>80: raise ValueError('too many nodes')
    def walk(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): return Decimal(str(n.value))
        if isinstance(n,ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](walk(n.left),walk(n.right))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): return walk(n.operand)*(-1 if isinstance(n.op,ast.USub) else 1)
        raise ValueError('not arithmetic')
    return normalize(walk(tree.body))

def recalculate(text, truncated=False):
    """Only execute an explicit final numeric equality matching the model's answer.
    No reference, symbolic inference, arbitrary code, or LLM extraction is used.
    """
    pred,_=answer(text,truncated)
    if pred is None: return {'available':False,'expression':None,'answer':None,'reason':'no_final_answer'}
    s=text.replace('−','-').replace('×','*').replace('÷','/').replace('\\times','*').replace('\\cdot','*').replace('\\div','/')
    s=s.replace('\\$','').replace('$','')
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=re.sub(r'\\(?:d?frac)\{([\d.]+)\}\{([\d.]+)\}',r'(\1/\2)',s)
    pattern=r'(?<![\w.])([\d.()+*/ \t-]+)\s*=\s*('+NUM+r')'
    candidates=[]
    for m in re.finditer(pattern,s):
        expr=m[1].strip()
        head=s[s.rfind('\n',0,m.start(1))+1:m.start(1)].rstrip()
        if head and not head.endswith(('=',':',r'\[',r'\(')): continue
        if not any(c in expr for c in '+-*/') or normalize(m[2])!=pred: continue
        try:
            value=calculate(expr)
            if value is not None: candidates.append((expr,value))
        except (ValueError,SyntaxError,ArithmeticError): pass
    if not candidates: return {'available':False,'expression':None,'answer':None,'reason':'no_explicit_final_numeric_equality'}
    expr,value=candidates[-1]
    return {'available':True,'expression':expr,'answer':value,'reason':'last_numeric_equality_with_rhs_matching_model_final_answer'}
