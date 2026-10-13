"""Reference-free numerical extraction v3 with explicit spans.
V3 fixes development-collection parser regressions before classifier fitting.
The legacy safe calculator and its own eligibility rules remain unchanged.
"""
import sys,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ab'))
from common import NUM,normalize,strict_answer,correct,recalculate,answer as legacy_answer
# Capture currency-inside-sign forms while retaining the exact numerical span.
VALUE=r'([+-]?\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)'
CUE=r'\b(?:will have|will be|has|have|had|is|are|was|were|equals?|needs?|takes?|paid|pay|costs?|owes?|earns?|buys?|runs?|contains?|came to)\s+(?:(?:approximately|about|exactly|a total of|a total|total of)\s+)?[*$\\( ]*'+VALUE
MARKER=r'(?:final answer\s*(?::|is\s*:?)|####)'
def extract(text,truncated=False):
    def result(m,status,offset=0):
        a,b=m.span(1);raw=m.group(1);left=len(raw)-len(raw.lstrip());a+=left
        return dict(answer=normalize(raw),span=[a+offset,b+offset],status=status)
    tagged=list(re.finditer(MARKER+r'\s*[*$\\( ]*'+VALUE,text,re.I))
    if tagged:return result(tagged[-1],'final_marker')
    boxed=list(re.finditer(r'\\boxed\{\s*('+NUM+r')\s*\}',text))
    if boxed:return result(boxed[-1],'boxed')
    tags=list(re.finditer(MARKER,text,re.I))
    if tags:
        tag=tags[-1];rest=text[tag.end():];offset=tag.end()
        # Read the next nonempty line if the marker has no same-line content.
        line=next((m for m in re.finditer(r'[^\n]+',rest) if m[0].strip(' *\t')),None)
        if line:
            segment=line[0];offset+=line.start()
            # An unresolved symbolic expression is not a final scalar answer.
            if re.search(r'\d[A-Za-z]\b',segment) and re.search(r'[+*/]',segment):return dict(answer=None,span=None,status='symbolic_final_marker')
            cue=re.search(CUE,segment,re.I)
            if cue:return result(cue,'prose_final_marker',offset)
            m=re.search('('+NUM+')',segment)
            if m:return result(m,'prose_final_marker',offset)
        return dict(answer=None,span=None,status='nonnumeric_final_marker')
    if truncated:return dict(answer=None,span=None,status='truncated_without_final')
    lines=list(re.finditer(r'[^\n]+',text));tail=next((m for m in reversed(lines) if m[0].strip()),None)
    if tail:
        m=re.search(CUE,tail[0],re.I)
        if m:return result(m,'concluding_clause',tail.start())
    nums=list(re.finditer('('+NUM+')',text))
    return result(nums[-1],'last_number') if nums else dict(answer=None,span=None,status='missing')
