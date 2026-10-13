"""One-time pilot-derived scoring amendment before any classifier fit/test inference."""
import time
from pipeline import RUN,read,dump
from common import answer,correct,recalculate
assert not (RUN/'selection_frozen.json').exists()
changes=[]
for p in (RUN/'records').glob('*.json'):
 r=read(p);assert r['split']!='test'
 old=read(p);pred,status=answer(r['original']['decoded_output'],r['original']['truncated']);r['actions']['Keep']['answer']=pred;r['parse_status']=status
 st=time.perf_counter();calc=recalculate(r['original']['decoded_output'],r['original']['truncated']);calc.update(seconds=time.perf_counter()-st,tokens=0);r['actions']['Recalculate']=calc
 for a in ['Rebuild','Review']:
  v=r['actions'][a];v['answer']=answer(v['decoded_output'],v['truncated'])[0]
 for a,v in r['actions'].items():v['success']=int(correct(v['answer'],r['gold'])) if v['available'] else None
 r['label_error']=1-r['actions']['Keep']['success']
 if any(r['actions'][a]['answer']!=old['actions'][a]['answer'] for a in r['actions']):
  dump(RUN/'pilot_debug_v1'/'pre_conclusion_fix_records'/p.name,old);changes.append(dict(id=r['id'],split=r['split'],old_answers={a:v['answer'] for a,v in old['actions'].items()},new_answers={a:v['answer'] for a,v in r['actions'].items()}))
 dump(p,r)
dump(RUN/'pilot_conclusion_fix.json',dict(trigger='Manual pilot inspection gsm8k_train_3784: answer 5844 precedes incidental 8 napkins. General concluding-verb parser added, reference-free; applied to all collected records before any fitting or test inference.',changes=changes))
