import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
from pathlib import Path
import json,time,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
out=Path(__file__).resolve().parents[1];path='/home/coder/yaicon-errorprobe/models/Qwen3-4B-Instruct-2507';torch.set_num_threads(4);torch.manual_seed(20261112)
tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left');model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
f=out/'check_outputs.jsonl';done={json.loads(l)['id'] for l in f.read_text().splitlines()} if f.exists() else set();pack=[json.loads(l) for l in (out/'check_packets.jsonl').read_text().splitlines()];todo=[r for r in pack if r['id'] not in done];todo.sort(key=lambda r:(r['task']=='reference_free_screen',len(r['prompt'])))
def rawgen(ps,cap):
 txt=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps];inp=tok(txt,return_tensors='pt',padding=True).to('cuda');assert inp['input_ids'].shape[1]<12000
 with torch.inference_mode():res=model.generate(**inp,max_new_tokens=cap,do_sample=False,pad_token_id=tok.pad_token_id)
 return tok.batch_decode(res[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
def gen(ps,cap):
 try:return rawgen(ps,cap)
 except torch.OutOfMemoryError:
  if len(ps)==1:raise
 torch.cuda.empty_cache();mid=len(ps)//2
 return gen(ps[:mid],cap)+gen(ps[mid:],cap)
def parse(raw,task):
 s=raw.strip()
 if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0]
 obj=json.loads(s)
 if task=='reference_free_screen':
  assert obj['status'] in ['math_error','no_math_error','answer_conflict','ambiguous']
  assert all(isinstance(obj[k],str) for k in ['independent_answer','check','error_description'])
  assert obj['evidence_quote'] is None or isinstance(obj['evidence_quote'],str)
 else:assert obj['judgment'] in ['same','different','uncertain'] and isinstance(obj['reason'],str)
 return obj
for i in range(0,len(todo),16):
 t=time.time();batch=todo[i:i+16];raw=gen([r['prompt'] for r in batch],400)
 for r,txt in zip(batch,raw):
  attempts=[txt];obj=None;err=None
  try:obj=parse(txt,r['task'])
  except Exception as e:
   err=str(e);fixed=gen([r['prompt']+'\nReturn a complete valid JSON object only.'],800)[0];attempts.append(fixed)
   try:obj=parse(fixed,r['task']);err=None
   except Exception as e:err=str(e)
  row={k:v for k,v in r.items() if k!='prompt'};row.update(model='Qwen/Qwen3-4B-Instruct-2507',revision='cdbee75f17c01a7cc42f958dc650907174af0554',reviewer_type='ai_provisional',do_sample=False,raw_attempts=attempts,output=obj,parse_error=err)
  with f.open('a') as h:h.write(json.dumps(row,ensure_ascii=False)+'\n')
 print('CHECK',i+len(batch),'/',len(todo),'seconds',round(time.time()-t,1),flush=True)
print('CHECKS COMPLETE',flush=True)
