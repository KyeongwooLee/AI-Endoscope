import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,time,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
out=Path(__file__).resolve().parents[1];path='/home/coder/yaicon-errorprobe/models/Qwen3-4B-Instruct-2507';torch.set_num_threads(4);torch.manual_seed(20261112)
tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left');model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
f=out/'ai_pair_judgments.jsonl';done={json.loads(l)['pair_id'] for l in f.read_text().splitlines()} if f.exists() else set();pack=[json.loads(l) for l in (out/'judge_packets.jsonl').read_text().splitlines()];todo=[r for r in pack if r['pair_id'] not in done];todo.sort(key=lambda r:len(r['prompt']))
def rawgen(ps):
 txt=[tok.apply_chat_template([{'role':'user','content':p}],tokenize=False,add_generation_prompt=True) for p in ps];inp=tok(txt,return_tensors='pt',padding=True).to('cuda');assert inp['input_ids'].shape[1]<10000
 with torch.inference_mode():res=model.generate(**inp,max_new_tokens=200,do_sample=False,pad_token_id=tok.pad_token_id)
 return tok.batch_decode(res[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
def gen(ps):
 try:return rawgen(ps)
 except torch.OutOfMemoryError:
  if len(ps)==1:raise
 torch.cuda.empty_cache()
 mid=len(ps)//2
 return gen(ps[:mid])+gen(ps[mid:])
def parse(raw):
 s=raw.strip()
 if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0]
 obj=json.loads(s);assert obj['judgment'] in ['same','different','uncertain'];assert isinstance(obj['reason'],str) and obj['reason'];return obj
for i in range(0,len(todo),16):
 t=time.time();batch=todo[i:i+16];raw=gen([r['prompt'] for r in batch])
 for r,txt in zip(batch,raw):
  attempts=[txt]
  try:obj=parse(txt)
  except Exception:
   fix=gen([r['prompt']+'\nReturn complete valid JSON only.'])[0];attempts.append(fix)
   try:obj=parse(fix)
   except Exception:obj={'judgment':None,'reason':'generation_or_parse_failed'}
  with f.open('a') as h:h.write(json.dumps({'pair_id':r['pair_id'],'reviewer_type':'ai_provisional','model':'Qwen/Qwen3-4B-Instruct-2507','model_revision':'cdbee75f17c01a7cc42f958dc650907174af0554','fresh_context':True,'independent_model_from_annotator':False,'raw_attempts':attempts,**obj},ensure_ascii=False)+'\n')
 print('JUDGE',i+len(batch),'/',len(todo),'seconds',round(time.time()-t,1),flush=True)
print('JUDGE COMPLETE',flush=True)
