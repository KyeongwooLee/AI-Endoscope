import os
os.environ['USE_TF']='0';os.environ['USE_FLAX']='0';os.environ['HF_HUB_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,sys,time,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
out=Path(__file__).resolve().parents[1];base=out.parents[1];sys.path.insert(0,str(base))
from edp.annotation import validate_output
cfg=json.load(open(out/'config.json'));modelpath=Path('/home/coder/yaicon-errorprobe/models/Qwen3-4B-Instruct-2507')
torch.set_num_threads(4);torch.manual_seed(20261112)
tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,padding_side='left');model=AutoModelForCausalLM.from_pretrained(modelpath,local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();print('MODEL LOADED',flush=True)
(out/'annotation').mkdir(exist_ok=True)
def parse(s):
 s=s.strip()
 if s.startswith('```'):s=s.split('\n',1)[1].rsplit('```',1)[0]
 return json.loads(s)
def append(f,r):
 with f.open('a') as h:h.write(json.dumps(r,ensure_ascii=False)+'\n');h.flush()
def rawgenerate(batch,maxnew):
 texts=[tok.apply_chat_template([{'role':'user','content':x}],tokenize=False,add_generation_prompt=True) for x in batch];inp=tok(texts,return_tensors='pt',padding=True).to('cuda');assert inp['input_ids'].shape[1]<8000
 with torch.inference_mode():res=model.generate(**inp,max_new_tokens=maxnew,do_sample=False,pad_token_id=tok.pad_token_id)
 return tok.batch_decode(res[:,inp['input_ids'].shape[1]:],skip_special_tokens=True)
def generate(batch,maxnew):
 try:return rawgenerate(batch,maxnew)
 except torch.OutOfMemoryError:
  if len(batch)==1:raise
 torch.cuda.empty_cache()
 mid=len(batch)//2
 return generate(batch[:mid],maxnew)+generate(batch[mid:],maxnew)
for cond in ['E','G']:
 packets=[json.loads(l) for l in (out/f'packets/{cond}.jsonl').read_text().splitlines()];f=out/f'annotation/{cond}.jsonl';done={r['source_id'] for r in [json.loads(l) for l in f.read_text().splitlines()]} if f.exists() else set();todo=[p for p in packets if p['source_id'] not in done]
 # Length bucketing reduces padding; each request is independent.
 todo.sort(key=lambda p:len(p['rendered_prompt']))
 for start in range(0,len(todo),16):
  batch=todo[start:start+16];t=time.time();prompts=[p['rendered_prompt'] for p in batch];raw=generate(prompts,1000);results=[];retry=[]
  for p,txt in zip(batch,raw):
   try:obj=parse(txt);err,warn,overlay=validate_output(cond,obj,p,cfg)
   except Exception as ex:obj=None;err=['parse_error:'+str(ex)];warn=[];overlay=None
   append(out/'annotation/raw_attempts.jsonl',{'condition':cond,'source_id':p['source_id'],'attempt':1,'raw':txt,'errors':err,'warnings':warn})
   results.append([p,obj,err,warn,overlay])
   if err:retry.append(len(results)-1)
  if retry:
   fix=[results[i][0]['rendered_prompt']+'\n\nYour previous output failed validation: '+str(results[i][2])+'. Return a complete valid JSON object. Use evidence_start=null and evidence_end=null and evidence_occurrence=1 if using a quote; quote exact characters from the solution. Copy the source_id and response_sha256 exactly. Keep the description between 25 and 60 words.' for i in retry]
   fixed=generate(fix,1000)
   for i,txt in zip(retry,fixed):
    p=results[i][0]
    try:obj=parse(txt);err,warn,overlay=validate_output(cond,obj,p,cfg)
    except Exception as ex:obj=None;err=['parse_error:'+str(ex)];warn=[];overlay=None
    append(out/'annotation/raw_attempts.jsonl',{'condition':cond,'source_id':p['source_id'],'attempt':2,'raw':txt,'errors':err,'warnings':warn});results[i]=[p,obj,err,warn,overlay]
  for p,obj,err,warn,overlay in results:
   append(f,{'source_id':p['source_id'],'condition':cond,'response_sha256':p['response_sha256'],'valid':not err,'errors':err,'warnings':warn,'output':obj,'overlay':overlay if not err else None})
  print(cond,start+len(batch),'/',len(todo),'seconds',round(time.time()-t,1),'valid',sum(not z[2] for z in results),flush=True)
print('ANNOTATION COMPLETE',flush=True)
