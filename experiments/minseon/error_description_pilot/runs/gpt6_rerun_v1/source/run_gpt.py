from pathlib import Path
import json,os,sys,subprocess,time,hashlib,concurrent.futures,threading,argparse
out=Path(__file__).resolve().parents[1];base=out.parents[1];sys.path.insert(0,str(base));from edp.annotation import validate_output
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
cfg=json.load(open(out/'config.json'));lock=threading.Lock()
cmd=['codex','exec','--ignore-user-config','--ephemeral','--skip-git-repo-check','-C','/tmp/errorprobe-gpt-readonly','-s','read-only','-m','gpt-6-astra','-c','model_reasoning_effort="medium"','-c','web_search="disabled"','--json']
for f in ['apps','plugins','hooks','shell_tool','unified_exec','multi_agent','browser_use','computer_use','image_generation','view_image']:cmd+=['--disable',f]
for f in ['omit_skill_descriptions','skip_host_skill_discovery']:cmd+=['--enable',f]
def append(p,r):
 with lock:
  with p.open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
def unexpected(e):
 item=e.get('item',{})
 if item.get('type') in ['agent_message','reasoning']:return False
 if item.get('type')=='error' and item.get('message','').startswith('Under-development features enabled:'):return False
 return True

def call(job_id,prompt,schema,attempt):
 folder=out/'calls'/f'{job_id}-a{attempt}';folder.mkdir(exist_ok=True);(folder/'prompt.txt').write_text(prompt);target=folder/'response.json'
 if target.exists() and (folder/'metadata.json').exists():
  meta=json.loads((folder/'metadata.json').read_text())
  if meta['returncode']==0 and meta['prompt_sha256']==hashlib.sha256(prompt.encode()).hexdigest() and not any(unexpected(e) for e in meta['non_message_items']):
   raw=target.read_text();return json.loads(raw),meta,raw
 command=cmd+['--output-schema',str(out/f'schemas/{schema}.json'),'-o',str(target),'-'];start=time.time()
 with (folder/'events.jsonl').open('w') as stdout,(folder/'stderr.txt').open('w') as stderr:
  p=subprocess.run(command,input=prompt,text=True,stdout=stdout,stderr=stderr,timeout=300)
 events=jl(folder/'events.jsonl');usage={};toolitems=[]
 for e in events:
  if e.get('type')=='turn.completed':usage=e.get('usage',{})
  if e.get('type') in ['item.started','item.completed'] and unexpected(e):toolitems.append(e)
 meta={'requested_model':'gpt-6-astra','reasoning_effort':'medium','fresh_session':True,'route':'Codex CLI ChatGPT login','returncode':p.returncode,'seconds':time.time()-start,'usage':usage,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'schema_sha256':hashlib.sha256((out/f'schemas/{schema}.json').read_bytes()).hexdigest(),'non_message_items':toolitems,'command':command};(folder/'metadata.json').write_text(json.dumps(meta,indent=2))
 if p.returncode or not target.exists():raise RuntimeError(f'Codex execution failed for {job_id}, inspect {folder}; returncode={p.returncode}')
 if toolitems:raise RuntimeError(f'Unexpected tool activity for {job_id}; do not accept experimental output')
 raw=target.read_text();return json.loads(raw),meta,raw

def annotate(p):
 prompt='This is a standalone mathematical annotation request. Use only the supplied texts. Do not call tools, read files, run commands, search, or delegate. Complete the requested JSON directly.\n\n'+p['rendered_prompt'];history=[]
 for attempt in [1,2]:
  obj,meta,raw=call(p['job_id'],prompt,p['condition'],attempt);rawerr,rawwarn,_=validate_output(p['condition'],obj,p,cfg);fixed=json.loads(json.dumps(obj));corrected=False
  if p['condition']=='G' and isinstance(fixed.get('evidence_quote'),str) and fixed['evidence_quote'] and p['response'].count(fixed['evidence_quote'])==1:
   q=fixed['evidence_quote'];start=p['response'].index(q);corrected=(fixed.get('evidence_start'),fixed.get('evidence_end'),fixed.get('evidence_occurrence'))!=(start,start+len(q),1);fixed.update(evidence_start=start,evidence_end=start+len(q),evidence_occurrence=1)
  errors,warnings,overlay=validate_output(p['condition'],fixed,p,cfg);history.append({'attempt':attempt,'raw_output':obj,'raw_errors':rawerr,'errors_after_offset_resolution':errors,'offset_computed':corrected,'call_dir':f'calls/{p["job_id"]}-a{attempt}','usage':meta['usage']})
  if not errors:break
  prompt='This is a standalone mathematical annotation request. Use no tools or external resources.\n\n'+p['rendered_prompt']+'\n\nA prior attempt failed validation: '+json.dumps(errors)+'. Return complete valid JSON. Copy source_id and response_sha256 exactly. Keep the error description between25 and60 English words. Use an exact quote; character offsets may be null and will be calculated by code.'
 r={'job_id':p['job_id'],'source_id':p['source_id'],'dataset':p['dataset'],'condition':p['condition'],'response_sha256':p['response_sha256'],'valid':not errors,'errors':errors,'warnings':warnings,'output':fixed,'overlay':overlay if not errors else None,'attempts':history,'model':'gpt-6-astra','reasoning_effort':'medium','fresh_context_per_case_condition':True};return r

def judge(p):
 obj,meta,raw=call(p['job_id'],p['prompt'],'judge',1);assert obj.get('judgment') in ['same','different','uncertain'];return {k:v for k,v in p.items() if k!='prompt'}|{'output':obj,'model':'gpt-6-astra','reasoning_effort':'medium','usage':meta['usage'],'fresh_context':True,'human_ground_truth':False}

def judge_batch(p):
 obj,meta,raw=call(p['job_id'],p['prompt'],'judge_batch',1)
 result=obj['results'];expected={r['blind_id'] for r in p['pairs']}
 assert len(result)==len(expected) and {r['pair_id'] for r in result}==expected
 assert all(r['judgment'] in ['same','different','uncertain'] for r in result)
 return {k:v for k,v in p.items() if k!='prompt'}|{'output':obj,'model':'gpt-6-astra','reasoning_effort':'medium','usage':meta['usage'],'fresh_context_per_batch':True,'human_ground_truth':False}

def screen(p):
 obj,meta,raw=call(p['job_id'],p['prompt'],'screen',1)
 assert obj['status'] in ['math_error','no_math_error','answer_conflict','ambiguous']
 return {k:v for k,v in p.items() if k!='prompt'}|{'output':obj,'model':'gpt-6-astra','reasoning_effort':'medium','usage':meta['usage'],'fresh_context':True,'reference_provided':False,'human_ground_truth':False}

def run(phase):
 if phase=='annotation':jobs=jl(out/'annotation_packets.jsonl');dest=out/'annotation/all.jsonl';fn=annotate
 elif phase=='controls':jobs=jl(out/'control_jobs.jsonl');dest=out/'control_outputs.jsonl';fn=judge
 elif phase=='pairs':jobs=jl(out/'judge_jobs.jsonl');dest=out/'pair_judgments.jsonl';fn=judge
 elif phase=='screen':jobs=jl(out/'screen_jobs.jsonl');dest=out/'screen_outputs.jsonl';fn=screen
 elif phase=='real_audit':jobs=jl(out/'real_audit_jobs.jsonl');dest=out/'real_audit_outputs.jsonl';fn=judge_batch
 elif phase in ['controls_batch','pairs_batch']:
  mode=phase.removesuffix('_batch');jobs=jl(out/f'{mode}_batch_jobs.jsonl');dest=out/f'{mode}_batch_outputs.jsonl';fn=judge_batch
 else:raise ValueError(phase)
 done={r['job_id'] for r in jl(dest)} if dest.exists() else set();todo=[r for r in jobs if r['job_id'] not in done];todo.sort(key=lambda p:(p.get('dataset')!='pilot130',len(p.get('rendered_prompt',p.get('prompt','')))))
 completed=len(done);start=time.time();fatal=None
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  it=iter(todo);pending={}
  for _ in range(4):
   p=next(it,None)
   if p:pending[ex.submit(fn,p)]=p
  while pending:
   ready,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
   for future in ready:
    p=pending.pop(future)
    try:r=future.result()
    except Exception as e:
     append(out/'execution_failures.jsonl',{'phase':phase,'job_id':p['job_id'],'error':str(e)});fatal=e;print('EXECUTION FAILURE',p['job_id'],str(e),flush=True);continue
    append(dest,r);completed+=1
    if completed%10==0 or completed==len(jobs):print(phase,completed,'/',len(jobs),'seconds',round(time.time()-start,1),flush=True)
    if fatal is None:
     nxt=next(it,None)
     if nxt:pending[ex.submit(fn,nxt)]=nxt
 if fatal:raise fatal
 assert completed==len(jobs)
 print(phase,'COMPLETE',flush=True)
if __name__=='__main__':
 phase=sys.argv[1]
 if not (out/'runtime.json').exists():(out/'runtime.json').write_text(json.dumps({'cli_version':subprocess.check_output(['codex','--version'],text=True).strip(),'command_template':cmd,'snapshot_version':'not exposed'},indent=2))
 run(phase)
 if phase=='pairs_batch':
  run('real_audit')
  run('screen')
