from pathlib import Path
import subprocess,sys,time,json
root=Path(__file__).resolve().parents[1];src=root/'source'
def step(script,*args):
 log=root/(script.removesuffix('.py')+('_'+'_'.join(args) if args else '')+'.log')
 print('START',script,*args,flush=True)
 with log.open('w') as f:subprocess.run([sys.executable,'-u',str(src/script),*args],stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,*args,flush=True)
def wait_for(n,dataset=None):
 last=None;lastchange=time.time()
 while True:
  p=root/'annotation/all.jsonl';rows=[]
  if p.exists():
   text=p.read_text();ls=text.splitlines()
   if text and not text.endswith('\n'):ls=ls[:-1]
   rows=[json.loads(l) for l in ls]
  count=sum(dataset is None or r['dataset']==dataset for r in rows)
  if count>=n:return
  if count!=last:last=count;lastchange=time.time()
  if time.time()-lastchange>1800:raise RuntimeError('No annotation progress for30 minutes; inspect annotation logs before resuming')
  time.sleep(5)
wait_for(260,'pilot130');step('analyze.py','pilot130')
wait_for(1260);step('run_gpt.py','controls_batch');step('control_gate.py');step('analyze.py','expansion500');step('build_judge_batches.py','pairs')
if json.load(open(root/'control_results.json'))['passed']:step('run_gpt.py','pairs_batch')
step('evaluate.py');step('report.py');step('validate_and_package.py');print('PIPELINE COMPLETE',flush=True)
