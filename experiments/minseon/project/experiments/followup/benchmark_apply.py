"""Excluded-pilot single-request timing; no accuracy-based selection."""
import time,tempfile,sys
from pathlib import Path
import joblib,numpy as np
from collect import Engine,RUN,read,dump,REBUILD,REVIEW
from learn import Features,Constant,inputs,availability,predict_policies
from evaluate import probabilities,ACTIONS
from common import recalculate

def main():
 obj=joblib.load(RUN/'models/n1000_seed101.joblib');f=obj['features'];e=Engine();t=e.torch
 rows=[read(RUN/'records'/f"{r['id']}.json") for r in read(RUN/'manifest.json') if r['split']=='pilot'][:6];timings=[]
 with tempfile.TemporaryDirectory(prefix='followup_apply_') as temp:
  f.hidden_dir=Path(temp)
  for r in rows:
   o=r['original'];ids=r['verification']['prompt_ids'];st=time.perf_counter();e.nll(o['prompt_ids'],o['generated_ids']);original_nll_seconds=time.perf_counter()-st
   st=time.perf_counter();v=e.generate(e.tok.decode(ids),8,token_ids=ids);e.nll(ids,v['generated_ids'])
   for label in ['CORRECT','INCORRECT']:e.nll(ids,e.tok.encode(label,add_special_tokens=False))
   verifier_seconds=time.perf_counter()-st
   assert v['decoded_output']==r['verification']['decoded_output']
   positions=r['verification']['positions'];capture={};names=[p for p,i in positions.items() if i is not None]
   def hook(module,inp,out):
    h=out[0] if isinstance(out,tuple) else out
    for p in names:capture[p+'_L14']=h[0,positions[p]].detach().float().cpu().numpy().copy()
   handle=e.model.model.layers[13].register_forward_hook(hook);xx=t.tensor([ids],device='cuda');t.cuda.synchronize();st=time.perf_counter()
   try:
    with t.inference_mode():forward=e.model(xx,attention_mask=t.ones_like(xx),use_cache=False)
    del forward
   finally:handle.remove()
   t.cuda.synchronize();hidden_seconds=time.perf_counter()-st
   if 'numeric_L14' not in capture:capture['numeric_L14']=np.zeros(1536,dtype=np.float32)
   np.savez_compressed(Path(temp)/f"{r['id']}.npz",**capture)
   x=inputs([r]);mask=availability(x)
   for variant,(aa,bb) in obj['chosen'].items():
    ca=obj['selection'][aa];cb=obj['selection'][bb];st=time.perf_counter();error=obj['models'][aa].predict_proba(f.transform(x,ca['key'],ca['rep']))[:,1];cpu_a=time.perf_counter()-st
    st=time.perf_counter();success=probabilities(obj['models'][bb],f.transform(x,cb['key'],cb['rep']));cpu_b=time.perf_counter()-st
    choices=predict_policies(error,success,mask,obj['policies'][variant])
    for policy,ch in choices.items():
     action=ACTIONS[ch[0]];st=time.perf_counter();output=None
     if action=='Recalculate':output=recalculate(o['decoded_output'],o['truncated'])
     elif action in ('Rebuild','Review'):
      prompt=REBUILD+r['question'] if action=='Rebuild' else REVIEW+'Question:\n'+r['question']+'\n\nProposed response:\n'+o['decoded_output'];output=e.generate(e.chat(prompt),512)
     repair_seconds=time.perf_counter()-st if action!='Keep' else 0.;cpu=cpu_a if policy=='trigger' else cpu_b
     timings.append(dict(id=r['id'],variant=variant,policy=policy,action=action,original_confidence_seconds=original_nll_seconds,verifier_seconds=verifier_seconds,extra_hidden_seconds=hidden_seconds if variant=='internal' else 0.,cpu_seconds=cpu,repair_seconds=repair_seconds,total_excluding_original_confidence=verifier_seconds+(hidden_seconds if variant=='internal' else 0.)+cpu+repair_seconds,repair_output=output))
   print('Benchmarked',r['id'],flush=True)
 summary={}
 for variant in ['external','internal']:
  for policy in ['selector','conservative','trigger']:
   rows=[r for r in timings if r['variant']==variant and r['policy']==policy];summary[variant+'_'+policy]={k:float(np.mean([r[k] for r in rows])) for k in ['original_confidence_seconds','verifier_seconds','extra_hidden_seconds','cpu_seconds','repair_seconds','total_excluding_original_confidence']}
 dump(RUN/'deployment_benchmark.json',dict(scope='Six excluded pilot examples, batch1, loaded model, existing original response. Original-confidence recomputation measured separately. No accuracy analysis.',rows=timings,mean_seconds=summary))
if __name__=='__main__':main()
