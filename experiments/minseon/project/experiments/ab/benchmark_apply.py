#!/usr/bin/env python3
"""Batch-one deployment cost on first six excluded pilot questions, no accuracy tuning."""
import time,joblib,numpy as np
from threadpoolctl import threadpool_limits
from pipeline import Engine,RUN,read,dump,REBUILD,REVIEW
from evaluate import Features,Constant,inputs,availability,probabilities,choose,ACTIONS,load
from common import verdict,recalculate

def main():
    obj=joblib.load(RUN/'trained.joblib');f=obj['features'];rows=load('pilot')[:6];e=Engine();t=e.torch;records=[]
    for row in rows:
        # No reference field is passed into verification, prediction, or action prompts.
        ids=row['verification']['prompt_ids'];st=time.perf_counter();v=e.generate('',8,token_ids=ids);nll,_=e.nll(ids,v['generated_ids']);t.cuda.synchronize();verify_seconds=time.perf_counter()-st
        public=dict(id=row['id'],question=row['question'],original=row['original'],verification=dict(verdict=verdict(v['decoded_output']),mean_nll=nll));x=inputs([public]);mask=availability(x)
        for name in ['external','internal']:
            cfg=obj['selection'][name];hidden_seconds=0.;vec=None
            if cfg['key']:
                position,layer=cfg['key'].split('_L');layer=int(layer);pos=row['verification']['positions'][position];cache={}
                def hook(module,inp,out):
                    h=out[0] if isinstance(out,tuple) else out;cache['h']=h[0,pos].detach().float().cpu().numpy()
                handle=e.model.model.layers[layer-1].register_forward_hook(hook)
                tensor=t.tensor([ids],device='cuda');t.cuda.synchronize();st=time.perf_counter()
                try:
                    with t.inference_mode():out=e.model(tensor,attention_mask=t.ones_like(tensor),use_cache=False)
                    vec=cache['h'];del out
                finally:handle.remove()
                t.cuda.synchronize();hidden_seconds=time.perf_counter()-st
            # Matrix construction uses freshly measured verification and hidden values.
            old_hidden=f.hidden
            if vec is not None:f.hidden=lambda xx,key:np.array([vec])
            st=time.perf_counter()
            try:matrix=f.transform(x,cfg['kind'],cfg['key']);p=probabilities(obj['models'][name],matrix);chosen=int(choose(p,mask)[0])
            finally:f.hidden=old_hidden
            cpu_seconds=time.perf_counter()-st;action=ACTIONS[chosen];st=time.perf_counter();tokens=0
            if action=='Recalculate': recalculate(row['original']['decoded_output'],row['original']['truncated'])
            elif action in ['Rebuild','Review']:
                prompt=REBUILD+row['question'] if action=='Rebuild' else REVIEW+'Question:\n'+row['question']+'\n\nProposed response:\n'+row['original']['decoded_output']
                out=e.generate(e.chat(prompt),512);tokens=out['tokens']
            t.cuda.synchronize();repair_seconds=time.perf_counter()-st
            records.append(dict(id=row['id'],selector=name,action=action,verification_seconds=verify_seconds,extra_hidden_seconds=hidden_seconds,classifier_seconds=cpu_seconds,selected_repair_seconds=repair_seconds,selected_repair_tokens=tokens,total_incremental_seconds=verify_seconds+hidden_seconds+cpu_seconds+repair_seconds))
    summary={}
    for name in ['external','internal']:
        rr=[r for r in records if r['selector']==name]
        summary[name]={key:float(np.mean([r[key] for r in rr])) for key in ['verification_seconds','extra_hidden_seconds','classifier_seconds','selected_repair_seconds','total_incremental_seconds']}
    dump(RUN/'deployment_benchmark.json',dict(n_pilot_questions=6,batch_size=1,model_loading_excluded=True,original_generation_and_confidence_already_available=True,accuracy_not_evaluated=True,summary=summary,records=records))
    print(summary)
if __name__=='__main__':
    with threadpool_limits(limits=4):main()
