"""Standardized resumable FP32 inference. Heavy GPU jobs run sequentially."""
import sys,time,shutil,argparse,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
import pipeline as base
from pipeline import read,dump,log,VERIFY,REBUILD,REVIEW
from common import verdict
from scoring import extract,correct,strict_answer,recalculate,legacy_answer
RUN=ROOT/'runs/20261012_followup';base.RUN=RUN
class Engine(base.Engine):
    def verify(self,question,response,check=False,truncated=False):
        t=self.torch
        prefix='Question:\n'+question+'\n\nProposed response:\n'
        user=prefix+response+'\n\n'+VERIFY
        prompt=self.chat(user,'You are a careful mathematics verifier.')
        start=prompt.index(prefix)+len(prefix);end=start+len(response)
        # Explicit segment boundary prevents BPE merging response punctuation with separator.
        ids=[];off=[]
        for lo,hi in [(0,end),(end,end+2),(end+2,len(prompt))]:
            enc=self.tok(prompt[lo:hi],add_special_tokens=False,return_offsets_mapping=True)
            ids.extend(enc['input_ids']);off.extend((a+lo,b+lo) for a,b in enc['offset_mapping'])
        assert self.tok.decode(ids)==prompt
        a=max(i for i,(s,e) in enumerate(off) if s<end and e>start)
        separators=[i for i,(s,e) in enumerate(off) if s>=end and e<=end+2 and e>s]
        if not separators: raise RuntimeError('No pure separator token; do not silently shift hidden position')
        b=separators[0];parsed=extract(response,truncated)
        numeric=[i for i,(s,e) in enumerate(off) if parsed['span'] and s<start+parsed['span'][1] and e>start+parsed['span'][0]]
        positions={'numeric':numeric[-1] if numeric else a,'boundary':b,'preverdict':len(ids)-1}
        assert a<b<len(ids)-1
        cache={};handles=[]
        for l in [14]:
            def hook(module,inp,out,l=l):
                h=out[0] if isinstance(out,tuple) else out
                cache[l]=h[0,[positions[k] for k in positions]].detach().float().cpu().numpy()
            handles.append(self.model.model.layers[l-1].register_forward_hook(hook))
        x=t.tensor([ids],device='cuda');t.cuda.synchronize();st=time.perf_counter()
        try:
            with t.inference_mode(): out=self.model(x,attention_mask=t.ones_like(x),use_cache=False)
            hidden={f'{p}_L{l}':cache[l][j].copy() for l in [14] for j,p in enumerate(positions)}
            assert all(np.isfinite(v).all() for v in hidden.values());del out
        finally:
            for h in handles:h.remove()
        t.cuda.synchronize();feature_seconds=time.perf_counter()-st
        checks={}
        if check:
            # Independently forward only each prefix: no future prompt or verdict visible.
            for name,pos in positions.items():
                hs={};handles=[]
                for l in [14]:
                    def h(module,inp,out,l=l):
                        v=out[0] if isinstance(out,tuple) else out;hs[l]=v[0,-1].detach().float().cpu().numpy()
                    handles.append(self.model.model.layers[l-1].register_forward_hook(h))
                try:
                    with t.inference_mode(): result=self.model(x[:,:pos+1],attention_mask=t.ones_like(x[:,:pos+1]),use_cache=False)
                    for l in [14]:
                        rel=float(np.linalg.norm(hs[l]-hidden[f'{name}_L{l}'])/(np.linalg.norm(hidden[f'{name}_L{l}'])+1e-12));checks[f'{name}_L{l}']=rel
                        assert rel<1e-4,(name,l,rel)
                    del result
                finally:
                    for h in handles:h.remove()
        v=self.generate(prompt,self.cfg['max_verdict_tokens'],token_ids=ids);vn,vnll=self.nll(v['prompt_ids'],v['generated_ids'])
        v.update(verdict=verdict(v['decoded_output']),mean_nll=vn,token_nll=vnll,feature_seconds=feature_seconds,positions=positions,position_tokens={p:dict(index=i,token_id=ids[i],decoded=self.tok.decode([ids[i]]),offset=list(off[i])) for p,i in positions.items()},response_char_span=[start,end],causal_prefix_relative_l2=checks)
        scores={}
        for label in ['CORRECT','INCORRECT']:
            label_ids=self.tok.encode(label,add_special_tokens=False)
            _,loss=self.nll(ids,label_ids)
            scores[label]=dict(token_ids=label_ids,tokens=[self.tok.decode([i]) for i in label_ids],log_probability=-float(sum(loss)),token_nll=loss)
        logs=np.array([scores[k]['log_probability'] for k in ['CORRECT','INCORRECT']]);probs=np.exp(logs-np.logaddexp.reduce(logs))
        v.update(candidate_scores=scores,candidate_incorrect_probability=float(probs[1]),numeric_parse=parsed,numeric_available=bool(numeric))
        if not numeric:
            hidden['numeric_L14']=np.zeros_like(hidden['numeric_L14']);v['positions']['numeric']=None;v['position_tokens']['numeric']=None
        return v,hidden

def original_numeric(e,o):
    parsed=extract(o['decoded_output'],o['truncated']);span=parsed['span'];ids=o['generated_ids']
    ends=[len(e.tok.decode(ids[:i+1],skip_special_tokens=True)) for i in range(len(ids))];starts=[0]+ends[:-1]
    ix=[i for i,(a,b) in enumerate(zip(starts,ends)) if span and a<span[1] and b>span[0] and ids[i] not in e.tok.all_special_ids]
    if span:assert ix and e.tok.decode(ids,skip_special_tokens=True)==o['decoded_output']
    return dict(parsed=parsed,token_indices=ix,token_ids=[ids[i] for i in ix],decoded_tokens=[e.tok.decode([ids[i]]) for i in ix],mean_nll=float(np.mean([o['token_nll'][i] for i in ix])) if ix else 0.,max_nll=float(max(o['token_nll'][i] for i in ix)) if ix else 0.,available=bool(ix))

def collect(split,limit=None):
    cfg=read(RUN/'config.json');rows=[r for r in read(RUN/'manifest.json') if r['split']==split]
    if limit:rows=rows[:limit]
    if split!='pilot':assert (RUN/'pilot_audit.json').exists()
    if split=='test':assert (RUN/'selection_frozen.json').exists(),'Freeze all models and settings before generating fresh test'
    e=Engine();t=e.torch
    group_size=8 if split=='pilot' else cfg['batch_size']
    for offset in range(0,len(rows),group_size):
        group=[r for r in rows[offset:offset+group_size] if not (RUN/'records'/f"{r['id']}.json").exists()]
        if not group:continue
        assert shutil.disk_usage(ROOT).free>2**30
        t.cuda.reset_peak_memory_stats();st=time.perf_counter();originals={};pending=[]
        for row in group:
            stage=RUN/'stages'/row['id'];stage.mkdir(parents=True,exist_ok=True);op=stage/'original.json'
            if op.exists():originals[row['id']]=read(op)
            else:pending.append(row)
        if pending:
            results=e.generate_batch([e.chat(cfg['user_instruction']+r['question']) for r in pending],512)
            for r,o in zip(pending,results):
                o['mean_nll'],o['token_nll']=e.nll(o['prompt_ids'],o['generated_ids']);o['numeric']=original_numeric(e,o);o['reused']=False;originals[r['id']]=o;dump(RUN/'stages'/r['id']/'original.json',o)
        parts=[];jobs=[]
        for i,row in enumerate(group):
            rid=row['id'];stage=RUN/'stages'/rid;o=originals[rid];vp=stage/'verification.json';hp=RUN/'hidden'/f'{rid}.npz'
            if vp.exists() and hp.exists():v=read(vp)
            else:
                v,h=e.verify(row['question'],o['decoded_output'],check=split=='pilot' and offset==0 and i<3,truncated=o['truncated']);hp.parent.mkdir(exist_ok=True);np.savez_compressed(hp,**h);dump(vp,v)
            parts.append((row,o,v));
            for a,prompt in [('Rebuild',REBUILD+row['question']),('Review',REVIEW+'Question:\n'+row['question']+'\n\nProposed response:\n'+o['decoded_output'])]:
                if not (stage/f'{a}.json').exists():jobs.append((rid,a,e.chat(prompt)))
        if jobs:
            results=e.generate_batch([j[2] for j in jobs],512)
            for (rid,a,_),out in zip(jobs,results):dump(RUN/'stages'/rid/f'{a}.json',out)
        elapsed=time.perf_counter()-st
        for row,o,v in parts:
            actions={'Keep':dict(answer=o['numeric']['parsed']['answer'],available=True,seconds=0.,tokens=0)}
            ct=time.perf_counter();actions['Recalculate']=recalculate(o['decoded_output'],o['truncated']);actions['Recalculate'].update(seconds=time.perf_counter()-ct,tokens=0)
            legacy={};strict={}
            for a in ['Keep','Rebuild','Review']:
                obj=o if a=='Keep' else read(RUN/'stages'/row['id']/f'{a}.json')
                parsed=extract(obj['decoded_output'],obj['truncated'])
                if a!='Keep':actions[a]=dict(obj,answer=parsed['answer'],available=True,parse=parsed)
                legacy[a]=int(correct(legacy_answer(obj['decoded_output'],obj['truncated'])[0],row['gold']));strict[a]=int(correct(strict_answer(obj['decoded_output']),row['gold']))
            for a,obj in actions.items():obj['success']=int(correct(obj['answer'],row['gold'])) if obj['available'] else None
            r=dict(id=row['id'],split=split,question=row['question'],question_sha256=row['question_sha256'],gold=row['gold'],original=o,verification=v,actions=actions,label_error=1-actions['Keep']['success'],legacy_success=legacy,strict_success=strict,collection_seconds=elapsed/len(parts),peak_gpu_allocated_bytes=t.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=t.cuda.max_memory_reserved())
            dump(RUN/'records'/f"{row['id']}.json",r)
        dump(RUN/'checkpoint.json',dict(split=split,offset=offset,completed=len(list((RUN/'records').glob('*.json')))))
        log(f'{split}: {min(offset+group_size,len(rows))}/{len(rows)}; {elapsed/len(parts):.2f}s/example; peak {t.cuda.max_memory_reserved()/1e9:.2f}GB')
    if split=='pilot' and not limit:
        rr=[read(RUN/'records'/f"{r['id']}.json") for r in rows];checks=[v for r in rr for v in r['verification']['causal_prefix_relative_l2'].values()]
        assert len(checks)==9 and max(checks)<1e-4
        errors=sum(r['label_error'] for r in rr);assert 0<errors<len(rr)
        report=dict(n=len(rr),errors=errors,correct=len(rr)-errors,repairable_errors=sum(r['label_error'] and any(r['actions'][a]['success']==1 for a in ['Recalculate','Rebuild','Review']) for r in rr),seconds=sum(r['collection_seconds'] for r in rr),peak_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in rr),prefix_checks=checks,verdict_counts={k:sum(r['verification']['verdict']==k for r in rr) for k in ['CORRECT','INCORRECT','INVALID']},numeric_available=sum(r['original']['numeric']['available'] for r in rr))
        dump(RUN/'pilot_audit.json',report);log(str(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('split',choices=['pilot','train','validation','test']);p.add_argument('--limit',type=int);a=p.parse_args();collect(a.split,a.limit)
