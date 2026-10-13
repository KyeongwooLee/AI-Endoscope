#!/usr/bin/env python3
import os
os.environ.setdefault('HF_HOME','/tmp/yaicon-ab/huggingface')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse, hashlib, json, time, sys, platform, subprocess
from pathlib import Path
import numpy as np
from common import answer, strict_answer, correct, recalculate, verdict, normalize
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'runs/20261009T0542Z'
RUN=ROOT/'runs/20261012_AB'
SEED=20261012
LAYERS=[7,14,21,28]
VERIFY='Is the proposed response correct? Respond only with CORRECT or INCORRECT.'
REBUILD='Solve this problem afresh. Identify the needed quantities and their relationships, calculate carefully, and give a concise solution. End with Final answer: <number>.\n\nProblem:\n'
REVIEW='Check the proposed solution for reasoning and arithmetic mistakes. Revise only if needed. Give a concise checked solution and end with Final answer: <number>.\n\n'
def dump(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(p)
def read(p): return json.loads(Path(p).read_text())
def log(s): print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),s,flush=True)
def prepare():
    from datasets import load_dataset
    from importlib.metadata import distributions
    if (RUN/'manifest.json').exists(): return
    cfg=read(OLD/'config.json')
    old=read(OLD/'manifest.json'); rows=[]
    limits={'pilot':40,'train':200,'validation':80}
    for split,n in limits.items():
        for r in [r for r in old if r['split']==split][:n]:
            rows.append(dict(r,source_partition='train',reuse_original=str(OLD/'records'/f"{r['id']}.json")))
    ds=load_dataset(cfg['dataset'],'main',split='test',revision=cfg['data_revision'])
    seen={r['question_sha256'] for r in old}
    for i in np.random.default_rng(SEED).permutation(len(ds)):
        r=ds[int(i)]; h=hashlib.sha256(' '.join(r['question'].split()).encode()).hexdigest()
        if h in seen: continue
        seen.add(h);rows.append(dict(id=f'gsm8k_test_{i}',source_index=int(i),question=r['question'],gold_answer=r['answer'],gold=normalize(r['answer'].split('####')[-1]),question_sha256=h,split='test',source_partition='test',reuse_original=None))
        if sum(r['split']=='test' for r in rows)==100: break
    assert len({r['question_sha256'] for r in rows})==len(rows)
    dump(RUN/'manifest.json',rows)
    cfg.update(seed=SEED,source_split='official train for pilot/train/validation; official test for final test',split_counts={s:sum(r['split']==s for r in rows) for s in ['pilot','train','validation','test']},verification_prompt=VERIFY,rebuild_prompt=REBUILD,review_prompt=REVIEW,max_repair_tokens=256,max_verdict_tokens=8,verification_system='You are a careful mathematics verifier.',C_grid=[.01,.1,1.],hidden_candidates=[f'{p}_L{l}' for p in ['answer','boundary','preverdict'] for l in LAYERS],dtype='float32',dtype_reason='GTX1080Ti has slow native FP16; FP32 chosen for stable feasible inference. Reused originals were generated FP16 on V100.',original_reuse='Previously saved natural original responses and generation confidence; immutable source path per record.',scoring='last explicit Final answer/#### number; else last numeric boxed answer; else last number. Exact Decimal equality; dollars/commas normalized; percent stays in question units; no rounding tolerance. Strict-format sensitivity retained.',recalculate='Conservative numeric equality matching own extracted final answer, safe Decimal AST calculator; no reference. Unavailable masked, fixed policy falls back to Keep.',text_features='Train-fit TF-IDF unigram/bigram max_features=1500, min_df=2; question + original response only',hidden_features='Raw post-block output at last response-overlapping token, first pure newline separator token, final verification prompt token; layers 7/14/21/28. Before any verdict token.',selection='A: validation AUROC. B: validation final accuracy then mean per-action log loss. Independent selection before either test evaluation.',bootstrap_iterations=2000,bootstrap_seed=20261013)
    dump(RUN/'config.json',cfg)
    dump(RUN/'packages.json',{d.metadata['Name']:d.version for d in distributions()})
    dump(RUN/'environment.json',dict(hostname=platform.node(),python=sys.version,executable=sys.executable,gpu=subprocess.getoutput('nvidia-smi'),disk=subprocess.getoutput('df -h /home/coder /tmp')))
    log(f'Prepared {len(rows)} records')

class Engine:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        self.torch=torch; self.cfg=read(RUN/'config.json');c=self.cfg
        torch.set_num_threads(4);set_seed(SEED)
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        self.tok=AutoTokenizer.from_pretrained(c['model'],revision=c['tokenizer_revision'])
        self.model=AutoModelForCausalLM.from_pretrained(c['model'],revision=c['model_revision'],dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        assert self.model.config.num_hidden_layers==28
        dump(RUN/'model_config.json',self.model.config.to_dict());(RUN/'chat_template.txt').write_text(self.tok.chat_template)
        dump(RUN/'cuda_check.json',dict(torch=torch.__version__,cuda_available=torch.cuda.is_available(),device=torch.cuda.get_device_name(),tensor_sum=float(torch.ones(8,device='cuda').sum()),frozen=all(not p.requires_grad for p in self.model.parameters())))
    def chat(self,user,system=None):
        return self.tok.apply_chat_template([{'role':'system','content':system or self.cfg['system_prompt']},{'role':'user','content':user}],tokenize=False,add_generation_prompt=True)
    def generate(self,prompt,limit,token_ids=None):
        t=self.torch;ids=token_ids if token_ids is not None else self.tok.encode(prompt,add_special_tokens=False);x=t.tensor([ids],device='cuda');t.cuda.synchronize();start=time.perf_counter()
        with t.inference_mode(): out=self.model.generate(x,attention_mask=t.ones_like(x),do_sample=False,num_beams=1,max_new_tokens=limit,use_cache=True,pad_token_id=self.tok.eos_token_id)
        t.cuda.synchronize();dt=time.perf_counter()-start;gen=out[0,len(ids):].tolist()
        return dict(prompt_ids=ids,generated_ids=gen,decoded_output=self.tok.decode(gen,skip_special_tokens=True),seconds=dt,tokens=len(gen),truncated=len(gen)>=limit and gen[-1] not in self.tok.all_special_ids)
    def generate_batch(self,prompts,limit):
        t=self.torch;self.tok.padding_side='left';self.tok.pad_token=self.tok.eos_token
        enc=self.tok(prompts,add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        t.cuda.synchronize();st=time.perf_counter()
        with t.inference_mode():
            out=self.model.generate(**enc,do_sample=False,num_beams=1,max_new_tokens=limit,use_cache=True,pad_token_id=self.tok.eos_token_id)
        t.cuda.synchronize();seconds=time.perf_counter()-st;results=[]
        for i,prompt in enumerate(prompts):
            gen=out[i,enc['input_ids'].shape[1]:].tolist()
            stop=next((j+1 for j,k in enumerate(gen) if k in self.tok.all_special_ids),len(gen));gen=gen[:stop]
            results.append(dict(prompt_ids=self.tok.encode(prompt,add_special_tokens=False),generated_ids=gen,decoded_output=self.tok.decode(gen,skip_special_tokens=True),seconds=seconds/len(prompts),batch_seconds=seconds,batch_size=len(prompts),cost_kind='amortized_batch_wall_time',tokens=len(gen),truncated=len(gen)>=limit and gen[-1] not in self.tok.all_special_ids))
        return results
    def nll(self,prompt_ids,generated_ids):
        t=self.torch;x=t.tensor([prompt_ids+generated_ids],device='cuda');p=len(prompt_ids)
        with t.inference_mode():
            out=self.model(x,attention_mask=t.ones_like(x),use_cache=False)
            z=out.logits[0,p-1:-1].float(); n=(t.logsumexp(z,-1)-z.gather(1,x[0,p:,None]).squeeze(1)).cpu().numpy()
        good=[i for i,k in enumerate(generated_ids) if k not in self.tok.all_special_ids]
        assert np.isfinite(n).all()
        return float(n[good].mean()),n.tolist()
    def verify(self,question,response,check=False):
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
        b=separators[0];positions={'answer':a,'boundary':b,'preverdict':len(ids)-1}
        assert a<b<len(ids)-1
        cache={};handles=[]
        for l in LAYERS:
            def hook(module,inp,out,l=l):
                h=out[0] if isinstance(out,tuple) else out
                cache[l]=h[0,[positions[k] for k in positions]].detach().float().cpu().numpy()
            handles.append(self.model.model.layers[l-1].register_forward_hook(hook))
        x=t.tensor([ids],device='cuda');t.cuda.synchronize();st=time.perf_counter()
        try:
            with t.inference_mode(): out=self.model(x,attention_mask=t.ones_like(x),use_cache=False)
            hidden={f'{p}_L{l}':cache[l][j].copy() for l in LAYERS for j,p in enumerate(positions)}
            assert all(np.isfinite(v).all() for v in hidden.values());del out
        finally:
            for h in handles:h.remove()
        t.cuda.synchronize();feature_seconds=time.perf_counter()-st
        checks={}
        if check:
            # Independently forward only each prefix: no future prompt or verdict visible.
            for name,pos in positions.items():
                hs={};handles=[]
                for l in LAYERS:
                    def h(module,inp,out,l=l):
                        v=out[0] if isinstance(out,tuple) else out;hs[l]=v[0,-1].detach().float().cpu().numpy()
                    handles.append(self.model.model.layers[l-1].register_forward_hook(h))
                try:
                    with t.inference_mode(): result=self.model(x[:,:pos+1],attention_mask=t.ones_like(x[:,:pos+1]),use_cache=False)
                    for l in LAYERS:
                        rel=float(np.linalg.norm(hs[l]-hidden[f'{name}_L{l}'])/(np.linalg.norm(hidden[f'{name}_L{l}'])+1e-12));checks[f'{name}_L{l}']=rel
                        assert rel<1e-4,(name,l,rel)
                    del result
                finally:
                    for h in handles:h.remove()
        v=self.generate(prompt,self.cfg['max_verdict_tokens'],token_ids=ids);vn,vnll=self.nll(v['prompt_ids'],v['generated_ids'])
        v.update(verdict=verdict(v['decoded_output']),mean_nll=vn,token_nll=vnll,feature_seconds=feature_seconds,positions=positions,position_tokens={p:dict(index=i,token_id=ids[i],decoded=self.tok.decode([ids[i]]),offset=list(off[i])) for p,i in positions.items()},response_char_span=[start,end],causal_prefix_relative_l2=checks)
        return v,hidden

def collect(split,limit=None):
    cfg=read(RUN/'config.json');rows=[r for r in read(RUN/'manifest.json') if r['split']==split]
    if limit: rows=rows[:limit]
    if split!='pilot' and not (RUN/'pilot_audit.json').exists(): raise RuntimeError('pilot audit required')
    e=Engine();t=e.torch;completed=0
    for offset in range(0,len(rows),cfg['collection_group_size']):
        group=rows[offset:offset+cfg['collection_group_size']]; pending=[];t.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for i,row in enumerate(group,offset):
            p=RUN/'records'/f"{row['id']}.json"
            if p.exists(): completed+=1;continue
            if __import__('shutil').disk_usage(ROOT).free<1073741824: raise RuntimeError('storage reserve')
            rid=row['id'];stage=RUN/'stages'/rid;stage.mkdir(parents=True,exist_ok=True);op=stage/'original.json'
            if op.exists(): orig=read(op)
            elif row['reuse_original']:
                r=read(row['reuse_original']);orig={k:r[k] for k in ['prompt_ids','generated_ids','decoded_output']};orig.update(mean_nll=r['features']['response_mean_nll'],token_nll=r['token_nll'],seconds=0.,tokens=r['features']['response_tokens'],truncated=r['truncation'],reused=True);dump(op,orig)
            else:
                orig=e.generate(e.chat(cfg['user_instruction']+row['question']),cfg['max_new_tokens']);orig['mean_nll'],orig['token_nll']=e.nll(orig['prompt_ids'],orig['generated_ids']);orig['reused']=False;dump(op,orig)
            vp=stage/'verification.json';hp=RUN/'hidden'/f'{rid}.npz'
            if vp.exists() and hp.exists(): v=read(vp)
            else:
                v,hidden=e.verify(row['question'],orig['decoded_output'],check=split=='pilot' and i<3);hp.parent.mkdir(exist_ok=True);np.savez_compressed(hp,**hidden);dump(vp,v)
            pred,parse=answer(orig['decoded_output'],orig['truncated']);actions={'Keep':dict(answer=pred,available=True,seconds=0.,tokens=0)}
            ct=time.perf_counter();actions['Recalculate']=recalculate(orig['decoded_output'],orig['truncated']);actions['Recalculate'].update(seconds=time.perf_counter()-ct,tokens=0)
            pending.append(dict(row=row,stage=stage,p=p,orig=orig,v=v,pred=pred,parse=parse,actions=actions))
        jobs=[]
        for item in pending:
            row=item['row'];orig=item['orig']
            for action,prompt in [('Rebuild',REBUILD+row['question']),('Review',REVIEW+'Question:\n'+row['question']+'\n\nProposed response:\n'+orig['decoded_output'])]:
                ap=item['stage']/f'{action}.json'
                if ap.exists(): item['actions'][action]=read(ap)
                else: jobs.append((item,action,ap,e.chat(prompt)))
        if jobs:
            results=e.generate_batch([job[3] for job in jobs],cfg['max_repair_tokens'])
            for (item,action,ap,_),result in zip(jobs,results): dump(ap,result);item['actions'][action]=result
        elapsed=time.perf_counter()-start
        for item in pending:
            row=item['row'];orig=item['orig'];actions=item['actions'];v=item['v']
            for action in ['Rebuild','Review']:
                result=actions[action];result.update(answer=answer(result['decoded_output'],result['truncated'])[0],available=True)
            for action,obj in actions.items(): obj['success']=int(correct(obj['answer'],row['gold'])) if obj['available'] else None
            error=1-actions['Keep']['success'];assert actions['Keep']['success']==1-error
            record=dict(id=row['id'],split=split,question=row['question'],question_sha256=row['question_sha256'],original=orig,verification=v,actions=actions,label_error=error,gold=row['gold'],parse_status=item['parse'],strict_prediction=strict_answer(orig['decoded_output']),strict_error=int(not correct(strict_answer(orig['decoded_output']),row['gold'])),collection_seconds=elapsed/len(pending),collection_cost_kind='amortized_group_wall_time',peak_gpu_allocated_bytes=t.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=t.cuda.max_memory_reserved())
            dump(item['p'],record);completed+=1;dump(RUN/'checkpoint.json',dict(split=split,last_id=row['id'],completed_records=len(list((RUN/'records').glob('*.json')))))
            log(f'{split} {completed}/{len(rows)} {row["id"]} error={error} actions={[a["success"] for a in actions.values()]} amortized_seconds={record["collection_seconds"]:.1f}')

def audit():
    rows=[read(p) for p in sorted((RUN/'records').glob('*.json')) if read(p)['split']=='pilot']
    assert len(rows)==40,len(rows)
    errors=sum(r['label_error'] for r in rows);assert 0<errors<len(rows)
    outcomes=np.array([[a['success'] if a['success'] is not None else r['actions']['Keep']['success'] for a in r['actions'].values()] for r in rows])
    checks=[v for r in rows for v in r['verification']['causal_prefix_relative_l2'].values()];assert len(checks)==36 and max(checks)<1e-4
    report=dict(n=len(rows),errors=errors,correct=len(rows)-errors,verdict_counts={k:sum(r['verification']['verdict']==k for r in rows) for k in ['CORRECT','INCORRECT','INVALID']},recalculate_available=sum(r['actions']['Recalculate']['available'] for r in rows),fixed_policy_successes_with_keep_fallback=outcomes.sum(0).tolist(),oracle_successes=int(outcomes.max(1).sum()),different_actions_help=int(((outcomes[:,1:]>outcomes[:,:1]).any(1)).sum()),collection_seconds=sum(r['collection_seconds'] for r in rows),peak_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in rows),peak_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in rows),causal_checks=checks,passed=True)
    dump(RUN/'pilot_audit.json',report)
    examples=[]
    for r in rows:
        if len(examples)<6 and (r['label_error'] or len(examples)<2): examples.append(r)
    dump(RUN/'pilot_readable_examples.json',examples);log(json.dumps(report))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','collect','audit']);p.add_argument('--split',default='pilot',choices=['pilot','train','validation','test']);p.add_argument('--limit',type=int);a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='audit': audit()
    else: collect(a.split,a.limit)
