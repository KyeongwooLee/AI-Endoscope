"""Collect 4,000 additional parsed, nontruncated, distinct Qwen errors.

Only the frozen question prompt is supplied to the generator. References are used
for strict grading and are never fed to the generator or feature extractor.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import zipfile

import numpy as np
import torch
from transformers import StoppingCriteria,StoppingCriteriaList

from protocol import sha,write_json,commit,read_records

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'experiments/sunghun'
sys.path.insert(0,str(SOURCE/'gsm8k-logit-study'))
sys.path.insert(0,str(SOURCE/'smollm2-detailed-v1'))
from gsm8k_lens.engine import load_model,make_generation_config
from gsm8k_lens.answers import parse_answer,label_answer
from extract import extract

def text_hash(text): return hashlib.sha256(text.encode()).hexdigest()

class FinalLines(StoppingCriteria):
    def __init__(self,tokenizer,prompt_length,batch_size):
        self.tokenizer=tokenizer
        self.prompt_length=prompt_length
        self.lengths=[None]*batch_size
    def __call__(self,input_ids,scores=None,**kwargs):
        last=input_ids[:,-1].tolist()
        pieces=self.tokenizer.batch_decode([[token] for token in last],skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
        stop=[]
        for i,piece in enumerate(pieces):
            if self.lengths[i] is None and piece.endswith(('\n','\r')):
                text=self.tokenizer.decode(input_ids[i,self.prompt_length:].tolist(),skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
                if parse_answer(text)['status']=='ok':
                    self.lengths[i]=input_ids.shape[1]-self.prompt_length
            stop.append(self.lengths[i] is not None)
        return torch.tensor(stop,device=input_ids.device,dtype=torch.bool)

def make_schedule(questions,cfg):
    pools={domain:[r for r in questions if r['dataset']==domain] for domain in ['gsm8k','svamp']}
    schedule=[]
    cursor=Counter()
    decks={}
    while len(schedule)<cfg['maximum_generated_responses']:
        for domain in ['gsm8k']*4+['svamp']:
            cycle,position=divmod(cursor[domain],len(pools[domain]))
            if (domain,cycle) not in decks:
                deck=list(pools[domain])
                random.Random(f"{cfg['seed']}:{domain}:{cycle}").shuffle(deck)
                decks[(domain,cycle)]=deck
            original=decks[(domain,cycle)][position]
            row=dict(original,id=f"{original['question_id']}__extra20261029__a{cycle:03d}",attempt=cycle)
            schedule.append(row)
            cursor[domain]+=1
            if len(schedule)>=cfg['maximum_generated_responses']: break
    assert len({r['id'] for r in schedule})==len(schedule)
    return schedule

@torch.inference_mode()
def generate_batch(model,tokenizer,rows,cfg):
    prompts=[]
    for row in rows:
        messages=[{'role':'system','content':cfg['prompt']['system']},
                  {'role':'user','content':cfg['prompt']['user_template'].replace('{question}',row['question'])}]
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        assert len(ids)<=cfg['generation']['max_prompt_tokens']
        prompts.append(ids)
    gcfg=make_generation_config(model,tokenizer,cfg)
    gcfg.do_sample=True
    gcfg.temperature=cfg['generation']['temperature']
    gcfg.top_p=cfg['generation']['top_p']
    gcfg.top_k=0
    seed=int(text_hash(str(cfg['seed'])+':'+','.join(row['id'] for row in rows))[:8],16)%(2**31)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    width=max(map(len,prompts))
    ids=torch.full((len(rows),width),gcfg.pad_token_id,device=model.device,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,prompt in enumerate(prompts):
        ids[i,-len(prompt):]=torch.tensor(prompt,device=model.device)
        mask[i,-len(prompt):]=1
    stopping=FinalLines(tokenizer,width,len(rows))
    start=time.monotonic()
    output=model.generate(ids,attention_mask=mask,generation_config=gcfg,
        stopping_criteria=StoppingCriteriaList([stopping]),return_dict_in_generate=True)
    sequences=output.sequences[:,width:].cpu().tolist()
    del output
    eos=gcfg.eos_token_id if isinstance(gcfg.eos_token_id,list) else [gcfg.eos_token_id]
    results=[]
    for i,sequence in enumerate(sequences):
        eos_index=next((j for j,token in enumerate(sequence) if token in eos),None)
        final_length=stopping.lengths[i]
        if final_length is not None and (eos_index is None or final_length<=eos_index+1):
            sequence=sequence[:final_length]
            reason='final_answer_line'
        elif eos_index is not None:
            sequence=sequence[:eos_index+1]
            reason='eos'
        else:
            reason='max_new_tokens'
        text=tokenizer.decode(sequence,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        results.append({'prompt_token_ids':prompts[i],'generated_token_ids':sequence,'text':text,
            'prompt_tokens':len(prompts[i]),'generated_tokens':len(sequence),'truncated':reason=='max_new_tokens',
            'stop_reason':reason,'batch_seed':seed,'batch_ids':[r['id'] for r in rows],
            'effective_generation_config':gcfg.to_dict(),'batch_seconds':time.monotonic()-start})
    return results

def main(smoke=False):
    root=ROOT
    lock=(root/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((root/'input/config.json').read_text())
    questions=json.loads((root/'input/questions.json').read_text())
    seen=set(json.loads((root/'input/existing_pairs.json').read_text()))
    old_records=list(read_records(root))
    accepted=[r for r in old_records if r['selected_new_error']]
    for r in old_records: seen.add(r['question_group']+':'+r['response_sha256'])
    schedule=make_schedule(questions,cfg)
    cursor=len(old_records)
    assert [r['id'] for r in old_records]==[r['id'] for r in schedule[:cursor]]
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    state={'phase':'loading_model','pid':os.getpid(),'target':cfg['target_new_eligible_errors'],
           'generated':cursor,'selected_errors':len(accepted),'updated_unix':time.time()}
    write_json(root/'progress.json',state)
    local_cfg=json.loads(json.dumps(cfg))
    local_cfg['model']['id']=str(root/'model')
    model,tokenizer=load_model(local_cfg,cfg['model']['revision'])
    model.requires_grad_(False)
    write_json(root/'environment.json',{'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
        'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'model_id':cfg['model']['id'],
        'revision':cfg['model']['revision'],'dtype':str(next(model.parameters()).dtype),
        'chat_template_sha256':text_hash(tokenizer.chat_template),'parameters':sum(p.numel() for p in model.parameters()),
        'tf32':False,'batch_size':cfg['generation']['batch_size']})
    batch_size=cfg['generation']['batch_size']
    feature_receipts=sorted((root/'outbox').glob('features_*.json'))
    featured={identifier for p in feature_receipts for identifier in json.loads(p.read_text())['ids']}
    def extract_selected(records,batch_index):
        pending=[r for r in records if r['selected_new_error'] and r['id'] not in featured]
        if not pending: return
        members={}
        for r in pending:
            arrays,qa,names=extract(model,tokenizer,r['generation'],cfg,quality=len(featured)<4)
            arrays.update(posthoc_question=arrays['hidden'][0,-1],posthoc_end=arrays['pooled_hidden'][-1,0],
                          posthoc_mean=arrays['pooled_hidden'][-1,1],posthoc_number=arrays['pooled_hidden'][-1,2])
            r['feature_quality']=qa
            r['confidence_names']=names
            buf=io.BytesIO()
            np.savez_compressed(buf,**arrays)
            members['features/'+r['id']+'.npz']=buf.getvalue()
            featured.add(r['id'])
        members['records.jsonl']=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in pending).encode()
        commit(root,f'features_{batch_index:06d}',members,{'kind':'features','count':len(pending),'ids':[r['id'] for r in pending]})
    # Resume feature work from already durable raw generations before creating new samples.
    if old_records:
        for batch_index in sorted({r['batch_index'] for r in old_records}):
            extract_selected([r for r in old_records if r['batch_index']==batch_index],batch_index)
    started=time.monotonic()
    initial_cursor=cursor
    batches_this_session=0
    totals=Counter(r['status'] for r in old_records)
    while len(accepted)<cfg['target_new_eligible_errors'] and cursor<len(schedule):
        rows=schedule[cursor:cursor+batch_size]
        batch_index=cursor//batch_size
        generated=generate_batch(model,tokenizer,rows,cfg)
        records=[]
        for row,generation in zip(rows,generated):
            label=label_answer(generation['text'],row['reference'],generation['truncated'])
            pair=row['question_group']+':'+text_hash(generation['text'])
            if pair in seen: status='duplicate'
            elif not label['analysis_eligible']: status='format_or_truncation'
            elif label['answer_correct']: status='correct'
            else: status='eligible_error'
            selected=status=='eligible_error' and len(accepted)<cfg['target_new_eligible_errors']
            record=dict(row,generation=generation,label=label,response_sha256=text_hash(generation['text']),
                selected_new_error=selected,status=status,batch_index=batch_index,
                math_error_type=None,human_review_status='not_reviewed',official_gsm_test=False)
            records.append(record)
            seen.add(pair)
            totals[status]+=1
            if selected: accepted.append(record)
        payload=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records).encode()
        commit(root,f'generations_{batch_index:06d}',{'records.jsonl':payload},
               {'kind':'generations','count':len(records),'ids':[r['id'] for r in records],
                'selected_errors':sum(r['selected_new_error'] for r in records)})
        cursor+=len(rows)
        elapsed=time.monotonic()-started
        state.update(phase='extracting_features',generated=cursor,selected_errors=len(accepted),
            feature_errors=len(featured),counts=dict(totals),elapsed_session_seconds=elapsed,
            examples_per_second=(cursor-initial_cursor)/max(elapsed,.001),updated_unix=time.time())
        write_json(root/'progress.json',state)
        print('GENERATED',json.dumps(state),flush=True)
        extract_selected(records,batch_index)
        state.update(phase='running',feature_errors=len(featured),updated_unix=time.time())
        write_json(root/'progress.json',state)
        batches_this_session+=1
        if smoke and batches_this_session>=2:
            state.update(phase='smoke_complete',updated_unix=time.time())
            write_json(root/'progress.json',state)
            print('SMOKE_COMPLETE',json.dumps(state),flush=True)
            return
    if len(accepted)!=cfg['target_new_eligible_errors']:
        raise RuntimeError('Sampling budget exhausted before collecting 4000 eligible errors')
    assert len(featured)==len(accepted)==4000
    write_json(root/'selected_ids.json',[r['id'] for r in accepted])
    write_json(root/'summary.json',{'selected_errors':4000,'generated':cursor,'counts':dict(totals),
        'datasets':dict(Counter(r['dataset'] for r in accepted)),
        'unique_questions':len({r['question_group'] for r in accepted}),
        'same_question_grouped_folds':True,'existing_text_duplicates_excluded':True,
        'taxonomy_status':'not yet annotated; do not treat strict numerical failure as a confirmed mechanism',
        'feature_count':len(featured),'official_gsm_test_loaded':False})
    state.update(phase='complete',feature_errors=len(featured),updated_unix=time.time())
    write_json(root/'progress.json',state)
    print('COLLECTION_COMPLETE',json.dumps(state),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    try: main(args.smoke)
    except Exception as error:
        write_json(ROOT/'failure.json',{'error':repr(error),'updated_unix':time.time()})
        print('FAILED',repr(error),flush=True)
        raise
