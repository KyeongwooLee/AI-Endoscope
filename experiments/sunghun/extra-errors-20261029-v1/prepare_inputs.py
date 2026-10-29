"""Freeze a new, grouped, reproducible sampling pool; never rewrite old labels."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT=Path(__file__).resolve().parent
def digest(value): return hashlib.sha256(value.encode()).hexdigest()
def group(question): return digest(' '.join(unicodedata.normalize('NFKC',question).split()))
def read_jsonl(path): return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

baseline=[]
for name in ['gsm8k_answers.jsonl','svamp_answers.jsonl']:
    baseline += read_jsonl(ROOT/'baseline_labels'/name)
old_gsm={r['question_id'] for r in baseline if r['dataset']=='gsm8k'}
svamp={r['question_id']:r for r in baseline if r['dataset']=='svamp'}
existing_pairs=sorted({group(r['question'])+':'+digest(r['response']) for r in baseline})
questions=json.loads((ROOT/'starter/AI_STARTER/questions.json').read_text(encoding='utf-8'))
rows=[]
for r in questions:
    if r['id'] in old_gsm: continue
    question_group=group(r['question'])
    bucket=int(digest('2026102901:'+question_group)[:8],16)%10
    rows.append({'question_id':r['id'],'dataset':'gsm8k','question':r['question'],
                 'reference':r['gold'],'reference_equation':None,'question_group':question_group,
                 'fold':'fit' if bucket<6 else 'validation' if bucket<8 else 'test'})
for r in svamp.values():
    rows.append({key:r.get(key) for key in ['question_id','dataset','question','reference','reference_equation','fold']}
                | {'question_group':group(r['question'])})
assert len(questions)==7473 and len(old_gsm)==500 and len(svamp)==1000
assert len(rows)==7973 and len({(r['dataset'],r['question_id']) for r in rows})==7973
assert not any(r['question_id'] in old_gsm for r in rows if r['dataset']=='gsm8k')
group_folds={}
for r in rows:
    previous=group_folds.setdefault(r['question_group'],r['fold'])
    assert previous==r['fold'],'Conflicting group folds'
config=json.loads((ROOT/'source/gsm8k-logit-study/configs/rtx3090.fp32-pinned.json').read_text(encoding='utf-8'))
config['study']='qwen-extra-errors-20261029-v1'
config['seed']=2026102901
config['target_new_eligible_errors']=4000
config['maximum_generated_responses']=40000
config['generation'].update(batch_size=32,temperature=0.7,top_p=0.95,top_k=0,do_sample=True)
config['source_mix']={'gsm8k':4,'svamp':1}
config['selection']='first 4000 distinct question/text pairs that are parsed, nontruncated and strict-numerically incorrect'
config['extraction'].update(checkpoints=[0,32,64,128],primary_layer=14,readout_layers=[7,10,14,28],
    readout_prefix_tokens=129,projection_chunk_tokens=16,top_k=5)
config['runtime_note']='Batched sampling; same pinned model, FP32/SDPA, prompt, T=.7/top_p=.95/top_k=0 and stopping rule. Not token-identical to old serial runs.'
folder=ROOT/'input'
folder.mkdir(exist_ok=True)
for name,obj in [('questions.json',rows),('existing_pairs.json',existing_pairs),('config.json',config)]:
    (folder/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'questions':dict(Counter(r['dataset'] for r in rows)),'excluded_old_gsm_sampling_questions':500,
         'existing_response_pairs':len(existing_pairs),'target_new_errors':4000,
         'source_revision':'origin/sunghun a178c22','old_labels_unchanged':True,
         'sampling_policy':'New GSM train question pool; SVAMP keeps original grouped folds; fixed 4:1 response mix; repeated stochastic attempts permitted.'}
(folder/'input_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary))
