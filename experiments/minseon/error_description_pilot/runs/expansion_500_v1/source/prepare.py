from pathlib import Path
import json,hashlib,random,sys,collections
base=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parents[1];sys.path.insert(0,str(base))
from edp.annotation import build_packet,prompt_info
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def lines(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
old=base/'runs/pilot_v1';raw=base/'data/qwen-gsm8k-self-errors-v1/errors.jsonl';assert hashlib.sha256(raw.read_bytes()).hexdigest()=='cc54b199164cdef0c42365a1b806f3dc1e99580762d3b6976dabfba3d2c1993b'
rows=jl(raw);prev=jl(old/'sample_manifest.jsonl');excluded=set(json.load(open(old/'prior_question_exclusions.json')))|{r['question_key'] for r in prev};oldtexts={r['question_sha256'] for r in prev}
groups=collections.defaultdict(list)
for r in rows:
 q=f"gsm8k:{r['split']}:{r['pid']}"
 if q not in excluded and r['finish']=='stop' and hashlib.sha256(r['question'].encode()).hexdigest() not in oldtexts:groups[q].append(r)
rng=random.Random(20261112);keys=rng.sample(sorted(groups),500);selected=[];seen=set()
for i,k in enumerate(keys):
 r=rng.choice(sorted(groups[k],key=lambda x:x['id']));h=hashlib.sha256(r['question'].encode()).hexdigest();assert h not in seen;seen.add(h)
 selected.append({**{z:r[z] for z in ['id','split','pid','question','response','reference']},'role':'query' if i<100 else 'gallery','question_key':k,'question_sha256':h,'response_sha256':hashlib.sha256(r['response'].encode()).hexdigest()})
lines(out/'sample.jsonl',selected)
cfg=json.load(open(base/'config.json'));cfg['run_dir']='runs/expansion_500_v1';cfg['annotation']['annotator']={'route':'local_transformers_causal_lm','provider':'Qwen','model_identifier':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554','decoding_settings':{'do_sample':False,'max_new_tokens':1000,'dtype':'float16'},'condition_context_isolated':True,'note':'Each item and condition receives a fresh prompt; no other outputs in context. Model differs from pilot_v1.'};cfg.update({'protocol_name': 'error-description-expansion-500', 'protocol_version': 'expansion500-v1', 'sampling': {'seed': 20261112, 'eval_questions': 500, 'eval_queries': 100, 'eval_gallery': 400, 'dev_uniform': 0, 'dev_enriched_strata': [], 'method': 'Python random.Random: uniform question groups, then one response per question; prior audited questions and pilot130 excluded'}, 'split': {'method': 'fixed random question sample split into first100 queries and remaining400 gallery', 'legacy_sae_partition_policy': 'union; not a new SAE confirmation test'}, 'retrieval': {'k': 5, 'random_seed': 20261112, 'methods': ['R', 'E', 'G', 'Q', 'Random'], 'cohorts': ['native', 'common']}, 'bootstrap': {'draws': 2000, 'seed': 20261112, 'level': 0.95, 'contrasts': [['G', 'E'], ['G', 'R'], ['E', 'R']]}, 'clustering': {'k_values': [4, 6, 8, 10, 15, 20], 'seeds': [42, 43, 44], 'n_init': 10, 'representations': ['R', 'E', 'G', 'Q'], 'display_k': 10, 'display_seed': 42, 'cohorts': ['native', 'common'], 'algorithms': ['KMeans', 'average-link cosine']}});cfg['embedding']['batch_size']=4;write(out/'config.json',cfg)
for cond,pr in prompt_info(cfg,base).items():lines(out/f'packets/{cond}.jsonl',[build_packet(r,'eval',cond,pr,cfg) for r in selected])
write(out/'protocol.json',{'date_kst':'2026-11-12','seed':20261112,'sample_questions':500,'queries':100,'gallery':400,'eligible_questions':len(groups),'exclusion_question_keys':len(excluded),'overlap_previous_130':0,'prior_audited_questions_excluded':True,'sampling':'uniform question sample then uniform response within question, finish=stop; union of old SAE partitions; NOT an untouched SAE confirmation set','annotation_model':cfg['annotation']['annotator'],'embedding':cfg['embedding'],'clustering':{'k':[4,6,8,10,15,20],'seeds':[42,43,44],'display_k':10,'display_seed':42,'algorithms':['KMeans n_init=10','average-link cosine'],'fit_on_original_1024_dimensions':True},'limits':['Annotator changed; no causal sample-size comparison with pilot.','Local judge will be AI provisional; no human truth claims.'],'status':'prepared'})
print('prepared',len(selected),'eligible',len(groups))
