from pathlib import Path
import json,hashlib,shutil
out=Path(__file__).resolve().parents[1];base=out.parents[1];old130=base/'runs/pilot_v1';old500=base/'runs/expansion_500_v1'
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
def save(n,x):p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def lines(n,x):p=out/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
assert not (out/'protocol.json').exists()
manifest=jl(old130/'sample_manifest.jsonl');lookup={r['source_id']:r for r in manifest};sources={};packets=[]
for stage in ['dev','eval']:
 for c in 'EG':
  for p in jl(old130/f'annotation/packets/{stage}/{c}.jsonl'):
   sid=p['source_id'];m=lookup[sid];role={'development':'development','eval_query':'query','eval_gallery':'gallery'}[m['pilot_role']]
   sources[sid]={'id':sid,'question':p['question'],'response':p['response'],'reference':p['reference'],'response_sha256':p['response_sha256'],'question_sha256':m['question_sha256'],'question_key':m['question_key'],'role':role,'dataset':'pilot130'}
   packets.append({'job_id':f'pilot130-{c}-{sid}','dataset':'pilot130','condition':c,**p})
for s in jl(old500/'sample.jsonl'):sources[s['id']]={**s,'dataset':'expansion500'}
for c in 'EG':
 for p in jl(old500/f'packets/{c}.jsonl'):packets.append({'job_id':f'expansion500-{c}-{p["source_id"]}','dataset':'expansion500','condition':c,**p})
assert len(sources)==630 and len({s['question'] for s in sources.values()})==630;assert len(packets)==1260
lines('sample.jsonl',sources.values());lines('annotation_packets.jsonl',packets)
for n in ['annotation','calls','schemas','datasets','analysis']:(out/n).mkdir(exist_ok=True)
for dataset in ['pilot130','expansion500']:
 rows=[s for s in sources.values() if s['dataset']==dataset];lines(f'datasets/{dataset}/sample.jsonl',rows)
cfg=json.load(open(old500/'config.json'));cfg['annotation']['annotator']={'provider':'OpenAI','model_identifier':'gpt-6-astra','route':'authenticated Codex CLI exec, ephemeral fresh session per request','reasoning_effort':'medium','exact_snapshot_known':False,'decoding_settings':'temperature/top_p not set; CLI defaults except reasoning_effort=medium','condition_context_isolated':True};save('config.json',cfg)
nullable=lambda t:{'type':[t,'null']}
common={'source_id':{'type':'string'},'response_sha256':{'type':'string'},'status':{'type':'string','enum':cfg['annotation']['statuses']},'reference_review_needed':{'type':'boolean'}}
props={'E':{**common,'explanation':nullable('string')},'G':{**common,'evidence_quote':nullable('string'),'evidence_start':nullable('integer'),'evidence_end':nullable('integer'),'evidence_occurrence':nullable('integer'),'question_evidence_quote':nullable('string'),'observed_error':nullable('string'),'violated_condition':nullable('string'),'error_description':nullable('string'),'reference_review_reason':nullable('string'),'secondary_errors':{'type':'array','items':{'type':'string'}},'uncertainty_note':nullable('string')}}
for c,p in props.items():save(f'schemas/{c}.json',{'type':'object','properties':p,'required':list(p),'additionalProperties':False})
for n in ['prompts/E_v1.txt','prompts/G_v1.txt']:
 p=out/'original_prompts'/Path(n).name;p.parent.mkdir(exist_ok=True);shutil.copy2(base/n,p)
# Freeze controls before the first experimental annotation or judge call.
controls=jl(base/'runs/expansion_500_v2/judge_controls.jsonl');lines('judge_controls.jsonl',controls)
judge='''You are comparing observable mathematical errors. Use no tools or outside resources. For each case, check its solution against its question; the reference is fallible and must not override the question. Distinguish an erroneous relation, a numerical evaluation error of a correctly written expression, an unsupported final number after a correct formula, and answering the wrong requested quantity. A contradictory leading answer marker with an otherwise correct derivation is an answer conflict, not evidence of an invented math mistake. Decide same only if BOTH cases have the same specific observable error mechanism. Different stories, numerical values, or arithmetic operators do not by themselves make mechanisms different. Decide different if both have diagnosable errors and their mechanisms differ. Decide uncertain if either case has no clear mathematical error, a materially ambiguous interpretation, or only answer-marker conflict. Never invent an unshown calculation. Return concise JSON: case1_error, case2_error, judgment (same/different/uncertain), reason. No method names, embedding scores, cluster labels or generated annotations are supplied.\n'''
(out/'judge_prompt.txt').write_text(judge);jp={k:{'type':'string'} for k in ['case1_error','case2_error','reason']};jp['judgment']={'type':'string','enum':['same','different','uncertain']};save('schemas/judge.json',{'type':'object','properties':jp,'required':list(jp),'additionalProperties':False})
checkjobs=[]
for c in controls:
 prompt=judge
 for i,s in enumerate([c['a'],c['b']]):prompt+=f'\nCASE {i+1}\nQUESTION: {s["question"]}\nSOLUTION: {s["response"]}\nREFERENCE: {s["reference"]}\n'
 checkjobs.append({'job_id':c['id'],'prompt':prompt,'expected':c['expected'],'base_pair':c['base_pair'],'schema':'judge'})
lines('control_jobs.jsonl',checkjobs)
files={}
for folder in [old130,old500,base/'runs/expansion_500_v2']:
 for p in folder.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:files[str(p.relative_to(base))]=hashlib.sha256(p.read_bytes()).hexdigest()
save('baseline_hashes.json',files)
save('protocol.json',{'model':'gpt-6-astra','authorization':'user explicitly approved separate Codex GPT-6-astra sessions','reasoning_effort':'medium','route':'Codex CLI ChatGPT login; not OpenAI API key','exact_model_snapshot':'not exposed; record requested model alias and CLI version','sample':'original130 plus same expansion500, no replacement or resampling','unique_questions':630,'annotation_requests':1260,'annotation_prompt_policy':'original E/G text retained; JSON output schema and no-tools execution wrapper added; unique exact quote offsets recomputed in code; raw model output also validated and preserved','max_annotation_attempts':2,'context_isolation':'one case, one condition, one fresh ephemeral session per attempt; no previous outputs or other cases','concurrency':4,'embedding':'same BAAI/bge-m3 revision5617a9f61b028005a4858fdac845db406aefb181; dense1024 CLS normalized','primary_clusters':'pilot evaluation100 only; expansion500; separate native and E/G common cohorts; k4,6,8,10,15,20 seeds42,43,44; primary display10 seed42','retrieval':'pilot20queries80gallery; expansion100queries400gallery; top5 plus random; unique pairs judged once, original questions/solutions/references only','judge_gate':'36 controls (18 underlying pairs in both orders); at least10/12 each class and at least16/18 both orders agree, no invalid judgment output; fail => do not publish automated retrieval accuracy','judge_is_independent_model':False,'human_ground_truth':False,'historical_comparison_limits':'different serving route, structured output enforcement, isolation for old pilot; cannot attribute every difference solely to model weights','stage':'prepared'})
print('Prepared630 original cases,1260 isolated annotation jobs,36 frozen judge controls')
