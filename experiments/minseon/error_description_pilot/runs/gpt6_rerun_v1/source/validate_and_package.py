from pathlib import Path
import json,sys,hashlib,zipfile,collections
import numpy as np
root=Path(__file__).resolve().parents[1];base=root.parents[1];sys.path.insert(0,str(base))
from edp.annotation import validate_output
from run_gpt import unexpected
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baselines=json.load(open(root/'baseline_hashes.json'));changed=[p for p,h in baselines.items() if not (base/p).is_file() or sha(base/p)!=h];assert not changed,changed
samples=jl(root/'sample.jsonl');byid={r['id']:r for r in samples};assert len(samples)==len(byid)==len({r['question'] for r in samples})==630
packets={r['job_id']:r for r in jl(root/'annotation_packets.jsonl')};anns=jl(root/'annotation/all.jsonl');assert len(anns)==len(packets)==1260;assert {r['job_id'] for r in anns}==set(packets)
cfg=json.load(open(root/'config.json'));allthreads=set();provider_calls=0
for meta_path in (root/'calls').glob('*/metadata.json'):
 meta=json.load(open(meta_path));assert meta['requested_model']=='gpt-6-astra';assert meta['command'][meta['command'].index('-m')+1]=='gpt-6-astra';assert meta['returncode']==0;assert not any(unexpected(e) for e in meta['non_message_items']);assert sha(meta_path.parent/'prompt.txt')==meta['prompt_sha256'];events=jl(meta_path.parent/'events.jsonl');ids=[e['thread_id'] for e in events if e['type']=='thread.started'];assert len(ids)==1 and ids[0] not in allthreads;allthreads.add(ids[0]);assert any(e['type']=='turn.completed' for e in events);provider_calls+=1
for r in anns:
 p=packets[r['job_id']];sid=r['source_id'];s=byid[sid];assert p['question']==s['question'] and p['response']==s['response'] and p['reference']==s['reference'];assert hashlib.sha256(s['response'].encode()).hexdigest()==p['response_sha256'];errors,_,_=validate_output(r['condition'],r['output'],p,cfg);assert r['valid']==(not errors)
 last=r['attempts'][-1];raw=json.load(open(root/last['call_dir']/'response.json'));assert raw==last['raw_output']
 for k,v in raw.items():
  if r['condition']=='G' and k in ['evidence_start','evidence_end','evidence_occurrence']:continue
  assert r['output'][k]==v,(sid,k)
 first_prompt=(root/f'calls/{r["job_id"]}-a1/prompt.txt').read_text();assert first_prompt.endswith(p['rendered_prompt']);assert 'Do not call tools' in first_prompt
counts={};pair_total=0
for dataset,n in [('pilot130',130),('expansion500',500)]:
 out=root/'datasets'/dataset;ss=jl(out/'sample.jsonl');assert len(ss)==n;lookup={r['id']:r for r in ss};role=collections.Counter(r['role'] for r in ss);assert role==({'development':30,'query':20,'gallery':80} if n==130 else {'query':100,'gallery':400})
 old=base/'runs'/('pilot_v1' if n==130 else 'expansion_500_v2');ann={c:{r['source_id']:r for r in jl(out/f'annotation/{c}.jsonl')} for c in 'EG'};vectors={};counts[dataset]={}
 for c in 'REGQ':
  z=np.load(out/f'embeddings/{c}.npz');ids=z['ids'].tolist();x=z['vectors'];inp=jl(out/f'embeddings/{c}_inputs.jsonl');assert ids==[r['source_id'] for r in inp];assert x.shape==(len(ids),1024);assert np.isfinite(x).all();assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
  expected=set(lookup) if c in 'RQ' else {sid for sid,r in ann[c].items() if r['valid'] and r['output']['status']=='supported_error'};assert set(ids)==expected
  for r in inp:
   s=lookup[r['source_id']];text=f'Question: {s["question"]}\nSolution: {s["response"]}' if c=='R' else s['question'] if c=='Q' else ann[c][s['id']]['output']['explanation' if c=='E' else 'error_description'];assert r['text']==text;assert r['text_sha256']==hashlib.sha256(text.encode()).hexdigest()
  vectors[c]=dict(zip(ids,x));counts[dataset][c]=len(ids)
  if c in 'RQ':
   paths=[old/f'embeddings/{phase}/{c}.npz' for phase in ['dev','eval']] if n==130 else [old/f'embeddings/{c}.npz'];oldmap={}
   for p in paths:
    z=np.load(p);oldmap.update(dict(zip(z['ids'].tolist(),z['vectors'])))
   assert all(np.array_equal(v,oldmap[sid]) for sid,v in vectors[c].items())
 for c in 'EG':
  p=old/f'embeddings/eval/{c}.npz' if n==130 else old/f'embeddings/{c}.npz';z=np.load(p);vectors['old_'+c]=dict(zip(z['ids'].tolist(),z['vectors']))
 cohorts=json.load(open(out/'cohorts.json'));keys=jl(out/'pair_key.jsonl');keyset={(p['query_id'],p['gallery_id']) for p in keys};assert len(keys)==len(keyset);pair_total+=len(keys);seen=set()
 for r in jl(out/'retrieval.jsonl'):
  eligible=set(cohorts[r['cohort']]['ids']);q=r['query_id'];assert lookup[q]['role']=='query' and q in eligible;c=r['method'];gs=[s['id'] for s in ss if s['role']=='gallery' and s['id'] in eligible and (c=='Random' or s['id'] in vectors[c])];assert r['gallery_n']==len(gs);actual=[p['id'] for p in r['top5']];assert len(actual)==len(set(actual))
  if r['top5']:
   assert len(actual)==min(5,len(gs)) and all(g in gs for g in actual)
   if c!='Random':
    scores=np.array([vectors[c][g] for g in gs])@vectors[c][q];order=sorted(range(len(gs)),key=lambda i:(-float(scores[i]),gs[i]))[:5];assert actual==[gs[i] for i in order]
  for g in actual:assert lookup[q]['question']!=lookup[g]['question'];seen.add((q,g))
 assert seen==keyset
 status=json.load(open(out/'analysis_status.json'));status['judging_complete']=json.load(open(root/'control_results.json'))['passed'];(out/'analysis_status.json').write_text(json.dumps(status,indent=2))
gate=json.load(open(root/'control_results.json'));judgments=jl(root/'pair_judgments.jsonl');assert len(judgments)==len({r['pair_id'] for r in judgments})
if gate['passed']:assert len(judgments)==pair_total
screen=jl(root/'screen_outputs.jsonl');assert len(screen)==len({r['source_id'] for r in screen})==500;assert {r['source_id'] for r in screen}=={s['id'] for s in samples if s['dataset']=='expansion500'}
for r in screen:
 assert r['model']=='gpt-6-astra' and r['reference_provided'] is False
 assert r['output']==json.load(open(root/f'calls/{r["job_id"]}-a1/response.json'))
assert (root/'analysis_report.html').is_file();assert (root/'cluster_comparison.png').is_file()
validation={'passed':True,'unchanged_baseline_files':len(baselines),'fixed_unique_questions':630,'annotation_records':1260,'annotation_models':sorted({r['model'] for r in anns}),'valid_annotations':sum(r['valid'] for r in anns),'invalid_annotations':sum(not r['valid'] for r in anns),'embedding_counts':counts,'unique_retrieval_pairs':pair_total,'GPT_judged_pairs':len(judgments),'reference_free_GPT_diagnoses':500,'recorded_fresh_GPT_sessions':provider_calls,'no_model_tool_activity':True,'human_ground_truth':False,'limitations':'Structural/source/retrieval/provenance checks, not proof of semantic annotation accuracy.'}
(root/'validation.json').write_text(json.dumps(validation,indent=2));(root/'status.json').write_text(json.dumps({'completed':True,'annotation_complete':True,'embedding_analysis_complete':True,'judge_gate_passed':gate['passed'],'reference_free_diagnosis_complete':True,'report_complete':True},indent=2))
files=[p for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json' and p.name not in ['pipeline.log','validate_and_package.log']]
manifest={'files':{str(p.relative_to(root)):sha(p) for p in files}};(root/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(root/'artifact_manifest.json');archive=root.parent/'gpt6_rerun_v1_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,arcname=root.name+'/'+str(p.relative_to(root)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n');print(json.dumps({'validation':validation,'archive':str(archive),'bytes':archive.stat().st_size,'files':len(files)},ensure_ascii=False),flush=True)
