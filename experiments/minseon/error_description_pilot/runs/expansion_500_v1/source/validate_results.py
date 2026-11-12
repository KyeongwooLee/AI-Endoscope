from pathlib import Path
import json,hashlib,sys,collections
import numpy as np
out=Path(__file__).resolve().parents[1];base=out.parents[1];sys.path.insert(0,str(base));from edp.annotation import validate_output
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
sample=jl(out/'sample.jsonl');ids={r['id'] for r in sample};byid={r['id']:r for r in sample};assert len(ids)==len(sample)==500
assert len({r['question_key'] for r in sample})==len({r['question'] for r in sample})==500
prev=jl(base/'runs/pilot_v1/sample_manifest.jsonl');oldkeys={r['question_key'] for r in prev};oldhashes={r['question_sha256'] for r in prev};assert not oldkeys&{r['question_key'] for r in sample};assert not oldhashes&{r['question_sha256'] for r in sample}
qs={r['id'] for r in sample if r['role']=='query'};gs=ids-qs;assert len(qs)==100 and len(gs)==400 and not qs&gs
cfg=json.load(open(out/'config.json'));ann={};emb_ids={};nvec={}
for c in 'EG':
 rs=jl(out/f'annotation/{c}.jsonl');assert len(rs)==500 and {r['source_id'] for r in rs}==ids;ann[c]={r['source_id']:r for r in rs};pack={r['source_id']:r for r in jl(out/f'packets/{c}.jsonl')}
 for sid,r in ann[c].items():
  assert r['response_sha256']==hashlib.sha256(byid[sid]['response'].encode()).hexdigest()
  errors,_,_=validate_output(c,r['output'],pack[sid],cfg);assert r['valid']==(not errors)
for c in 'REGQ':
 inputs=jl(out/f'embeddings/{c}_inputs.jsonl');z=np.load(out/f'embeddings/{c}.npz');ss=z['ids'].tolist();x=z['vectors'];assert x.shape==(len(ss),1024) and np.isfinite(x).all();assert np.max(abs(np.linalg.norm(x,axis=1)-1))<1e-3;assert [r['source_id'] for r in inputs]==ss and len(set(ss))==len(ss);emb_ids[c]=set(ss);nvec[c]=len(ss)
 expected=ids if c in 'RQ' else {sid for sid,r in ann[c].items() if r['valid'] and r['output']['status']=='supported_error'};assert set(ss)==expected
 for r in inputs:
  sid=r['source_id'];s=byid[sid];text=f"Question: {s['question']}\nSolution: {s['response']}" if c=='R' else s['question'] if c=='Q' else ann[c][sid]['output']['explanation' if c=='E' else 'error_description'];assert r['text']==text;assert hashlib.sha256(text.encode()).hexdigest()==r['text_sha256']
common=emb_ids['E']&emb_ids['G'];retr=jl(out/'retrieval.jsonl');seen=set();actualpairs=set()
for r in retr:
 k=(r['cohort'],r['method'],r['query_id']);assert k not in seen;seen.add(k);assert r['query_id'] in qs
 assert len(r['top5']) in [0,5]
 if r['status']=='ranked':assert len(r['top5'])==5
 for i,s in enumerate(r['top5']):
  assert s['rank']==i+1 and s['id'] in gs
  if r['method']!='Random':assert s['id'] in emb_ids[r['method']] and r['query_id'] in emb_ids[r['method']]
  if r['cohort']=='common':assert s['id'] in common and r['query_id'] in common
  actualpairs.add((r['query_id'],s['id']))
for co,n in [('native',100),('common',len(qs&common))]:
 for m in ['R','E','G','Q','Random']:assert sum(r['cohort']==co and r['method']==m for r in retr)==n
keys=jl(out/'pair_key.jsonl');assert {(r['query_id'],r['gallery_id']) for r in keys}==actualpairs
judges=jl(out/'ai_pair_judgments.jsonl');assert len(judges)==len(keys) and {r['pair_id'] for r in judges}=={r['pair_id'] for r in keys};assert all(r['judgment'] in ['same','different','uncertain'] for r in judges)
before=json.load(open(out/'pilot_preservation_hashes.json'));changed=[f for f,h in before.items() if hashlib.sha256((base/'runs/pilot_v1'/f).read_bytes()).hexdigest()!=h];assert not changed
v={'passed':True,'source_questions':500,'source_responses':500,'prior130_question_overlap':0,'queries':100,'gallery':400,'annotation_requests':1000,'embedded_vectors':nvec,'common_questions':len(common),'common_queries':len(qs&common),'common_gallery':len(gs&common),'unique_pairs':len(keys),'complete_AI_judgments':len(judges),'missing_AI_judgments':0,'unchanged_prior_files':len(before),'human_ground_truth':False,'notes':'Shape, norm, exact source text/hash, eligibility, query/gallery roles, pair reuse, judgment completeness and old-run preservation checked.'}
(out/'validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(v),flush=True)
