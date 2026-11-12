from pathlib import Path
import json,hashlib,sys,zipfile,collections
import numpy as np
out=Path(__file__).resolve().parents[1];old=out.parent/'expansion_500_v1';base=out.parents[1];sys.path.insert(0,str(base));from edp.annotation import validate_output
from exact_quote import attach_unique_quote_offsets
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.load(open(out/'baseline_hashes.json'));changed=[n for n,h in baseline.items() if not (old/n).is_file() or sha(old/n)!=h];assert not changed,changed
assert sha(out/'sample.jsonl')==sha(old/'sample.jsonl');assert sha(out/'annotation/E.jsonl')==sha(old/'annotation/E.jsonl')
sample={r['id']:r for r in jl(out/'sample.jsonl')};assert len(sample)==500 and len({r['question'] for r in sample.values()})==500
cfg=json.load(open(out/'config.json'));ann={c:{r['source_id']:r for r in jl(out/f'annotation/{c}.jsonl')} for c in 'EG'};oldg={r['source_id']:r for r in jl(old/'annotation/G.jsonl')};repairs=json.load(open(out/'offset_repairs.json'));fixed={r['source_id'] for r in repairs};assert len(fixed)==77
for sid,r in ann['G'].items():
 if sid not in fixed:assert r==oldg[sid]
 else:
  assert r['output']==attach_unique_quote_offsets(oldg[sid]['output'],sample[sid]['response'])
  for k,v in oldg[sid]['output'].items():
   if k not in ['evidence_start','evidence_end','evidence_occurrence']:assert r['output'][k]==v
for c in 'EG':
 ps={r['source_id']:r for r in jl(out/f'packets/{c}.jsonl')};assert len(ann[c])==500
 for sid,r in ann[c].items():
  errs,_,_=validate_output(c,r['output'],ps[sid],cfg);assert r['valid']==(not errs),(c,sid)
counts={}
for c in 'REGQ':
 z=np.load(out/f'embeddings/{c}.npz');ids=z['ids'].tolist();x=z['vectors'];inputs=jl(out/f'embeddings/{c}_inputs.jsonl');assert ids==[r['source_id'] for r in inputs];assert x.shape==(len(ids),1024);assert np.all(np.isfinite(x));assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
 expected=set(sample) if c in 'RQ' else {sid for sid,r in ann[c].items() if r['valid'] and r['output']['status']=='supported_error'};assert set(ids)==expected
 for r in inputs:
  s=sample[r['source_id']];t=f"Question: {s['question']}\nSolution: {s['response']}" if c=='R' else s['question'] if c=='Q' else ann[c][r['source_id']]['output']['explanation' if c=='E' else 'error_description'];assert t==r['text'];assert hashlib.sha256(t.encode()).hexdigest()==r['text_sha256']
 counts[c]=len(ids)
assert counts=={'R':500,'E':447,'G':338,'Q':500}
freeze=json.load(open(out/'checks_frozen.json'));assert sha(out/'judge_controls.jsonl')==freeze['controls_sha256'];assert sha(out/'prompts.json')==freeze['prompts_sha256']
ps=jl(out/'check_packets.jsonl');results=jl(out/'check_outputs.jsonl');assert len(results)==len(ps)==582;assert {r['id'] for r in results}=={r['id'] for r in ps};assert len({r['id'] for r in results})==582
repairedg=set(np.load(out/'embeddings/G.npz')['ids'].tolist());e=set(np.load(out/'embeddings/E.npz')['ids'].tolist());common=e&repairedg;assert len(common)==326
roles=collections.Counter(r['role'] for r in sample.values());assert roles=={'query':100,'gallery':400}
for r in jl(out/'retrieval.jsonl'):
 assert sample[r['query_id']]['role']=='query'
 if r['top5']:
  assert len(r['top5'])==5 and len({x['id'] for x in r['top5']})==5
  for x in r['top5']:assert sample[x['id']]['role']=='gallery' and sample[x['id']]['question']!=sample[r['query_id']]['question']
 if r['cohort']=='common':assert r['query_id'] in common and all(x['id'] in common for x in r['top5'])
quarantine={r['source_id'] for r in json.load(open(out/'confirmed_issue_registry.json'))};assert len(quarantine)==7
for g in json.load(open(out/'sensitivity/G_clusters.json')):assert not (set(g['members'])&quarantine)
assert json.load(open(out/'evaluation_status.json'))['automatic_retrieval_accuracy_published'] is False
assert (out/'analysis_report.html').exists() and (out/'comparison.png').exists()
val={'passed':True,'baseline_files_unchanged':len(baseline),'source_questions_unchanged':500,'offset_only_repairs':77,'description_changes':0,'embedded_counts':counts,'common_n':326,'check_requests_complete':582,'unparsed_AI_outputs':sum(r['output'] is None for r in results),'known_issue_quarantine_n':7,'human_ground_truth':False,'automatic_accuracy_claim':False}
(out/'validation.json').write_text(json.dumps(val,indent=2));status={'experiment_complete':True,'offset_repair_complete':True,'clustering_and_retrieval_complete':True,'reference_screening_complete':True,'control_checks_complete':True,'full_new_retrieval_judging':'not_run_because_basic_controls_failed','report_complete':True};(out/'analysis_status.json').write_text(json.dumps(status,indent=2))
files=[p for p in sorted(out.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_manifest.json'];manifest={'files':{str(p.relative_to(out)):sha(p) for p in files}};(out/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2));files.append(out/'artifact_manifest.json');archive=out.parent/'expansion_500_v2_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,arcname=out.name+'/'+str(p.relative_to(out)))
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
archive.with_suffix('.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
print(json.dumps({'validation':val,'archive':str(archive),'archive_bytes':archive.stat().st_size,'files':len(files)},ensure_ascii=False))
