from pathlib import Path
import json,collections,csv
import numpy as np
root=Path(__file__).resolve().parents[1];base=root.parents[1]
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
import subprocess,sys
if not (root/'screen_outputs.jsonl').exists() or len(jl(root/'screen_outputs.jsonl'))<500:
 subprocess.run([sys.executable,'-u',str(root/'source/run_gpt.py'),'screen'],check=True)
if not (root/'real_audit_outputs.jsonl').exists() or len(jl(root/'real_audit_outputs.jsonl'))<4:
 subprocess.run([sys.executable,'-u',str(root/'source/run_gpt.py'),'real_audit'],check=True)
gate=json.load(open(root/'control_results.json'));judgments={}
if gate['passed']:
 mapping={r['blind_id']:r for r in jl(root/'pairs_batch_mapping.jsonl')};outputs=jl(root/'pairs_batch_outputs.jsonl');jobs=jl(root/'pairs_batch_jobs.jsonl');assert len(outputs)==len(jobs)
 for b in outputs:
  for o in b['output']['results']:
   pair_id=mapping[o['pair_id']]['pair_id'];assert pair_id not in judgments;judgments[pair_id]={'pair_id':pair_id,**{k:v for k,v in o.items() if k!='pair_id'},'reviewer_type':'AI_provisional','model':'gpt-6-astra','batch':b['job_id']}
 assert len(judgments)==len(mapping)
(root/'pair_judgments.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in judgments.values()))
results={};exports=[]
for dataset in ['pilot130','expansion500']:
 out=root/'datasets'/dataset;keys=jl(out/'pair_key.jsonl');lookup={(r['query_id'],r['gallery_id']):r['pair_id'] for r in keys};sample={r['id']:r for r in jl(out/'sample.jsonl')};retr=jl(out/'retrieval.jsonl');data={}
 for cohort in ['native','common','historical_common']:
  rs=[r for r in retr if r['cohort']==cohort];methods=sorted({r['method'] for r in rs});summary={};perquery={}
  for method in methods:
   ms=sorted([r for r in rs if r['method']==method],key=lambda r:r['query_id']);qs=[]
   for r in ms:
    counts=collections.Counter(judgments.get(lookup[(r['query_id'],t['id'])],{}).get('judgment','unjudged') for t in r['top5']);qs.append({'query_id':r['query_id'],'same':counts['same'],'different':counts['different'],'uncertain':counts['uncertain'],'unjudged':counts['unjudged'],'returned':len(r['top5'])})
   n=len(qs);counts={k:sum(q[k] for q in qs) for k in ['same','different','uncertain','unjudged','returned']};valid=gate['passed'] and counts['unjudged']==0
   summary[method]={'queries':n,**counts,'denominator':5*n,'AI_same_fraction':counts['same']/(5*n) if valid and n else None,'uncertain_as_same_upper':(counts['same']+counts['uncertain'])/(5*n) if valid and n else None,'query_coverage':sum(q['returned']>0 for q in qs)/n if n else None};perquery[method]=qs
  contrasts={}
  if gate['passed']:
   for a,b in [('G','E'),('E','R'),('G','R')]+([('E','old_E'),('G','old_G')] if cohort=='historical_common' else []):
    assert [q['query_id'] for q in perquery[a]]==[q['query_id'] for q in perquery[b]]
    delta=np.array([(x['same']-y['same'])/5 for x,y in zip(perquery[a],perquery[b])]);n=len(delta)
    if n:
     rng=np.random.default_rng(20261112);draw=delta[rng.integers(n,size=(2000,n))].mean(axis=1);contrasts[a+'-'+b]={'difference':float(delta.mean()),'ci95_low':float(np.quantile(draw,.025)),'ci95_high':float(np.quantile(draw,.975)),'resampling_unit':'query','draws':2000,'note':'Does not account for AI judging mistakes or all shared-gallery dependence.'}
  data[cohort]={'methods':summary,'contrasts':contrasts,'per_query':perquery}
 results[dataset]=data;save(out/'evaluation.json',data)
 for p in keys:
  j=judgments.get(p['pair_id'],{});a=sample[p['query_id']];b=sample[p['gallery_id']];exports.append({'dataset':dataset,'pair_id':p['pair_id'],'query_id':p['query_id'],'gallery_id':p['gallery_id'],'judgment':j.get('judgment'),'case1_error':j.get('case1_error'),'case2_error':j.get('case2_error'),'reason':j.get('reason'),'query_question':a['question'],'query_solution':a['response'],'gallery_question':b['question'],'gallery_solution':b['response'],'returned_by':json.dumps(p['returned_by'])})
 print(dataset,{c:{m:{k:v for k,v in s.items() if k in ['queries','same','uncertain','denominator','AI_same_fraction']} for m,s in d['methods'].items()} for c,d in data.items()},flush=True)
with (root/'pair_reviews.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(exports[0]));w.writeheader()
 for r in exports:w.writerow({k:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v for k,v in r.items()})
save(root/'evaluation.json',{'gate':gate,'datasets':results,'human_ground_truth':False,'judge_annotation_same_model':True,'judged_unique_pairs':len(judgments)})
# Compare annotation decisions to pre-existing AI original-text diagnoses, without treating them as human gold.
ann={c:{r['source_id']:r for r in jl(root/'annotation/all.jsonl') if r['condition']==c} for c in 'EG'};audit=json.load(open(base/'runs/expansion_500_v2/annotation_audit_original_diagnoses.json'));cases=[]
for r in audit:
 sid=r['source_id'];cases.append({**r,'GPT_E':ann['E'][sid]['output'],'GPT_G':ann['G'][sid]['output'],'E_valid':ann['E'][sid]['valid'],'G_valid':ann['G'][sid]['valid']})
save(root/'prior20_annotation_comparison.json',{'source':'existing20 randomly selected expansion cases with previous Codex AI diagnosis','not_human_gold':True,'cases':cases})
known=json.load(open(base/'runs/expansion_500_v2/confirmed_issue_registry.json'));save(root/'known_issue_responses.json',[{**r,'GPT_E':ann['E'][r['source_id']]['output'],'GPT_G':ann['G'][r['source_id']]['output']} for r in known])
arithmetic=json.load(open(base/'runs/expansion_500_v2/arithmetic_flag_review.json'));confirmed={r['source_id'] for r in arithmetic if r['review_category']=='explicit_arithmetic_mismatch'}
comparison=[{'source_id':sid,'verified_equations':[r for r in arithmetic if r['source_id']==sid],'GPT_E':ann['E'][sid]['output'],'GPT_G':ann['G'][sid]['output']} for sid in sorted(confirmed)]
save(root/'verified_arithmetic_responses.json',{'n_sources':len(confirmed),'conditions':{c:{'supported_error':sum(ann[c][sid]['valid'] and ann[c][sid]['output']['status']=='supported_error' for sid in confirmed),'total':len(confirmed)} for c in 'EG'},'cases':comparison,'note':'Known numeric-equation errors, selected by a limited parser and prior original-line review. Not a random sample or comprehensive accuracy measure.'})
screenrows=jl(root/'screen_outputs.jsonl');assert len(screenrows)==500
screenby={r['source_id']:r for r in screenrows};prior=json.load(open(base/'runs/expansion_500_v2/screen_audit20.json'));aud=[]
for a in prior['rows']:
 sid=a['source_id'];v=screenby[sid]['output'];aud.append({'source_id':sid,'prior_AI_expected_status':a['expected_AI_audit_status'],'GPT_status':v['status'],'exact_status_agreement':v['status']==a['expected_AI_audit_status'],'GPT_output':v,'previous_Qwen_status':a['actual_status']})
samples={r['id']:r for r in jl(root/'sample.jsonl')};quote_errors=[]
for r in screenrows:
 q=r['output']['evidence_quote']
 if q is not None and (not q or q not in samples[r['source_id']]['response']):quote_errors.append(r['source_id'])
screen_summary={'attempted':500,'complete':len(screenrows),'statuses':dict(collections.Counter(r['output']['status'] for r in screenrows)),'known19_arithmetic_error_cases':{'GPT_math_error':sum(screenby[sid]['output']['status']=='math_error' for sid in confirmed),'total':len(confirmed),'previous_Qwen_math_error':3},'prior20_AI_status_agreement':sum(r['exact_status_agreement'] for r in aud),'prior20_n':20,'evidence_quote_not_exact_ids':quote_errors,'note':'Prior20 are AI reference diagnoses, not human gold. Known19 arithmetic cases are selected, not random. Same GPT model, separate fresh contexts, no reference answers in these requests.'}
save(root/'screen_summary.json',screen_summary);save(root/'screen_audit20.json',aud);save(root/'screen_known_issues.json',[{**r,'GPT_screen':screenby[r['source_id']]['output']} for r in known]);print('SCREEN',json.dumps(screen_summary,ensure_ascii=False),flush=True)
# Preserve total provider-reported usage and requested settings, not a fabricated dollar estimate.
usage=collections.Counter();calls=0
for p in (root/'calls').glob('*/metadata.json'):
 m=json.load(open(p));calls+=1;usage.update({k:v for k,v in m.get('usage',{}).items() if isinstance(v,(int,float))})
save(root/'usage_summary.json',{'recorded_calls':calls,'provider_reported_usage':dict(usage),'billing_route':'ChatGPT-authenticated Codex; no API-key billing estimate','model_alias':'gpt-6-astra','snapshot_unknown':True})
print('EVALUATION COMPLETE',flush=True)

real_expected={r['blind_id']:r for r in jl(root/'real_audit_expected.jsonl')};real_rows=[]
for b in jl(root/'real_audit_outputs.jsonl'):
 for r in b['output']['results']:real_rows.append({**real_expected[r['pair_id']],'GPT_output':r,'agrees':r['judgment']==real_expected[r['pair_id']]['expected']})
save(root/'real_audit_results.json',{'n':len(real_rows),'label_agreement':sum(r['agrees'] for r in real_rows),'rows':real_rows,'not_human_gold':True,'not_random_sample':True})
