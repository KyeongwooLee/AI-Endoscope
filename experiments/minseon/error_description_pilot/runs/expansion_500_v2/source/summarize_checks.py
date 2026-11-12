from pathlib import Path
import json,collections,re,csv
out=Path(__file__).resolve().parents[1];old=out.parent/'expansion_500_v1'
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
def save(n,x):(out/n).write_text(json.dumps(x,ensure_ascii=False,indent=2))
rows=jl(out/'check_outputs.jsonl');pack=jl(out/'check_packets.jsonl');assert len(rows)==len(pack)==582 and len({r['id'] for r in rows})==582
controls={r['id']:r for r in jl(out/'judge_controls.jsonl')};summary={};control_rows=[]
for method in ['original','revised']:
 rs=[r for r in rows if r['task']=='judge_control' and r['method']==method];types={};lookup={r['control_id']:r for r in rs}
 for expected in ['same','different','uncertain']:
  sub=[r for r in rs if controls[r['control_id']]['expected']==expected];types[expected]={'n_ordered':len(sub),'n_base_pairs':len({controls[r['control_id']]['base_pair'] for r in sub}),'correct':sum((r['output'] or {}).get('judgment')==expected for r in sub),'outputs':dict(collections.Counter((r['output'] or {}).get('judgment','unparsed') for r in sub))}
 swapped=[]
 for c in controls.values():
  if c['swapped']:continue
  other=c['id'][:-1]+'1';a=(lookup[c['id']]['output'] or {}).get('judgment');b=(lookup[other]['output'] or {}).get('judgment');swapped.append({'base_pair':c['base_pair'],'first':a,'swapped':b,'consistent':a is not None and a==b})
 summary[method]={'by_expected':types,'correct':sum(v['correct'] for v in types.values()),'total':len(rs),'order_consistency_n':sum(r['consistent'] for r in swapped),'base_pairs':len(swapped),'order_details':swapped,'missing_parse':sum(r['output'] is None for r in rs)}
 for r in rs:
  c=controls[r['control_id']];v=r['output'] or {};control_rows.append({'control_id':c['id'],'method':method,'expected':c['expected'],'actual':v.get('judgment'),'reason':v.get('reason'),'case1_error':v.get('case1_error'),'case2_error':v.get('case2_error')})
save('judge_control_results.json',{'methods':summary,'rows':control_rows,'interpretation':'Constructed sanity checks,18 base pairs each in two orders, not36 independent natural cases. Both prompts fail basic same-error and uncertainty checks. Do not use as a reliable retrieval evaluator.'})
oldreal={r['pair_id']:r for r in jl(old/'ai_pair_judgments.jsonl')};expected={r['pair_id']:r for r in json.load(open(out/'pair_audit_codex.json'))};real=[]
for r in rows:
 if r['task']!='real_pair_check':continue
 pid=r['id'].removesuffix('-revised');e=expected[pid];real.append({'pair_id':pid,'previous_Codex_AI_label':e['judgment'],'previous_Codex_AI_reason':e.get('reason_ko'),'original_Qwen':oldreal[pid]['judgment'],'revised_Qwen':r['output'],'label_agrees':(r['output'] or {}).get('judgment')==e['judgment']})
save('real_pair_check_results.json',{'rows':real,'label_agreement':sum(r['label_agrees'] for r in real),'n':len(real),'note':'Existing10 AI-reviewed real pairs, dominated by different labels. Agreement is not correctness of reasons, nor independent human accuracy.'})
samples={r['id']:r for r in jl(out/'sample.jsonl')};screens={r['source_id']:r for r in rows if r['task']=='reference_free_screen'};audits=json.load(open(out/'annotation_audit_original_diagnoses.json'));marker={'train-7430-4','train-7421-3','train-1297-4'};audit=[]
for a in audits:
 sid=a['source_id'];exp={'yes':'math_error','no':'no_math_error','uncertain':'ambiguous'}[a['observable_math_error']]
 if sid in marker:exp='answer_conflict'
 r=screens[sid];v=r['output'] or {};act=v.get('status');audit.append({'source_id':sid,'expected_AI_audit_status':exp,'actual_status':act,'exact_status_agreement':act==exp,'math_error_presence_agreement':None if exp=='ambiguous' else act in ['math_error','no_math_error','answer_conflict'] and (act=='math_error')==(exp=='math_error'),'prior_diagnosis_ko':a['diagnosis_ko'],'new_screen':v})
save('screen_audit20.json',{'n':20,'exact_status_agreement_n':sum(r['exact_status_agreement'] for r in audit),'math_error_presence_agreement_n':sum(r['math_error_presence_agreement'] is True for r in audit),'binary_comparable_n':sum(r['math_error_presence_agreement'] is not None for r in audit),'rows':audit,'note':'Same prior20 randomly selected cases; Codex AI diagnoses, no independent human gold; label agreement alone does not validate description content.'})
def number(text):
 if not isinstance(text,str):return None
 nums=re.findall(r'(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?',text)
 return float(nums[0].replace(',','')) if len(nums)==1 else None
triage=[];ann={c:{r['source_id']:r for r in jl(out/f'annotation/{c}.jsonl')} for c in 'EG'}
known={r['source_id']:r for r in json.load(open(out/'confirmed_issue_registry.json'))}
arithmetic_review=json.load(open(out/'arithmetic_flag_review.json'));arithmetic_ids={r['source_id'] for r in arithmetic_review if r['review_category']=='explicit_arithmetic_mismatch'}
for sid,r in screens.items():
 s=samples[sid];v=r['output'] or {};refmarks=re.findall(r'####\s*([-\d.,]+)',s['reference']);refnum=number(refmarks[-1]) if refmarks else None;ans=number(v.get('independent_answer'));diff=ans is not None and refnum is not None and abs(ans-refnum)>1e-8
 quote=v.get('evidence_quote');quoteok=quote is None or (isinstance(quote,str) and quote in s['response']);flags=[]
 if r['output'] is None:flags.append('screen_output_unparsed')
 if diff:flags.append('AI_independent_answer_differs_from_reference')
 if v.get('status') in ['answer_conflict','ambiguous','no_math_error']:flags.append('AI_did_not_confirm_math_error')
 if not quoteok:flags.append('AI_quote_not_exact')
 if sid in known:flags.append('Codex_original_text_issue_confirmed')
 if sid in arithmetic_ids:flags.append('explicit_arithmetic_mismatch_verified_by_code_and_original_line')
 supported={c:ann[c][sid]['valid'] and (ann[c][sid]['output'] or {}).get('status')=='supported_error' for c in 'EG'}
 if v.get('status') in ['answer_conflict','ambiguous','no_math_error'] and any(supported.values()):flags.append('AI_screen_disagrees_with_embedded_annotation')
 triage.append({'source_id':sid,'screen_status':v.get('status'),'independent_answer':v.get('independent_answer'),'reference_final_answer':refmarks[-1] if refmarks else None,'numeric_reference_disagreement':diff,'quote_exact':quoteok,'verified_arithmetic_mismatch':sid in arithmetic_ids,'flags':flags,'confirmed_issue':known.get(sid),'screen_output':v,'embedded_E':supported['E'],'embedded_G':supported['G'],'question':s['question'],'model_solution':s['response'],'reference_solution':s['reference']})
save('reference_triage.json',triage)
save('screen_summary.json',{'confirmed_arithmetic_error_sources':len(arithmetic_ids),'screen_missed_confirmed_arithmetic_errors':sum((screens[sid]['output'] or {}).get('status')!='math_error' for sid in arithmetic_ids),'attempted':500,'parsed':sum(r['output'] is not None for r in screens.values()),'statuses':dict(collections.Counter((r['output'] or {}).get('status','unparsed') for r in screens.values())),'numeric_reference_disagreement_n':sum(r['numeric_reference_disagreement'] for r in triage),'not_exact_evidence_quote_n':sum(not r['quote_exact'] for r in triage),'any_flag_n':sum(bool(r['flags']) for r in triage),'disagrees_with_annotation_n':sum('AI_screen_disagrees_with_embedded_annotation' in r['flags'] for r in triage),'note':'Flags are review candidates only, not confirmed bad references or corrected annotation labels. No screening output used in primary embeddings.'})
with (out/'reference_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['source_id','screen_status','independent_answer','reference_final_answer','flags','question','model_solution','reference_solution','AI_check'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for r in triage:
  if not r['flags']:continue
  v={k:r.get(k) for k in fields};v['flags']='; '.join(r['flags']);v['AI_check']=r['screen_output'].get('check');w.writerow({k:"'"+x if isinstance(x,str) and x.startswith(('=','+','-','@')) else x for k,x in v.items()})
save('evaluation_status.json',{'automatic_retrieval_accuracy_published':False,'reason':'Both original and revised judge fail frozen elementary positive/uncertainty controls; real diagnosis reasons contain substantive errors. No reliable automatic same-error accuracy estimate.','retrieval_recomputed':True,'full_new_pair_judging_skipped_after_failed_sanity_checks':True,'screening_use':'triage only; not automatic relabeling','human_ground_truth':False})
print(json.dumps({'controls':{m:{k:v for k,v in s.items() if k!='order_details'} for m,s in summary.items()},'screen':json.load(open(out/'screen_summary.json')),'audit20_agreement':sum(r['exact_status_agreement'] for r in audit)},ensure_ascii=False))
