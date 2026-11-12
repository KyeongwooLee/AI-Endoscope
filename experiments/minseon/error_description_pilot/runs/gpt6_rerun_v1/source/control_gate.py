from pathlib import Path
import json,collections
root=Path(__file__).resolve().parents[1]
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
single=jl(root/'control_outputs.jsonl');batch=jl(root/'controls_batch_outputs.jsonl');assert len(single)==36 and len(batch)==8
mapping={r['blind_id']:r for r in jl(root/'controls_batch_mapping.jsonl')};flat=[]
for b in batch:
 for r in b['output']['results']:flat.append({**mapping[r['pair_id']],'output':r})
assert len(flat)==36
summ={}
for mode,rows in [('single_pair',single),('five_pair_batches',flat)]:
 byclass={c:{'correct':sum(r['expected']==c and r['output']['judgment']==c for r in rows),'n':sum(r['expected']==c for r in rows)} for c in ['same','different','uncertain']};groups=collections.defaultdict(list)
 for r in rows:groups[r['base_pair']].append(r['output']['judgment'])
 consistent=sum(len(v)==2 and v[0]==v[1] for v in groups.values());passed=all(v['correct']>=10 for v in byclass.values()) and consistent>=16
 summ[mode]={'by_expected':byclass,'correct':sum(v['correct'] for v in byclass.values()),'n':len(rows),'order_consistent_pairs':consistent,'underlying_pairs':18,'passed_preregistered_gate':passed}
res={'passed':summ['five_pair_batches']['passed_preregistered_gate'],'methods':summ,'limitations':'Synthetic controls only. Passing is a basic sanity check, not validation of natural-data accuracy. Same GPT model used for annotation and judging in separate sessions; no human ground truth.'};(root/'control_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2));(root/'control_batch_flat.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in flat));print(json.dumps(res,ensure_ascii=False))
