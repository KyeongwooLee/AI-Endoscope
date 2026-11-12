from pathlib import Path
import json,random
root=Path(__file__).resolve().parents[1];base=root.parents[1]
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
samples={r['id']:r for r in jl(root/'sample.jsonl')};oldkeys={r['pair_id']:r for r in jl(base/'runs/expansion_500_v1/pair_key.jsonl')};prior=json.load(open(base/'runs/expansion_500_v1/pair_audit_codex.json'));pairs=[]
for r in prior:
 k=oldkeys[r['pair_id']];pairs.append({'source_a':k['query_id'],'source_b':k['gallery_id'],'expected':r['judgment'],'basis':r['reason_ko'],'kind':'previous10_AI_audited_real_pairs'})
for a,b in [('4085-1','7093-4'),('2348-2','3340-1'),('7025-0','7003-3'),('1640-4','4135-1'),('5304-0','4731-0'),('3248-0','525-2')]:pairs.append({'source_a':'train-'+a,'source_b':'train-'+b,'expected':'same','basis':'Both explicitly evaluate a correct numeric expression incorrectly; original numeric equations independently checked in prior arithmetic audit.','kind':'real_numeric_equation_errors'})
for a,b in [('1081-4','4085-1'),('3995-2','7093-4'),('3795-3','4085-1'),('7430-4','7093-4')]:pairs.append({'source_a':'train-'+a,'source_b':'train-'+b,'expected':'uncertain','basis':'First solution has no mathematical error under question wording, or only conflicting answer marker; second has a numeric arithmetic error.','kind':'known_reference_or_answer_marker_issue'})
assert len(pairs)==20;random.Random(20261115).shuffle(pairs);jobs=[];mapping=[];prefix=(root/'batch_judge_prompt.txt').read_text()
for start in range(0,len(pairs),5):
 payload=[];ids=[]
 for i,r in enumerate(pairs[start:start+5]):
  bid=f'A{start+i:04d}';item={'pair_id':bid}
  for name,key in [('case1','source_a'),('case2','source_b')]:
   s=samples[r[key]];item[name]={'question':s['question'],'solution':s['response'],'reference':s['reference']}
  payload.append(item);ids.append({'blind_id':bid,**r});mapping.append(ids[-1])
 jobs.append({'job_id':f'real-audit-{start//5:03d}','pairs':ids,'prompt':prefix+json.dumps(payload,ensure_ascii=False)})
for name,rows in [('real_audit_jobs.jsonl',jobs),('real_audit_expected.jsonl',mapping)]: (root/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(root/'real_audit_note.json').write_text(json.dumps({'n_pairs':20,'expected_label_origin':'previous Codex original-text audits and exact arithmetic; not human ground truth','selection':'targeted known issues, not random; fixed before GPT pair judgments','purpose':'check real-text behavior as a supplement to synthetic gate; does not estimate overall accuracy'},indent=2))
