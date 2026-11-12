from pathlib import Path
import json,sys,collections,csv
import numpy as np
out=Path(__file__).resolve().parents[1];sys.path.insert(0,str(out.parents[1]));from edp.evaluate import paired_bootstrap
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
if any(r['judgment'] is None for r in jl(out/'ai_pair_judgments.jsonl')):
 import subprocess
 subprocess.run([sys.executable,'-u',str(out/'source/repair_judge.py')],check=True)
keys=jl(out/'pair_key.jsonl');judges=jl(out/'ai_pair_judgments.jsonl');assert len(judges)==len(keys) and len({j['pair_id'] for j in judges})==len(keys)
byid={j['pair_id']:j for j in judges};lookup={(k['query_id'],k['gallery_id']):byid[k['pair_id']] for k in keys};retr=jl(out/'retrieval.jsonl');result={}
for cohort in ['native','common']:
 stats={}
 for m in ['R','E','G','Q','Random']:
  rows=[]
  for r in [r for r in retr if r['cohort']==cohort and r['method']==m]:
   labs=[lookup[(r['query_id'],s['id'])]['judgment'] for s in r['top5']];co=collections.Counter(labs);rows.append({'query_id':r['query_id'],'returned':len(labs),'same':co['same'],'uncertain':co['uncertain'],'missing':co[None],'status':r['status'],'yield':co['same']/5 if not co[None] else None})
  missing=sum(r['missing'] for r in rows);n=len(rows);stats[m]={'query_n':n,'returned':sum(r['returned'] for r in rows),'same':sum(r['same'] for r in rows),'uncertain':sum(r['uncertain'] for r in rows),'missing':missing,'same_at_5':sum(r['same'] for r in rows)/(n*5) if not missing else None,'uncertain_as_same_bound':sum(r['same']+r['uncertain'] for r in rows)/(n*5) if not missing else None,'coverage':sum(r['status']=='ranked' for r in rows)/n,'per_query':rows}
 contr={}
 for a,b in [('G','E'),('G','R'),('E','R')]:
  ar=sorted(stats[a]['per_query'],key=lambda r:r['query_id']);br=sorted(stats[b]['per_query'],key=lambda r:r['query_id']);assert [r['query_id'] for r in ar]==[r['query_id'] for r in br]
  contr[a+'-'+b]=paired_bootstrap([r['yield'] for r in ar],[r['yield'] for r in br],2000,20261112,.95) if not stats[a]['missing'] and not stats[b]['missing'] else {'status':'missing_judgments'}
 result[cohort]={'methods':stats,'paired_query_bootstrap':contr}
result['judge']={'model':'Qwen/Qwen3-4B-Instruct-2507','revision':'cdbee75f17c01a7cc42f958dc650907174af0554','same_model_as_annotator':True,'inputs':'raw questions, model solutions, references only; fresh contexts; no generated descriptions/methods/scores','counts':dict(collections.Counter(j['judgment'] for j in judges)),'limitations':'AI provisional only. Same model family as solver and exact same model as annotator, no independent ground truth. Bootstrap excludes judge uncertainty.'}
if (out/'pair_audit_codex.json').exists():
 audit=json.load(open(out/'pair_audit_codex.json'))
 comparison=[{'pair_id':r['pair_id'],'codex_judgment':r['judgment'],'qwen_judgment':byid[r['pair_id']]['judgment'],'agreement':r['judgment']==byid[r['pair_id']]['judgment'],'codex_reason_ko':r['reason_ko'],'qwen_reason':byid[r['pair_id']]['reason']} for r in audit]
 (out/'pair_audit_comparison.json').write_text(json.dumps({'status':'two_AI_judges_not_human_ground_truth','sample_n':len(comparison),'agreement_n':sum(r['agreement'] for r in comparison),'rows':comparison},ensure_ascii=False,indent=2))
(out/'ai_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
with (out/'ai_pair_reviews.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['pair_id','query_id','gallery_id','judgment','reason']);w.writeheader()
 for k in keys:w.writerow({**{z:k[z] for z in ['pair_id','query_id','gallery_id']},**{z:byid[k['pair_id']][z] for z in ['judgment','reason']}})
for c in ['native','common']:
 print(c,{m:{k:v for k,v in s.items() if k!='per_query'} for m,s in result[c]['methods'].items()});print(result[c]['paired_query_bootstrap'])
