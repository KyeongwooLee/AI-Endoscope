import os
os.environ.update(USE_TF='0',USE_FLAX='0',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import json,sys,hashlib,csv,collections
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score
root=Path(__file__).resolve().parents[1];base=root.parents[1];sys.path.insert(0,str(base));from edp.embedding import Embedder,check_vectors
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l]
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def lines(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in x))
dataset=sys.argv[1];out=root/'datasets'/dataset;sample=jl(out/'sample.jsonl');byid={r['id']:r for r in sample};rows=[r for r in jl(root/'annotation/all.jsonl') if r['dataset']==dataset];assert len(rows)==2*len(sample),(dataset,len(rows));ann={c:{r['source_id']:r for r in rows if r['condition']==c} for c in 'EG'};cfg=json.load(open(root/'config.json'))
for c in 'EG':lines(out/f'annotation/{c}.jsonl',ann[c].values())
summary={}
for c in 'EG':
 a=list(ann[c].values());summary[c]={'total':len(a),'statuses':dict(collections.Counter(r['output']['status'] if r['valid'] else 'invalid' for r in a)),'valid':sum(r['valid'] for r in a),'reference_review_needed':sum(r['valid'] and r['output'].get('reference_review_needed',False) for r in a),'required_second_attempt':sum(len(r['attempts'])>1 for r in a),'raw_last_attempt_errors':dict(collections.Counter(e for r in a for e in r['attempts'][-1]['raw_errors'])),'eval_statuses':dict(collections.Counter(r['output']['status'] if r['valid'] else 'invalid' for r in a if byid[r['source_id']]['role']!='development'))}
save(out/'annotation_summary.json',summary)
local=json.loads(json.dumps(cfg));local['embedding'].update(model='/home/coder/yaicon-errorprobe/models/BAAI-bge-m3',batch_size=4);embed=Embedder(local);embed.provenance.update(model='BAAI/bge-m3',local_artifact_path=local['embedding']['model'],weights_sha256='b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38');save(out/'embedding_provenance.json',embed.provenance)
old=base/'runs'/('pilot_v1' if dataset=='pilot130' else 'expansion_500_v2');vec={};texts={}
for c in 'REGQ':
 inp=[]
 for s in sample:
  t=f'Question: {s["question"]}\nSolution: {s["response"]}' if c=='R' else s['question'] if c=='Q' else None
  if c in 'EG':
   a=ann[c][s['id']]
   if a['valid'] and a['output']['status']=='supported_error':t=a['output']['explanation' if c=='E' else 'error_description']
  if t is not None:
   n=embed.token_length(t);assert n<=8192;inp.append({'source_id':s['id'],'text':t,'text_sha256':hashlib.sha256(t.encode()).hexdigest(),'token_count':n})
 lines(out/f'embeddings/{c}_inputs.jsonl',inp);ids=[r['source_id'] for r in inp];fn=out/f'embeddings/{c}.npz'
 if fn.exists():z=np.load(fn);assert z['ids'].tolist()==ids;x=check_vectors(z['vectors'],len(ids))
 elif c in 'RQ':
  paths=[old/f'embeddings/{phase}/{c}.npz' for phase in ['dev','eval']] if dataset=='pilot130' else [old/f'embeddings/{c}.npz'];prior={}
  for p in paths:
   z=np.load(p);prior.update({sid:v for sid,v in zip(z['ids'].tolist(),z['vectors'])})
  x=check_vectors(np.array([prior[sid] for sid in ids]),len(ids));np.savez(fn,ids=np.array(ids),vectors=x)
 else:x=embed.encode([r['text'] for r in inp]);np.savez(fn,ids=np.array(ids),vectors=x)
 vec[c]=dict(zip(ids,x));texts[c]={r['source_id']:r['text'] for r in inp};print('EMBED',dataset,c,len(ids),flush=True)
del embed
import torch,gc
gc.collect();torch.cuda.empty_cache()
for c in 'EG':
 p=old/f'embeddings/eval/{c}.npz' if dataset=='pilot130' else old/f'embeddings/{c}.npz';z=np.load(p);vec['old_'+c]=dict(zip(z['ids'].tolist(),z['vectors']));ip=old/f'embeddings/eval/{c}_inputs.jsonl' if dataset=='pilot130' else old/f'embeddings/{c}_inputs.jsonl';texts['old_'+c]={r['source_id']:r['text'] for r in jl(ip) if r.get('text') is not None}
eval_ids=[s['id'] for s in sample if s['role']!='development'];ev=set(eval_ids);common=ev&set(vec['E'])&set(vec['G']);historical=common&set(vec['old_E'])&set(vec['old_G']);cohorts={'native':ev,'common':common,'historical_common':historical}
if dataset=='pilot130':cohorts['development']={s['id'] for s in sample if s['role']=='development'}
if dataset=='expansion500':
 known={r['source_id'] for r in json.load(open(base/'runs/expansion_500_v2/confirmed_issue_registry.json'))};cohorts['known_issue_excluded_common']=common-known
save(out/'cohorts.json',{k:{'n':len(v),'ids':sorted(v),'queries':sum(byid[s]['role']=='query' for s in v),'gallery':sum(byid[s]['role']=='gallery' for s in v)} for k,v in cohorts.items()})
metrics=[];assign=[];detail={}
for cohort,eligible in cohorts.items():
 methods=['R','E','G','Q','old_E','old_G'] if cohort=='historical_common' else list('REGQ')
 for c in methods:
  ids=[s['id'] for s in sample if s['id'] in eligible and s['id'] in vec[c]];x=np.array([vec[c][s] for s in ids]);groups=[]
  if len(ids)<5:continue
  for k in [4,6,8,10,15,20]:
   if k>=len(ids):continue
   labs=[]
   for seed in [42,43,44]:
    lab=KMeans(n_clusters=k,random_state=seed,n_init=10).fit_predict(x);labs.append(lab);metrics.append({'cohort':cohort,'method':c,'algorithm':'kmeans','k':k,'seed':seed,'n':len(ids),'silhouette':float(silhouette_score(x,lab,metric='cosine')),'sizes':np.bincount(lab).tolist()})
    if k==10 and seed==42:
     for j in range(k):
      ix=np.where(lab==j)[0];cen=x[ix].mean(axis=0);cen/=np.linalg.norm(cen);order=ix[np.argsort(-(x[ix]@cen))];groups.append({'cluster':j,'n':len(ix),'members':[ids[i] for i in ix],'center_examples':[{'source_id':ids[i],'text':texts[c][ids[i]],'center_similarity':float(x[i]@cen)} for i in order[:6]]});assign.extend({'cohort':cohort,'method':c,'source_id':ids[i],'cluster':j} for i in ix)
   metrics.append({'cohort':cohort,'method':c,'algorithm':'stability','k':k,'ari':[float(adjusted_rand_score(labs[a],labs[b])) for a,b in [(0,1),(0,2),(1,2)]]})
   lab=AgglomerativeClustering(n_clusters=k,metric='cosine',linkage='average').fit_predict(x);metrics.append({'cohort':cohort,'method':c,'algorithm':'average_link_cosine','k':k,'n':len(ids),'silhouette':float(silhouette_score(x,lab,metric='cosine')),'sizes':np.bincount(lab).tolist()})
  detail[f'{cohort}:{c}']=groups;print('CLUSTER',dataset,cohort,c,len(ids),flush=True)
save(out/'cluster_metrics.json',metrics);save(out/'cluster_details.json',detail)
with (out/'cluster_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=['cohort','method','source_id','cluster']);w.writeheader();w.writerows(assign)
# Reuse old native R/Q/Random top5 exactly, preventing accidental baseline changes.
if dataset=='pilot130':oldret={(r['method'],r['query_id']):r['top_k'] for r in jl(old/'retrieval/top5.jsonl') if r['method'] in ['R','Q','Random']}
else:oldret={(r['method'],r['query_id']):r['top5'] for r in jl(old/'retrieval.jsonl') if r['cohort']=='native' and r['method'] in ['R','Q','Random']}
queries=[r['id'] for r in sample if r['role']=='query'];gallery=[r['id'] for r in sample if r['role']=='gallery'];retr=[];pairs={}
for cohort in ['native','common','historical_common']:
 eligible=cohorts[cohort];qs=[q for q in queries if q in eligible];methods=['R','E','G','Q','Random']+(['old_E','old_G'] if cohort=='historical_common' else [])
 for c in methods:
  gs=[g for g in gallery if g in eligible and (c=='Random' or g in vec[c])];rng=np.random.default_rng(20261112 if dataset=='expansion500' else 20261108)
  for q in qs:
   if cohort=='native' and c in ['R','Q','Random']:top=oldret[(c,q)];status='ranked'
   elif c=='Random':top=[{'id':g,'rank':i+1,'score':None} for i,g in enumerate(rng.choice(gs,min(5,len(gs)),replace=False))];status='ranked'
   elif q not in vec[c]:top=[];status='abstained_or_invalid'
   else:
    scores=np.array([vec[c][g] for g in gs])@vec[c][q];order=sorted(range(len(gs)),key=lambda i:(-float(scores[i]),gs[i]))[:5];top=[{'id':gs[i],'rank':j+1,'score':float(scores[i])} for j,i in enumerate(order)];status='ranked'
   retr.append({'dataset':dataset,'cohort':cohort,'method':c,'query_id':q,'gallery_n':len(gs),'status':status,'top5':top})
   for t in top:pairs.setdefault((q,t['id']),[]).append({'cohort':cohort,'method':c,'rank':t['rank']})
lines(out/'retrieval.jsonl',retr);key=[]
for i,((q,g),methods) in enumerate(sorted(pairs.items())):key.append({'pair_id':f'{dataset}-pair-{i:04d}','dataset':dataset,'query_id':q,'gallery_id':g,'returned_by':methods})
lines(out/'pair_key.jsonl',key);save(out/'analysis_status.json',{'annotation_complete':True,'embedding_complete':True,'clustering_complete':True,'retrieval_complete':True,'unique_pairs':len(key),'judging_complete':False});print('ANALYSIS COMPLETE',dataset,len(key),flush=True)
