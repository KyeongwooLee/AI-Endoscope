import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
out=Path(__file__).resolve().parents[1];old=out.parent/'expansion_500_v1'
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
oldz=np.load(old/'embeddings/G.npz');newz=np.load(out/'embeddings/G.npz');oldids=oldz['ids'].tolist();newids=newz['ids'].tolist();index={s:i for i,s in enumerate(newids)};aligned=newz['vectors'][[index[s] for s in oldids]];diff=float(np.max(np.abs(aligned-oldz['vectors'])));assert np.allclose(aligned,oldz['vectors'],atol=1e-5)
oldmet=json.load(open(old/'cluster_metrics.json'));newmet=json.load(open(out/'cluster_metrics.json'))
def main(m):return [r for r in m if r['algorithm']=='kmeans' and r['k']==10 and r['seed']==42]
fixed=set(json.load(open(old/'common_cohort.json'))['ids']);fixedresults=[]
for c in 'REGQ':
 a=np.load(old/f'embeddings/{c}.npz');b=np.load(out/f'embeddings/{c}.npz');ia=a['ids'].tolist();ib=b['ids'].tolist();ss=[s for s in ia if s in fixed];x=a['vectors'][[ia.index(s) for s in ss]];y=b['vectors'][[ib.index(s) for s in ss]];assert np.allclose(x,y,atol=1e-5)
 l1=KMeans(n_clusters=10,random_state=42,n_init=10).fit_predict(x);l2=KMeans(n_clusters=10,random_state=42,n_init=10).fit_predict(y)
 fixedresults.append({'method':c,'n':len(ss),'old_silhouette':float(silhouette_score(x,l1,metric='cosine')),'new_silhouette':float(silhouette_score(y,l2,metric='cosine')),'old_vs_new_ari':float(adjusted_rand_score(l1,l2))})
oldret={r['query_id']:r for r in jl(old/'retrieval.jsonl') if r['cohort']=='native' and r['method']=='G' and r['top5']};newret={r['query_id']:r for r in jl(out/'retrieval.jsonl') if r['cohort']=='native' and r['method']=='G' and r['top5']}
changes=[];added=set(newids)-set(oldids)
for q,r in oldret.items():
 a=[v['id'] for v in r['top5']];b=[v['id'] for v in newret[q]['top5']];changes.append({'query_id':q,'old_top5':a,'new_top5':b,'shared_n':len(set(a)&set(b)),'newly_repaired_in_top5':len(set(b)&added),'identical_order':a==b})
res={'offset_repaired_n':len(added),'old_G_n':len(oldids),'new_G_n':len(newids),'max_old_embedding_coordinate_change':diff,'same_cases_same_vectors':True,'baseline_metrics_k10':main(oldmet),'new_metrics_k10':main(newmet),'fixed_baseline_common250':fixedresults,'old_common':json.load(open(old/'common_cohort.json')),'new_common':json.load(open(out/'common_cohort.json')),'retrieval':{'old_G_queries':len(oldret),'new_G_queries':len(newret),'old_G_gallery':next(iter(oldret.values()))['gallery_n'],'new_G_gallery':next(iter(newret.values()))['gallery_n'],'old_queries_changed_top5':sum(not r['identical_order'] for r in changes),'repaired_candidates_in_old_query_top5':sum(r['newly_repaired_in_top5'] for r in changes),'compared_slots':len(changes)*5,'details':changes},'interpretation':'Changed eligible membership, not changed descriptions. Native/common cohorts differ after repair; fixed old250 comparison isolates unchanged representations. No new accuracy claim.'}
(out/'before_after.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in res.items() if k not in ['old_common','new_common','retrieval']},ensure_ascii=False))
print({k:v for k,v in res['retrieval'].items() if k!='details'})
