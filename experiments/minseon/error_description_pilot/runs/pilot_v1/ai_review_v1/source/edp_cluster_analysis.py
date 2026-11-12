from pathlib import Path
import json,csv,sys,hashlib
import numpy as np
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
p=Path('/home/coder/yaicon-errorprobe/AI-Endoscope/experiments/minseon/error_description_pilot/runs/pilot_v1');o=p/'ai_review_v1';o.mkdir(exist_ok=True)
items=list(csv.DictReader((p/'review_package/templates/reviewer_1/items_stage1.csv').open()))
bytext={(r['question'],r['model_solution']):i for i,r in enumerate(items)}
inputs={c:[json.loads(s) for s in (p/f'embeddings/eval/{c}_inputs.jsonl').read_text().split('\n') if s] for c in 'REGQ'}
# Map source IDs from frozen review key only after independent original/description review.
key=[json.loads(s) for s in (p/'review_key_private/item_label_key.jsonl').read_text().split('\n') if s]
idx={r['item_code']:i for i,r in enumerate(items)}
sididx={r['source_id']:idx[r['item_code']] for r in key}
metrics=[];assign=[];details={};models={}
for c in 'REGQ':
 d=np.load(p/f'embeddings/eval/{c}.npz');X=d['vectors'];ids=d['ids'].tolist();tx={r['source_id']:r['text'] for r in inputs[c]}
 assert X.shape==(len(ids),1024) and np.allclose(np.linalg.norm(X,axis=1),1,atol=1e-3)
 for k in (4,6,8,10):
  runs=[]
  for seed in (42,43,44):
   model=KMeans(n_clusters=k,random_state=seed,n_init=10).fit(X);lab=model.labels_;runs.append(lab)
   metrics.append(dict(method=c,algorithm='kmeans',k=k,seed=seed,n=len(ids),silhouette_cosine=float(silhouette_score(X,lab,metric='cosine')),sizes=np.bincount(lab).tolist()))
   if k==6 and seed==42:
    models[c]=(X,ids,lab)
    clusters=[]
    for j in range(k):
     ix=np.where(lab==j)[0];cen=X[ix].mean(axis=0);cen/=np.linalg.norm(cen);order=ix[np.argsort(-(X[ix]@cen))]
     clusters.append({'cluster':j,'n':len(ix),'members':[sididx[ids[a]] for a in ix], 'center_examples':[{'index':sididx[ids[a]],'source_id':ids[a],'text':tx[ids[a]],'center_cosine':float(X[a]@cen)} for a in order[:4]]})
     for a in ix:assign.append({'method':c,'source_id':ids[a],'item_index':sididx[ids[a]],'item_code':items[sididx[ids[a]]]['item_code'],'cluster':j})
    details[c]=clusters
  metrics.append(dict(method=c,algorithm='kmeans_stability',k=k,ari_pairs=[float(adjusted_rand_score(runs[a],runs[b])) for a,b in [(0,1),(0,2),(1,2)]]))
  lab=AgglomerativeClustering(n_clusters=k,metric='cosine',linkage='average').fit_predict(X)
  metrics.append(dict(method=c,algorithm='average_link_cosine',k=k,n=len(ids),silhouette_cosine=float(silhouette_score(X,lab,metric='cosine')),sizes=np.bincount(lab).tolist()))
(o/'cluster_metrics.json').write_text(json.dumps(metrics,indent=2));(o/'cluster_details.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
with (o/'cluster_assignments.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(assign[0]));w.writeheader();w.writerows(assign)
(o/'cluster_protocol.json').write_text(json.dumps({'status':'posthoc_exploratory_AI_review','primary_display':'KMeans k=6, seed=42, n_init=10; chosen for readability before inspecting computed cluster results','sensitivity_k':[4,6,8,10],'seeds':[42,43,44],'secondary_algorithm':'average-link cosine agglomerative','input':'frozen eval BGE-M3 L2-normalized dense vectors, no reembedding, no label input, no PCA before clustering','cohorts':'R/Q n=100; E/G n=94; silhouette across unequal cohorts is descriptive, not a fair performance ranking','not_new_holdout':True},indent=2))
print(json.dumps(details,ensure_ascii=False))
print('METRICS',json.dumps(metrics))
