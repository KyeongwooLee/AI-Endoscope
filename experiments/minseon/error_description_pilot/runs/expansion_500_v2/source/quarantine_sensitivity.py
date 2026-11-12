import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import json,numpy as np,csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
out=Path(__file__).resolve().parents[1]
entries=[
('train-1081-4','reference_error','원래 풀이1060은 맞다. 참조가500/1000을1000/500으로 뒤집고, 둘째 벌 수800을700으로 바꿨다.','1000*(1-.2)=800; 500/1000=.5; 800*.5*1.4+500=1060'),
('train-3995-2','reference_error','타일 넓이가6제곱인치라고 주어졌다. 원래 풀이576이 문구에 맞고, 참조는6을 한 변 길이로 취급했다.','48*72/6=576'),
('train-3795-3','answer_marker_conflict','본문의 제한속도50은 맞지만 서두####40과 충돌한다. 본문의 계산 오류로 분류하지 않는다.','60/1-10=50'),
('train-7430-4','answer_marker_conflict','본문의 남은 좌석120은 맞지만 서두####20과 충돌한다.','50*4-80=120'),
('train-7421-3','answer_marker_conflict','본문의 상금160은 맞지만 서두####200과 충돌한다.','(2400-2400/3)/10=160'),
('train-1297-4','answer_marker_conflict','본문의 학생8명은 맞지만 서두####16과 충돌한다.','40*.8*.25=8'),
('train-6861-3','question_ambiguity','둘이 게임 하나를 공동 구매하면12개, 각각 하나씩 사면24개다. 참조는 후자를 가정한다. 문구만으로 전자의 풀이를 확실한 오류로 단정하기 어렵다.','60/5=12; 2*60/5=24')]
reg=[{'source_id':sid,'issue':issue,'reason_ko':reason,'arithmetic':math,'reviewer':'Codex direct original-text audit, not human ground truth','use':'quarantine from mathematical-error-only sensitivity analysis; keep originals and answer-marker cases separately'} for sid,issue,reason,math in entries];(out/'confirmed_issue_registry.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2));bad={r['source_id'] for r in reg};(out/'sensitivity').mkdir(exist_ok=True)
common=set(json.load(open(out/'common_cohort.json'))['ids'])-bad;metrics=[];assign=[];details=[];counts={}
for c in 'REGQ':
 z=np.load(out/f'embeddings/{c}.npz');ids=z['ids'].tolist();x=z['vectors'];counts[c]=sum(i not in bad for i in ids)
 for cohort in ['native','common']:
  ix=[j for j,s in enumerate(ids) if s not in bad and (cohort=='native' or s in common)];xx=x[ix];ss=[ids[j] for j in ix];labels=[]
  for seed in [42,43,44]:
   lab=KMeans(n_clusters=10,random_state=seed,n_init=10).fit_predict(xx);labels.append(lab);metrics.append({'cohort':cohort,'method':c,'n':len(ss),'k':10,'seed':seed,'silhouette':float(silhouette_score(xx,lab,metric='cosine')),'sizes':np.bincount(lab).tolist()})
   if seed==42:
    assign.extend({'cohort':cohort,'method':c,'source_id':sid,'cluster':int(label)} for sid,label in zip(ss,lab))
    if c=='G' and cohort=='native':
     for j in range(10):
      a=np.where(lab==j)[0];cen=xx[a].mean(axis=0);cen/=np.linalg.norm(cen);order=a[np.argsort(-(xx[a]@cen))];details.append({'cluster':j,'n':len(a),'center_ids':[ss[j] for j in order[:6]],'members':[ss[j] for j in a]})
  metrics.append({'cohort':cohort,'method':c,'algorithm':'stability','ari':[float(adjusted_rand_score(labels[a],labels[b])) for a,b in [(0,1),(0,2),(1,2)]]})
(out/'sensitivity/metrics.json').write_text(json.dumps({'excluded_ids':sorted(bad),'counts':counts,'common_n':len(common),'metrics':metrics,'note':'Targeted7 known issues only; remaining cases NOT verified clean. Post-hoc sensitivity; do not claim causally improved quality from silhouette changes.'},indent=2));(out/'sensitivity/G_clusters.json').write_text(json.dumps(details,indent=2))
with (out/'sensitivity/cluster_assignments.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(assign[0]));w.writeheader();w.writerows(assign)
print('sensitivity',counts,'common',len(common));print([r for r in metrics if r.get('seed')==42 and r['cohort']=='common'])
