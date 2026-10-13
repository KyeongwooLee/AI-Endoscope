#!/usr/bin/env python3
"""Training/selection and final evaluation are separate commands; shared X whitelist."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse, json, time, hashlib
from pathlib import Path
import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from threadpoolctl import threadpool_limits
from pipeline import RUN, read, dump
ACTIONS=['Keep','Recalculate','Rebuild','Review']
CS=[.01,.1,1.]
class Constant:
    def __init__(self,p): self.p=p
    def predict_proba(self,x): return np.tile([1-self.p,self.p],(x.shape[0],1))

def load(split):
    manifest=[r for r in read(RUN/'manifest.json') if r['split']==split]
    records=[read(RUN/'records'/f"{r['id']}.json") for r in manifest]
    assert len({r['question_sha256'] for r in records})==len(records)
    return records

def inputs(rows):
    # Deliberate whitelist: gold, labels, action outcomes and generated repairs absent.
    return [dict(id=r['id'],question=r['question'],response=r['original']['decoded_output'],truncated=r['original']['truncated'],mean_nll=r['original']['mean_nll'],tokens=sum(i not in (151643,151644,151645) for i in r['original']['generated_ids']),verdict=r['verification']['verdict'],verdict_nll=r['verification']['mean_nll']) for r in rows]

def availability(x):
    from common import recalculate
    return np.array([[True,recalculate(r['response'],r['truncated'])['available'],True,True] for r in x])

def labels(rows):
    y=np.array([r['label_error'] for r in rows]); z=np.array([[r['actions'][a]['success'] if r['actions'][a]['success'] is not None else -1 for a in ACTIONS] for r in rows]); assert np.array_equal(z[:,0],1-y)
    return y,z

class Features:
    def __init__(self):
        self.text=TfidfVectorizer(max_features=1500,min_df=2,ngram_range=(1,2),sublinear_tf=True)
        self.numeric=StandardScaler();self.hidden_scalers={}
    def scalars(self,x):
        return np.array([[r['mean_nll'],np.log1p(r['tokens']),r['verdict']=='CORRECT',r['verdict']=='INCORRECT',r['verdict']=='INVALID',r['verdict_nll']] for r in x],float)
    def texts(self,x): return [r['question']+'\n'+r['response'] for r in x]
    def hidden(self,x,key): return np.array([np.load(getattr(self,'hidden_dir',RUN/'hidden')/f"{r['id']}.npz")[key] for r in x],dtype=np.float64)
    def fit(self,x,keys):
        self.text.fit(self.texts(x));self.numeric.fit(self.scalars(x))
        for key in keys: self.hidden_scalers[key]=StandardScaler().fit(self.hidden(x,key))
        return self
    def transform(self,x,kind,key=None):
        n=self.numeric.transform(self.scalars(x));pieces=[sparse.csr_matrix(n[:,:2])]
        if kind!='D1': pieces.append(self.text.transform(self.texts(x)))
        if kind in ('D3','D4'): pieces.append(sparse.csr_matrix(n[:,2:]))
        if key:
            h=self.hidden_scalers[key].transform(self.hidden(x,key));pieces.append(sparse.csr_matrix(h/np.sqrt(h.shape[1])))
        z=sparse.hstack(pieces,format='csr');assert np.isfinite(z.data).all();return z

def fit_binary(x,y,c):
    if len(set(y.tolist()))<2: return Constant(float(y.mean()) if len(y) else .5)
    return LogisticRegression(C=c,solver='liblinear',max_iter=2000,random_state=20261012).fit(x,y)

def fit_selector(x,z,c):
    return [fit_binary(x[z[:,j]>=0],z[z[:,j]>=0,j],c) for j in range(4)]

def probabilities(models,x): return np.column_stack([m.predict_proba(x)[:,1] for m in models])
def choose(p,mask): return np.where(mask,p,-np.inf).argmax(1)
def outcome(z,choice): return z[np.arange(len(z)),choice]
def auc(y,p): return float(roc_auc_score(y,p)) if len(set(y.tolist()))>1 else None

def fit():
    if (RUN/'selection_frozen.json').exists(): raise RuntimeError('Already frozen. Use a new run for revised analyses.')
    started=time.perf_counter();tr=load('train');va=load('validation');x=inputs(tr);v=inputs(va);y,z=labels(tr);vy,vz=labels(va);mask=availability(v)
    keys=read(RUN/'config.json')['hidden_candidates'];features=Features().fit(x,keys)
    models={};selection={};grid=[];vp={}
    for kind in ['D1','D2','D3','D4']:
        best=None
        for key in (keys if kind=='D4' else [None]):
            tx=features.transform(x,kind,key);vx=features.transform(v,kind,key)
            for c in CS:
                m=fit_binary(tx,y,c);p=m.predict_proba(vx)[:,1];score=auc(vy,p)
                grid.append(dict(experiment='A',kind=kind,key=key,C=c,validation_auroc=score))
                if best is None or score>best[0]: best=(score,m,key,c,p)
        score,m,key,c,p=best;models[kind]=m;selection[kind]=dict(kind=kind,key=key,C=c,validation_auroc=score);vp[kind]=p.tolist()
    # B uses only its own validation outcomes; no A test results or A-selected layer.
    for name,kind in [('external','D3'),('internal','D4')]:
        best=None
        for key in (keys if kind=='D4' else [None]):
            tx=features.transform(x,kind,key);vx=features.transform(v,kind,key)
            for c in CS:
                m=fit_selector(tx,z,c);p=probabilities(m,vx);ch=choose(p,mask);acc=float(outcome(vz,ch).mean())
                loss=float(np.mean([log_loss(vz[vz[:,j]>=0,j],p[vz[:,j]>=0,j],labels=[0,1]) for j in range(4) if np.any(vz[:,j]>=0)]))
                grid.append(dict(experiment='B',kind=name,key=key,C=c,validation_accuracy=acc,validation_mean_log_loss=loss))
                if best is None or (acc,-loss)>best[0]: best=((acc,-loss),m,key,c,p)
        score,m,key,c,p=best;models[name]=m;selection[name]=dict(kind=kind,key=key,C=c,validation_accuracy=score[0],validation_mean_log_loss=-score[1]);vp[name]=p.tolist()
    fixed={a:float(outcome(vz,np.where(mask[:,j],j,0)).mean()) for j,a in enumerate(ACTIONS)}
    best_fixed=max(ACTIONS,key=lambda a:fixed[a]);best_repair=max(ACTIONS[1:],key=lambda a:fixed[a])
    selection.update(best_fixed=best_fixed,best_fixed_repair=best_repair,fixed_validation_accuracy=fixed)
    opportunity={}
    for split,rs,zz in [('train',tr,z),('validation',va,vz)]:
        zz=np.where(zz<0,zz[:,0:1],zz);best=float(zz.mean(0).max());oracle=float(zz.max(1).mean())
        opportunity[split]=dict(n=len(zz),fixed_action_accuracy=dict(zip(ACTIONS,zz.mean(0).tolist())),best_fixed_accuracy=best,oracle_accuracy=oracle,oracle_minus_best_fixed=oracle-best,repairable_errors=int(((zz[:,0]==0)&(zz[:,1:].max(1)==1)).sum()),unique_successes={a:int(((zz[:,j]==1)&(np.delete(zz,j,axis=1).max(1)==0)).sum()) for j,a in enumerate(ACTIONS)})
    joblib.dump(dict(features=features,models=models,selection=selection),RUN/'trained.joblib')
    dump(RUN/'validation_grid.json',grid);dump(RUN/'validation_predictions.json',dict(ids=[r['id'] for r in va],predictions=vp));dump(RUN/'repair_opportunity.json',opportunity)
    dump(RUN/'selection_frozen.json',dict(selection=selection,train_ids=[r['id'] for r in tr],validation_ids=[r['id'] for r in va],training_seconds=time.perf_counter()-started,trained_sha256=hashlib.sha256((RUN/'trained.joblib').read_bytes()).hexdigest(),test_not_loaded_during_fit=True))
    print(json.dumps(selection,indent=2),flush=True)

def a_metrics(y,p):
    n=max(1,int(np.ceil(.2*len(y))));idx=np.argsort(-p,kind='stable')[:n];flag=p>=.5
    return dict(AUROC=auc(y,p),AP=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),error_prevalence=float(y.mean()),error_recall_top20=float(y[idx].sum()/max(1,y.sum())),false_alarm_rate_at_half=float(flag[y==0].mean()),accuracy_at_half=float(((p>=.5)==y).mean()))

def b_metrics(rows,z,ch):
    s=outcome(z,ch);keep=z[:,0];wrong=keep==0;right=keep==1
    return dict(final_accuracy=float(s.mean()),successful_repairs=int(((s==1)&wrong).sum()),incorrect_to_correct_rate=float(s[wrong].mean()),harmful_revisions=int(((s==0)&right).sum()),correct_to_incorrect_rate=float((1-s[right]).mean()),selected_actions={a:int((ch==i).sum()) for i,a in enumerate(ACTIONS)},selected_repair_seconds=float(sum(r['actions'][ACTIONS[j]]['seconds'] for r,j in zip(rows,ch))),selected_repair_tokens=int(sum(r['actions'][ACTIONS[j]]['tokens'] for r,j in zip(rows,ch))))

def boot(y,ap,z,bp):
    rng=np.random.default_rng(20261013);values={k:[] for k in ['A_D4_minus_D3_AUROC','A_D4_minus_D3_AP','B_internal_minus_external','B_internal_minus_best_fixed','B_external_minus_best_fixed','B_internal_minus_keep']}
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y));yy=y[ix]
        if len(set(yy.tolist()))>1:
            values['A_D4_minus_D3_AUROC'].append(roc_auc_score(yy,ap['D4'][ix])-roc_auc_score(yy,ap['D3'][ix]));values['A_D4_minus_D3_AP'].append(average_precision_score(yy,ap['D4'][ix])-average_precision_score(yy,ap['D3'][ix]))
        for name,a,b in [('B_internal_minus_external','internal','external'),('B_internal_minus_best_fixed','internal','best_fixed'),('B_external_minus_best_fixed','external','best_fixed'),('B_internal_minus_keep','internal','keep')]: values[name].append(float((outcome(z,bp[a])[ix]-outcome(z,bp[b])[ix]).mean()))
    point={'A_D4_minus_D3_AUROC':auc(y,ap['D4'])-auc(y,ap['D3']),'A_D4_minus_D3_AP':average_precision_score(y,ap['D4'])-average_precision_score(y,ap['D3'])}
    for name,a,b in [('B_internal_minus_external','internal','external'),('B_internal_minus_best_fixed','internal','best_fixed'),('B_external_minus_best_fixed','external','best_fixed'),('B_internal_minus_keep','internal','keep')]:point[name]=float((outcome(z,bp[a])-outcome(z,bp[b])).mean())
    return {k:dict(difference=float(point[k]),ci95=np.quantile(v,[.025,.975]).tolist(),valid=len(v)) for k,v in values.items()}

def evaluate():
    frozen=read(RUN/'selection_frozen.json');assert frozen['trained_sha256']==hashlib.sha256((RUN/'trained.joblib').read_bytes()).hexdigest()
    obj=joblib.load(RUN/'trained.joblib');f=obj['features'];models=obj['models'];sel=obj['selection'];rows=load('test');x=inputs(rows);y,z=labels(rows);mask=availability(x);ap={};bp={};probs={};cost={}
    for name in ['D1','D2','D3','D4','external','internal']:
        st=time.perf_counter();c=sel[name];xx=f.transform(x,c['kind'],c['key'])
        if name.startswith('D'): ap[name]=models[name].predict_proba(xx)[:,1]
        else: probs[name]=probabilities(models[name],xx);bp[name]=choose(probs[name],mask)
        cost[name]=time.perf_counter()-st
    bp['keep']=np.zeros(len(y),int)
    for name,action in [('best_fixed',sel['best_fixed']),('best_fixed_repair',sel['best_fixed_repair'])]:
        j=ACTIONS.index(action);bp[name]=np.where(mask[:,j],j,0)
    # Explicit verifier's invalid output receives training prevalence; no test tuning.
    prior=float(labels(load('train'))[0].mean());ap['verdict_only']=np.array([1. if r['verdict']=='INCORRECT' else 0. if r['verdict']=='CORRECT' else prior for r in x])
    result=dict(n=len(rows),original_errors=int(y.sum()),original_correct=int((1-y).sum()),A={k:a_metrics(y,p) for k,p in ap.items()},B={k:b_metrics(rows,z,ch) for k,ch in bp.items()},paired_bootstrap=boot(y,ap,z,bp),feature_transform_and_prediction_seconds=cost)
    costs={}
    for split in ['pilot','train','validation','test']:
        rr=load(split);costs[split]=dict(n=len(rr),original_generation_seconds=sum(r['original']['seconds'] for r in rr),verification_generation_seconds=sum(r['verification']['seconds'] for r in rr),hidden_extraction_seconds=sum(r['verification']['feature_seconds'] for r in rr),all_repair_generation_seconds=sum(r['actions'][a]['seconds'] for r in rr for a in ACTIONS[1:]),record_collection_wall_seconds=sum(r['collection_seconds'] for r in rr),max_gpu_allocated_bytes=max(r['peak_gpu_allocated_bytes'] for r in rr),max_gpu_reserved_bytes=max(r['peak_gpu_reserved_bytes'] for r in rr),reused_originals=sum(r['original']['reused'] for r in rr))
    result['costs']=costs
    dump(RUN/'test_results.json',result)
    dump(RUN/'test_predictions.json',dict(ids=[r['id'] for r in rows],A={k:v.tolist() for k,v in ap.items()},B_probabilities={k:v.tolist() for k,v in probs.items()},B_choices={k:v.tolist() for k,v in bp.items()},error_labels=y.tolist(),action_success_labels=z.tolist()))
    # Fixed selected models, no retuning: strict answer format sensitivity on all actions.
    from common import strict_answer,correct
    zs=[]
    for r in rows:
        zs.append([int(correct(strict_answer(r['original']['decoded_output']),r['gold'])),r['actions']['Recalculate']['success'] if r['actions']['Recalculate']['available'] else -1]+[int(correct(strict_answer(r['actions'][a]['decoded_output']),r['gold'])) for a in ['Rebuild','Review']])
    zs=np.array(zs);dump(RUN/'strict_format_sensitivity.json',dict(description='Frozen models and choices; rescoring only, not retrained. Calculator output is numeric and does not require an LLM final-line format.',original_accuracy=float(zs[:,0].mean()),selected_policy_accuracy={k:float(outcome(zs,ch).mean()) for k,ch in bp.items()},D3_AUROC=auc(1-zs[:,0],ap['D3']),D4_AUROC=auc(1-zs[:,0],ap['D4'])))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','evaluate']);a=p.parse_args()
    with threadpool_limits(limits=4): fit() if a.command=='fit' else evaluate()
