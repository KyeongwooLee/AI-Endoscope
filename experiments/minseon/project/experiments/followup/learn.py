"""Train-only transforms, fixed development choices, frozen fresh-test evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import sys,time,json,hashlib,argparse
from pathlib import Path
import numpy as np,joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from pipeline import read,dump
from evaluate import Constant,fit_binary,fit_selector,probabilities,choose,outcome,labels,ACTIONS
RUN=ROOT/'runs/20261012_followup'
def load(split):return [read(RUN/'records'/f"{r['id']}.json") for r in read(RUN/'manifest.json') if r['split']==split]
def inputs(rows):
    # Explicit whitelist excludes references, scoring labels, and every repair outcome.
    return [dict(id=r['id'],question=r['question'],response=r['original']['decoded_output'],mean_nll=r['original']['mean_nll'],length=sum(i not in (151643,151644,151645) for i in r['original']['generated_ids']),truncated=r['original']['truncated'],numeric_mean=r['original']['numeric']['mean_nll'],numeric_max=r['original']['numeric']['max_nll'],numeric_available=r['original']['numeric']['available'],verdict=r['verification']['verdict'],verdict_nll=r['verification']['mean_nll'],correct_logp=r['verification']['candidate_scores']['CORRECT']['log_probability'],incorrect_logp=r['verification']['candidate_scores']['INCORRECT']['log_probability'],incorrect_prob=r['verification']['candidate_incorrect_probability']) for r in rows]
def availability(x):
    from common import recalculate
    return np.array([[True,recalculate(r['response'],r['truncated'])['available'],True,True] for r in x])
class Features:
    def __init__(self):self.text=TfidfVectorizer(max_features=1500,min_df=2,ngram_range=(1,2),sublinear_tf=True);self.numeric=StandardScaler();self.hs={};self.pca={}
    def texts(self,x):return [r['question']+'\n'+r['response'] for r in x]
    def scalars(self,x):return np.array([[r['mean_nll'],np.log1p(r['length']),r['verdict']=='CORRECT',r['verdict']=='INCORRECT',r['verdict']=='INVALID',r['verdict_nll'],r['numeric_mean'],r['numeric_max'],r['numeric_available'],r['truncated'],r['correct_logp'],r['incorrect_logp'],r['incorrect_prob'],availability([r])[0,1]] for r in x],float)
    def hidden(self,x,key):return np.array([np.load(getattr(self,'hidden_dir',RUN/'hidden')/f"{r['id']}.npz")[key] for r in x],dtype=np.float64)
    def fit(self,x):
        self.text.fit(self.texts(x));self.numeric.fit(self.scalars(x))
        for k in read(RUN/'config.json')['hidden_candidates']:
            h=self.hidden(x,k);self.hs[k]=StandardScaler().fit(h);self.pca[k]=PCA(n_components=32,svd_solver='randomized',random_state=17).fit(self.hs[k].transform(h))
        return self
    def transform(self,x,key='external',rep=None):
        n=self.numeric.transform(self.scalars(x));pieces=[sparse.csr_matrix(n[:,:2])]
        if key!='D1':pieces.append(self.text.transform(self.texts(x)))
        if key=='D3':pieces.append(sparse.csr_matrix(n[:,2:6]))
        elif key not in ('D1','D2'):pieces.append(sparse.csr_matrix(n[:,2:]))
        if rep:
            h=self.hs[key].transform(self.hidden(x,key))
            if rep=='scaled':h=h/np.sqrt(h.shape[1])
            elif rep=='pca32':h=self.pca[key].transform(h)
            elif rep!='standard':raise ValueError(rep)
            pieces.append(sparse.csr_matrix(h))
        out=sparse.hstack(pieces,format='csr');assert np.isfinite(out.data).all();return out

def best_thresholds(vy,vz,mask,errorp,successp,cfg):
    # No component is trained on validation; only policy settings are selected here.
    fixed={a:float(outcome(vz,np.where(mask[:,j],j,0)).mean()) for j,a in enumerate(ACTIONS)}
    fixedaction=max(range(4),key=lambda j:(fixed[ACTIONS[j]],-j))
    best=None
    for j in range(1,4):
        for threshold in cfg['detector_thresholds']:
            ch=np.where((errorp>=threshold)&mask[:,j],j,0);score=(float(outcome(vz,ch).mean()),-int((ch!=0).sum()))
            if best is None or score>best[0]:best=(score,j,threshold)
    repair=1+np.where(mask[:,1:],successp[:,1:],-np.inf).argmax(1);margins=successp[np.arange(len(vy)),repair]-successp[:,0];bestm=None
    for margin in cfg['margin_grid']:
        ch=np.where(margins>margin,repair,0);score=(float(outcome(vz,ch).mean()),-int((ch!=0).sum()))
        if bestm is None or score>bestm[0]:bestm=(score,margin)
    return dict(fixed_action=fixedaction,fixed_validation_accuracy=fixed,trigger_action=best[1],trigger_threshold=best[2],trigger_validation_accuracy=best[0][0],margin=bestm[1],conservative_validation_accuracy=bestm[0][0])

def predict_policies(errorp,successp,mask,settings):
    repair=1+np.where(mask[:,1:],successp[:,1:],-np.inf).argmax(1);margin=successp[np.arange(len(errorp)),repair]-successp[:,0];j=settings['trigger_action']
    return dict(selector=choose(successp,mask),conservative=np.where(margin>settings['margin'],repair,0),trigger=np.where((errorp>=settings['trigger_threshold'])&mask[:,j],j,0))

def fit():
    assert not (RUN/'selection_frozen.json').exists(),'Use a new run for retuning'
    st=time.perf_counter();cfg=read(RUN/'config.json');pool=load('train');va=load('validation');v=inputs(va);vy,vz=labels(va);mask=availability(v);allsettings={};grid=[];curves=[]
    opportunity={}
    for split,rs in [('train_pool',pool),('validation',va)]:
        _,zz=labels(rs);fallback=np.where(zz<0,zz[:,0:1],zz);opportunity[split]=dict(n=len(rs),fixed_policy_accuracy=dict(zip(ACTIONS,fallback.mean(0).tolist())),oracle_accuracy=float(zz.max(1).mean()),oracle_minus_best_fixed=float(zz.max(1).mean()-fallback.mean(0).max()),repairable_errors=int(((zz[:,0]==0)&(zz[:,1:].max(1)==1)).sum()))
    dump(RUN/'development_opportunity_before_test.json',opportunity)
    for seed in cfg['training_seeds']:
        order=np.random.default_rng(seed).permutation(len(pool))
        for n in cfg['sizes']:
            tr=[pool[i] for i in order[:n]];x=inputs(tr);y,z=labels(tr);f=Features().fit(x);models={};sel={};vp={};matrices={}
            candidates=[(k,None) for k in ['D1','D2','D3','external']]+[(k,r) for k in cfg['hidden_candidates'] for r in cfg['representations']]
            # Each position/representation reported; choose one internal overall on development.
            for k,rep in candidates:
                name=k if rep is None else k+'__'+rep;tx=f.transform(x,k,rep);vx=f.transform(v,k,rep);matrices[name]=(tx,vx);bestA=None;bestB=None
                for c in cfg['C_grid']:
                    m=fit_binary(tx,y,c);p=m.predict_proba(vx)[:,1];score=roc_auc_score(vy,p)
                    grid.append(dict(seed=seed,n=n,experiment='A',candidate=name,C=c,validation_AUROC=float(score)))
                    if bestA is None or score>bestA[0]:bestA=(score,c,m,p)
                    if k in ['D1','D2','D3']:continue
                    heads=fit_selector(tx,z,c);sp=probabilities(heads,vx);loss=float(np.mean([log_loss(vz[vz[:,j]>=0,j],sp[vz[:,j]>=0,j],labels=[0,1]) for j in range(4)]))
                    grid.append(dict(seed=seed,n=n,experiment='B',candidate=name,C=c,validation_logloss=loss,validation_accuracy=float(outcome(vz,choose(sp,mask)).mean())))
                    if bestB is None or loss<bestB[0]:bestB=(loss,c,heads,sp)
                score,c,m,p=bestA;models['A_'+name]=m;sel['A_'+name]=dict(key=k,rep=rep,C=c,validation_AUROC=float(score));vp['A_'+name]=p
                if bestB:
                    loss,c,heads,sp=bestB;models['B_'+name]=heads;sel['B_'+name]=dict(key=k,rep=rep,C=c,validation_logloss=loss);vp['B_'+name]=sp
            internalA=max([name for name in sel if name.startswith('A_') and '__' in name],key=lambda name:sel[name]['validation_AUROC']);internalB=min([name for name in sel if name.startswith('B_') and '__' in name],key=lambda name:sel[name]['validation_logloss'])
            chosen={'external':('A_external','B_external'),'internal':(internalA,internalB)};policies={}
            for name,(aa,bb) in chosen.items():
                policies[name]=best_thresholds(vy,vz,mask,vp[aa],vp[bb],cfg)
                pc=predict_policies(vp[aa],vp[bb],mask,policies[name]);curves.append(dict(seed=seed,n=n,variant=name,A_candidate=aa,B_candidate=bb,validation_AUROC=sel[aa]['validation_AUROC'],validation_policies={k:float(outcome(vz,ch).mean()) for k,ch in pc.items()}))
            grouped={}
            if n==1000:
                for variant,(aa,bb) in chosen.items():
                    c=sel[bb];tx=f.transform(x,c['key'],c['rep'])
                    for action in range(1,4):
                        for kind,original_error in [('recovery',1),('damage',0)]:
                            group=(y==original_error)&(z[:,action]>=0);target=z[group,action] if kind=='recovery' else 1-z[group,action];counts=np.bincount(target.astype(int),minlength=2)
                            item=dict(variant=variant,action=action,kind=kind,key=c['key'],rep=c['rep'],C=.1,train_n=int(group.sum()),train_counts=counts.tolist())
                            if min(counts)>=cfg['group_predictor_minority_min']:item['model']=fit_binary(tx[group],target,.1)
                            else:item['deferred']='Fewer than 20 training examples in one outcome class'
                            grouped[f'{variant}_{ACTIONS[action]}_{kind}']=item
            obj=dict(features=f,models=models,selection=sel,chosen=chosen,policies=policies,grouped=grouped,train_ids=[r['id'] for r in tr]);path=RUN/'models'/f'n{n}_seed{seed}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump(obj,path)
            allsettings[f'n{n}_seed{seed}']=dict(representation_summary={k:dict(hidden_dimension=1536,PCA_components=32,PCA_training_variance_retained=float(f.pca[k].explained_variance_ratio_.sum()),preprocessing_fit_n=n) for k in f.pca},chosen=chosen,selection=sel,policies=policies,train_ids=obj['train_ids'],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            dump(RUN/'development'/f'n{n}_seed{seed}.json',dict(ids=[r['id'] for r in va],predictions={k:p.tolist() for k,p in vp.items()},policies=policies));print('Fitted',n,seed,chosen,flush=True)
    dump(RUN/'validation_grid.json',grid);dump(RUN/'development_learning_curves.json',curves)
    dump(RUN/'selection_frozen.json',dict(frozen_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),settings=allsettings,validation_ids=[r['id'] for r in va],training_seconds=time.perf_counter()-st,config_sha256=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest(),code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'experiments/followup').glob('*.py')},test_records_present=len([r for r in (RUN/'records').glob('*.json') if read(r)['split']=='test'])))

def a_metrics(y,p):
    # Undefined subgroup metrics are null, including formatting-only sensitivity
    # cohorts with no compliant/correct answers. Never serialize NaN.
    n=max(1,int(np.ceil(.2*len(y))));idx=np.argsort(-p,kind='stable')[:n];flag=p>=.5
    return dict(AUROC=float(roc_auc_score(y,p)) if len(set(y))>1 else None,AP=float(average_precision_score(y,p)) if y.any() else None,Brier=float(brier_score_loss(y,p)),error_prevalence=float(y.mean()),error_recall_top20=float(y[idx].sum()/y.sum()) if y.any() else None,false_alarm_rate_at_half=float(flag[y==0].mean()) if (y==0).any() else None,accuracy_at_half=float(((p>=.5)==y).mean()))

def b_metrics(rows,z,ch):
    s=outcome(z,ch);keep=z[:,0];wrong=keep==0;right=keep==1
    return dict(accuracy=float(s.mean()),repairs=int(((s==1)&wrong).sum()),damage=int(((s==0)&right).sum()),repair_rate_among_errors=float(s[wrong].mean()) if wrong.any() else None,damage_rate_among_correct=float((1-s[right]).mean()) if right.any() else None,interventions=int((ch!=0).sum()),intervention_frequency=float((ch!=0).mean()),actions={a:int((ch==j).sum()) for j,a in enumerate(ACTIONS)},selected_repair_seconds=sum(r['actions'][ACTIONS[j]]['seconds'] for r,j in zip(rows,ch)),selected_repair_tokens=sum(r['actions'][ACTIONS[j]]['tokens'] for r,j in zip(rows,ch)))
def bootstrap(y,p1,p0,metric,seed=312):
    rng=np.random.default_rng(seed);values=[]
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y))
        if metric=='AUROC':
            if len(set(y[ix]))<2:continue
            values.append(roc_auc_score(y[ix],p1[ix])-roc_auc_score(y[ix],p0[ix]))
        else:values.append(float((p1[ix]-p0[ix]).mean()))
    point=roc_auc_score(y,p1)-roc_auc_score(y,p0) if metric=='AUROC' else float((p1-p0).mean())
    return dict(difference=float(point),ci95=np.quantile(values,[.025,.975]).tolist(),resamples=len(values))
def evaluate():
    cfg=read(RUN/'config.json');frozen=read(RUN/'selection_frozen.json');assert frozen['config_sha256']==hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest();assert frozen['test_records_present']==0
    rows=load('test');assert len(rows)==cfg['split_counts']['test'];x=inputs(rows);y,z=labels(rows);mask=availability(x);results={};predictions={}
    for run,settings in frozen['settings'].items():
        path=RUN/'models'/f'{run}.joblib';assert hashlib.sha256(path.read_bytes()).hexdigest()==settings['sha256'];obj=joblib.load(path);f=obj['features'];ap={};sp={};choices={'keep':np.zeros(len(rows),int)};timing={}
        for name,c in obj['selection'].items():
            st=time.perf_counter();xx=f.transform(x,c['key'],c['rep']);m=obj['models'][name]
            if name.startswith('A_'):ap[name]=m.predict_proba(xx)[:,1]
            elif name in [bb for aa,bb in obj['chosen'].values()]:sp[name]=probabilities(m,xx)
            timing[name]=time.perf_counter()-st
        ap['verdict_only']=np.array([1. if r['verdict']=='INCORRECT' else 0. if r['verdict']=='CORRECT' else .5 for r in x]);ap['candidate_probability_only']=np.array([r['incorrect_prob'] for r in x])
        for variant,(aa,bb) in obj['chosen'].items():
            for policy,ch in predict_policies(ap[aa],sp[bb],mask,obj['policies'][variant]).items():choices[variant+'_'+policy]=ch
        j=obj['policies']['external']['fixed_action'];choices['best_fixed']=np.where(mask[:,j],j,0)
        pairs={};aa=obj['chosen']['internal'][0];pairs['A_internal_minus_external']=bootstrap(y,ap[aa],ap['A_external'],'AUROC')
        for policy in ['selector','conservative','trigger']:
            ext=outcome(z,choices['external_'+policy]);internal=outcome(z,choices['internal_'+policy]);pairs['B_'+policy+'_internal_minus_external']=bootstrap(y,internal,ext,'accuracy')
            for variant in ['external','internal']:
                s=outcome(z,choices[variant+'_'+policy]);pairs[variant+'_'+policy+'_minus_keep']=bootstrap(y,s,z[:,0],'accuracy');pairs[variant+'_'+policy+'_minus_best_fixed']=bootstrap(y,s,outcome(z,choices['best_fixed']),'accuracy')
        for variant in ['external','internal']:pairs[variant+'_selector_minus_trigger']=bootstrap(y,outcome(z,choices[variant+'_selector']),outcome(z,choices[variant+'_trigger']),'accuracy')
        sensitivity={};sensitivityA={}
        for score in ['legacy_success','strict_success']:
            zz=np.array([[r[score]['Keep'],r['actions']['Recalculate']['success'] if r['actions']['Recalculate']['available'] else -1,r[score]['Rebuild'],r[score]['Review']] for r in rows]);sensitivity[score]={k:float(outcome(zz,ch).mean()) for k,ch in choices.items()};sensitivityA[score]={k:a_metrics(1-zz[:,0],p) for k,p in ap.items()}
        grouped={}
        for name,g in obj.get('grouped',{}).items():
            action=g['action'];group=(y==(1 if g['kind']=='recovery' else 0))&(z[:,action]>=0);target=z[group,action] if g['kind']=='recovery' else 1-z[group,action];counts=np.bincount(target.astype(int),minlength=2);item={k:v for k,v in g.items() if k!='model'};item.update(test_n=int(group.sum()),test_counts=counts.tolist())
            if 'model' in g and min(counts)>=2:
                gp=g['model'].predict_proba(f.transform(x,g['key'],g['rep'])[group])[:,1];item.update(AUROC=float(roc_auc_score(target,gp)),Brier=float(brier_score_loss(target,gp)),predictions=gp.tolist())
            grouped[name]=item
        for name,g in grouped.items():
            other=grouped.get(name.replace('internal_','external_',1),{})
            if name.startswith('internal_') and 'predictions' in g and 'predictions' in other:
                action=g['action'];group=(y==(1 if g['kind']=='recovery' else 0))&(z[:,action]>=0);target=z[group,action] if g['kind']=='recovery' else 1-z[group,action];g['paired_internal_minus_external']=bootstrap(target,np.array(g['predictions']),np.array(other['predictions']),'AUROC')
        result=dict(grouped=grouped,A={k:a_metrics(y,p) for k,p in ap.items()},B={k:b_metrics(rows,z,ch) for k,ch in choices.items()},paired=pairs,chosen=obj['chosen'],policies=obj['policies'],sensitivity=sensitivity,sensitivity_A=sensitivityA,cpu_transform_predict_seconds=timing)
        results[run]=result;predictions[run]=dict(ids=[r['id'] for r in rows],A={k:p.tolist() for k,p in ap.items()},B_probabilities={k:p.tolist() for k,p in sp.items()},choices={k:ch.tolist() for k,ch in choices.items()})
        print('Evaluated',run,flush=True)
    dump(RUN/'test_results.json',results);dump(RUN/'test_predictions.json',predictions)
    dump(RUN/'test_outcomes.json',dict(ids=[r['id'] for r in rows],error_labels=y.tolist(),action_success_labels=z.tolist(),oracle_accuracy=float(z.max(1).mean()),original_accuracy=float(z[:,0].mean())))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fit','evaluate']);a=p.parse_args()
    with threadpool_limits(limits=4):fit() if a.command=='fit' else evaluate()
