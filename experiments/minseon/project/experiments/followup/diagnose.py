"""Read-only exploratory diagnosis of the immutable first A/B run."""
import sys,json,csv,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OLD=ROOT/'runs/20261012_AB'; RUN=ROOT/'runs/20261012_followup'
sys.path.insert(0,str(ROOT/'experiments/ab'))
from common import answer,strict_answer,correct,NUM
from pipeline import read,dump
ACTIONS=['Keep','Recalculate','Rebuild','Review']
def main():
    manifest=read(OLD/'manifest.json'); result={}; cases=[]; audit=[]; rng=np.random.default_rng(20261014)
    test=read(OLD/'test_predictions.json'); val=read(OLD/'validation_predictions.json')
    for split in ['pilot','train','validation','test']:
        rows=[read(OLD/'records'/f"{m['id']}.json") for m in manifest if m['split']==split]; n=len(rows)
        z=np.array([[r['actions'][a]['success'] if r['actions'][a]['available'] else -1 for a in ACTIONS] for r in rows]); keep=z[:,0]; err=keep==0; repair=(z[:,1:]==1)&err[:,None]
        d=dict(n=n,original_correct=int(keep.sum()),oracle_accuracy=float((z.max(1)==1).mean()),repairable_errors=int(repair.any(1).sum()),actions={},repair_overlap={},calculator_reasons=dict(Counter(r['actions']['Recalculate']['reason'] for r in rows)))
        for j,a in enumerate(ACTIONS):
            available=z[:,j]>=0
            d['actions'][a]={f'{x}->{y}':int(((keep==x)&(z[:,j]==y)&available).sum()) for x,y in [(1,1),(1,0),(0,1),(0,0)]}
            d['actions'][a].update(available=int(available.sum()),unavailable=int((~available).sum()),unique_repairs=int((repair[:,j-1]& (np.delete(repair,j-1,axis=1).sum(1)==0)).sum()) if j else 0)
        for j,a in enumerate(ACTIONS[1:]):
            for k,b in enumerate(ACTIONS[1:]): d['repair_overlap'][a+' & '+b]=int((repair[:,j]&repair[:,k]).sum())
        changes=np.array([r['actions']['Review']['answer']!=r['actions']['Keep']['answer'] for r in rows]);d['review_answer_changes']=int(changes.sum());d['review_preserves']=int((~changes).sum())
        d['review_changed_outcomes']={f'{x}->{y}':int((changes&(keep==x)&(z[:,3]==y)).sum()) for x,y in [(1,1),(1,0),(0,1),(0,0)]}
        if split in ('validation','test'):
            pp= test['B_probabilities'] if split=='test' else {k:val['predictions'][k] for k in ['external','internal']};ids=test['ids'] if split=='test' else val['ids'];assert ids==[r['id'] for r in rows];d['selectors']={}
            for name,p in pp.items():
                p=np.array(p);ch=np.where(z>=0,p,-np.inf).argmax(1);success=z[np.arange(n),ch]; margins=p[np.arange(n),ch]-p[:,0]
                groups={'repair':err&(success==1),'harm':(~err)&(success==0),'missed_repairable':repair.any(1)&(success==0),'all_interventions':ch!=0}
                d['selectors'][name]={g:dict(n=int(mask.sum()),mean_chosen_probability=float(p[np.arange(n),ch][mask].mean()) if mask.any() else None,mean_keep_probability=float(p[:,0][mask].mean()) if mask.any() else None,mean_margin=float(margins[mask].mean()) if mask.any() else None) for g,mask in groups.items()}
                for i,r in enumerate(rows):
                    cases.append(dict(split=split,id=r['id'],selector=name,original_correct=int(keep[i]),oracle_correct=int(z[i].max()==1),chosen=ACTIONS[ch[i]],success=int(success[i]),probabilities=p[i].tolist(),margin=float(margins[i]),successful_actions=[a for j,a in enumerate(ACTIONS) if z[i,j]==1]))
        # All scoring disagreements and all damaging generative revisions, plus random agreements.
        agreements=[]
        for r in rows:
            for a in ['Keep','Rebuild','Review']:
                o=r['original'] if a=='Keep' else r['actions'][a]; txt=o['decoded_output']; pred,status=answer(txt,o['truncated']);strict=strict_answer(txt)
                reason=[]
                if correct(pred,r['gold'])!=correct(strict,r['gold']):reason.append('numeric_vs_strict_scoring')
                if a!='Keep' and r['actions']['Keep']['success']==1 and r['actions'][a]['success']==0:reason.append('harmful_revision')
                item=dict(id=r['id'],split=split,action=a,reasons=reason,question=r['question'],text=txt,gold=r['gold'],prediction=pred,parse=status,strict_prediction=strict,numerical_success=correct(pred,r['gold']),strict_success=correct(strict,r['gold']))
                (audit if reason else agreements).append(item)
        for i in rng.choice(len(agreements),min(10,len(agreements)),replace=False):agreements[int(i)]['reasons']=['random_agreement'];audit.append(agreements[int(i)])
        result[split]=d
    # Empirical norm diagnostic uses training only.
    from sklearn.preprocessing import StandardScaler
    tr=[read(OLD/'records'/f"{m['id']}.json") for m in manifest if m['split']=='train'];scale={}
    for key in ['preverdict_L14','boundary_L28']:
        h=np.array([np.load(OLD/'hidden'/f"{r['id']}.npz")[key] for r in tr]);s=StandardScaler().fit_transform(h)
        scale[key]=dict(dimension=h.shape[1],mean_raw_norm=float(np.linalg.norm(h,axis=1).mean()),mean_standardized_norm=float(np.linalg.norm(s,axis=1).mean()),mean_current_norm=float(np.linalg.norm(s/np.sqrt(h.shape[1]),axis=1).mean()),effective_hidden_coefficient_penalty_multiplier=h.shape[1])
    dump(RUN/'original_diagnostics.json',result);dump(RUN/'original_selector_cases.json',cases);dump(RUN/'scoring_audit_cases.json',audit);dump(RUN/'hidden_scaling_diagnostic.json',scale)
    dump(RUN/'original_preservation.json',{str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OLD.rglob('*') if p.is_file()})
    lines=['# Original-run exploratory diagnosis','Original test data are diagnostic only; no follow-up settings selected from these outcomes.','', '| Split | N | Keep | Full oracle | Repairable errors | Review changes |','|---|---:|---:|---:|---:|---:|']
    for s,d in result.items():lines.append(f"| {s} | {d['n']} | {d['original_correct']/d['n']:.3f} | {d['oracle_accuracy']:.3f} | {d['repairable_errors']} | {d['review_answer_changes']} |")
    for s,d in result.items():
        lines += ['',f'## {s}','| Action | C→C | C→I | I→C | I→I | Unavailable | Unique repairs |','|---|---:|---:|---:|---:|---:|---:|']
        for a,v in d['actions'].items():lines.append('| '+a+' | '+' | '.join(str(v[k]) for k in ['1->1','1->0','0->1','0->0','unavailable','unique_repairs'])+' |')
        lines+=['','```json',json.dumps({k:d[k] for k in ['repair_overlap','calculator_reasons','review_changed_outcomes']},indent=2),'```']
        if 'selectors' in d:lines+=['','```json',json.dumps(d['selectors'],indent=2),'```']
    (RUN/'DIAGNOSTICS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result['test'],indent=2)); print('Audit cases',len(audit)); print(scale)
if __name__=='__main__':main()
