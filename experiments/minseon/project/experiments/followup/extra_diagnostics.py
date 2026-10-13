import json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import brier_score_loss
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from pipeline import read,dump
OLD=ROOT/'runs/20261012_AB';RUN=ROOT/'runs/20261012_followup';ACTIONS=['Keep','Recalculate','Rebuild','Review']
def main():
 out={}
 for split in ['validation','test']:
  pred=read(OLD/('test_predictions.json' if split=='test' else 'validation_predictions.json'));rows=[read(OLD/'records'/f'{rid}.json') for rid in pred['ids']];z=np.array([[r['actions'][a]['success'] if r['actions'][a]['available'] else -1 for a in ACTIONS] for r in rows]);out[split]={}
  for variant in ['external','internal']:
   p=np.array(pred['B_probabilities' if split=='test' else 'predictions'][variant]);d={}
   for j,a in enumerate(ACTIONS):
    mask=z[:,j]>=0;d[a]=dict(mean_probability=float(p[mask,j].mean()),actual_success_rate=float(z[mask,j].mean()),Brier=float(brier_score_loss(z[mask,j],p[mask,j])),correlation_with_keep=float(np.corrcoef(p[:,0],p[:,j])[0,1]),by_original_correctness={})
    for c in [0,1]:
     ix=mask&(z[:,0]==c);d[a]['by_original_correctness'][str(c)]=dict(n=int(ix.sum()),mean_predicted_success=float(p[ix,j].mean()),actual_success_rate=float(z[ix,j].mean()),mean_predicted_gain=float((p[ix,j]-p[ix,0]).mean()),actual_gain=float((z[ix,j]-z[ix,0]).mean()))
   out[split][variant]=d
  if split=='validation':
   rng=np.random.default_rng(523);wins=np.zeros(4,int);zz=np.where(z<0,z[:,0:1],z)
   for _ in range(2000):wins[zz[rng.integers(len(z),size=len(z))].mean(0).argmax()]+=1
   out[split]['fixed_policy_bootstrap_winners_with_Keep_first_ties']=dict(zip(ACTIONS,(wins/2000).tolist()))
 dump(RUN/'original_calibration_diagnostics.json',out);print(json.dumps(out['validation']['fixed_policy_bootstrap_winners_with_Keep_first_ties']));print('Keep-Review probability correlations',[(s,v,out[s][v]['Review']['correlation_with_keep']) for s in ['validation','test'] for v in ['external','internal']])
if __name__=='__main__':main()
