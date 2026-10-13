"""Exploratory train/development logit contributions of frozen original models."""
import sys
from pathlib import Path
import numpy as np,joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from evaluate import Features,Constant,inputs
from pipeline import read,dump
OLD=ROOT/'runs/20261012_AB';RUN=ROOT/'runs/20261012_followup'
def main():
 obj=joblib.load(OLD/'trained.joblib');f=obj['features'];result={}
 for split in ['train','validation']:
  rows=[read(OLD/'records'/f"{r['id']}.json") for r in read(OLD/'manifest.json') if r['split']==split];x=inputs(rows);result[split]={}
  for name in ['D4','internal']:
   s=obj['selection'][name];xx=f.transform(x,s['kind'],s['key']);models=[obj['models'][name]] if name=='D4' else obj['models'][name];blocks={'generation':(0,2),'text':(2,2+len(f.text.vocabulary_)),'verification':(2+len(f.text.vocabulary_),xx.shape[1]-1536),'hidden':(xx.shape[1]-1536,xx.shape[1])};rr=[]
   for model in models:
    if not hasattr(model,'coef_'):rr.append({'constant':model.p});continue
    w=model.coef_[0];rr.append({key:dict(coefficient_L2=float(np.linalg.norm(w[lo:hi])),logit_contribution_mean=float(np.asarray(xx[:,lo:hi]@w[lo:hi]).mean()),logit_contribution_sd=float(np.asarray(xx[:,lo:hi]@w[lo:hi]).std())) for key,(lo,hi) in blocks.items()})
   result[split][name]=rr
 dump(RUN/'original_logit_contributions.json',result);print('Old A validation logit contribution SD:',{k:round(v['logit_contribution_sd'],4) for k,v in result['validation']['D4'][0].items()})
if __name__=='__main__':main()
