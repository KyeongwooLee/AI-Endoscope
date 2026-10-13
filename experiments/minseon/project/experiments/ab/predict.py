#!/usr/bin/env python3
"""Apply frozen classifiers to feature records without any reference or outcomes.
Input JSON: list with id, question, original {decoded_output,mean_nll,
generated_ids,truncated}, verification {verdict,mean_nll}. Hidden .npz per id.
"""
import argparse,json
from pathlib import Path
import joblib
from evaluate import Features,Constant,inputs,availability,choose,probabilities,ACTIONS
from pipeline import RUN,dump
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');p.add_argument('--hidden-dir',type=Path,default=RUN/'hidden');a=p.parse_args()
obj=joblib.load(RUN/'trained.joblib');f=obj['features'];f.hidden_dir=a.hidden_dir
x=inputs(json.loads(Path(a.input).read_text()));mask=availability(x);result={'ids':[r['id'] for r in x],'A':{},'B':{}}
for name,c in obj['selection'].items():
    if name not in ['D1','D2','D3','D4','external','internal']:continue
    xx=f.transform(x,c['kind'],c['key']);model=obj['models'][name]
    if name.startswith('D'):result['A'][name]=model.predict_proba(xx)[:,1].tolist()
    else:
        prob=probabilities(model,xx);result['B'][name]={'probabilities':prob.tolist(),'actions':[ACTIONS[i] for i in choose(prob,mask)]}
dump(a.output,result)
