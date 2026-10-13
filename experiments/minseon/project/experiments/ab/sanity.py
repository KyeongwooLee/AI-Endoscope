#!/usr/bin/env python3
"""Validation-only label shuffle sanity check after settings are frozen."""
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from evaluate import Features,Constant,load,inputs,labels,fit_binary,auc
from pipeline import RUN,dump
with threadpool_limits(limits=4):
    o=joblib.load(RUN/'trained.joblib');tr=load('train');va=load('validation');x=inputs(tr);v=inputs(va);y,_=labels(tr);vy,_=labels(va);results={}
    for name in ['D3','D4']:
        c=o['selection'][name];tx=o['features'].transform(x,c['kind'],c['key']);vx=o['features'].transform(v,c['kind'],c['key']);scores=[]
        for seed in range(20):
            yy=np.random.default_rng(9000+seed).permutation(y);m=fit_binary(tx,yy,c['C']);scores.append(auc(vy,m.predict_proba(vx)[:,1]))
        results[name]=dict(validation_auroc=scores,mean=float(np.mean(scores)),min=float(np.min(scores)),max=float(np.max(scores)))
    dump(RUN/'label_shuffle_validation.json',results)
