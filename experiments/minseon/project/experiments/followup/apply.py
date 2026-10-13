"""Apply frozen methods to whitelisted pre-repair features, with no reference labels."""
import argparse,json,hashlib
from pathlib import Path
import joblib,numpy as np
from learn import Features,Constant,RUN,read,dump,availability,predict_policies
from evaluate import probabilities,ACTIONS

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--hidden-dir',type=Path,required=True);p.add_argument('--model',default='n1000_seed101');p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=read(a.inputs)
 allowed={'id','question','response','mean_nll','length','truncated','numeric_mean','numeric_max','numeric_available','verdict','verdict_nll','correct_logp','incorrect_logp','incorrect_prob'}
 assert all(set(r)==allowed for r in x),'Pass exactly the documented pre-repair input whitelist'
 frozen=read(RUN/'selection_frozen.json');path=RUN/'models'/f'{a.model}.joblib';assert hashlib.sha256(path.read_bytes()).hexdigest()==frozen['settings'][a.model]['sha256'];obj=joblib.load(path);f=obj['features'];f.hidden_dir=a.hidden_dir;mask=availability(x);result={}
 for variant,(aa,bb) in obj['chosen'].items():
  ca=obj['selection'][aa];cb=obj['selection'][bb];error=obj['models'][aa].predict_proba(f.transform(x,ca['key'],ca['rep']))[:,1];success=probabilities(obj['models'][bb],f.transform(x,cb['key'],cb['rep']));policies=predict_policies(error,success,mask,obj['policies'][variant]);result[variant]=dict(error_probability=error.tolist(),action_success_probabilities=success.tolist(),choices={k:[ACTIONS[j] for j in ch] for k,ch in policies.items()})
 dump(a.output,dict(ids=[r['id'] for r in x],predictions=result))
if __name__=='__main__':main()
