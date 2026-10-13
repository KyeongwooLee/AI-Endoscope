"""Validate inference provenance, hidden positions, labels, split isolation and old preservation."""
import sys,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from pipeline import read,dump
from scoring import extract,correct,recalculate
RUN=ROOT/'runs/20261012_followup';OLD=ROOT/'runs/20261012_AB'
def main():
 manifest=read(RUN/'manifest.json');old=read(RUN/'original_preservation.json')
 for name,h in old.items():assert hashlib.sha256((OLD/name).read_bytes()).hexdigest()==h,name
 prior=read(OLD/'manifest.json')+read(ROOT/'runs/20261009T0542Z/manifest.json')+read(ROOT/'runs/20261009T0535Z/manifest.json');assert not ({r['question_sha256'] for r in manifest}&{r['question_sha256'] for r in prior});assert len({r['question_sha256'] for r in manifest})==len(manifest)
 results={};differences=[];positions=[]
 for m in manifest:
  p=RUN/'records'/f"{m['id']}.json"
  if not p.exists():continue
  r=read(p);assert r['split']==m['split'];assert m['source_partition']==('test' if r['split']=='test' else 'train');assert r['actions']['Keep']['success']==1-r['label_error'];assert r['actions']['Recalculate']['available']==recalculate(r['original']['decoded_output'],r['original']['truncated'])['available']
  o=r['original'];v=r['verification'];parsed=extract(o['decoded_output'],o['truncated']);assert o['numeric']['parsed']==parsed
  assert v['numeric_parse']==parsed,(r['id'],v['numeric_parse'],parsed)
  for a,obj in r['actions'].items():assert obj['success']==(int(correct(obj['answer'],r['gold'])) if obj['available'] else None)
  assert v['positions']['boundary']<v['positions']['preverdict']==len(v['prompt_ids'])-1
  if parsed['span']:
   assert v['positions']['numeric']<v['positions']['boundary'];off=v['position_tokens']['numeric']['offset'];start=v['response_char_span'][0];assert off[0]<start+parsed['span'][1] and off[1]>start+parsed['span'][0]
  for label in ['CORRECT','INCORRECT']:
   score=v['candidate_scores'][label];assert abs(sum(score['token_nll'])+score['log_probability'])<1e-6
  h=np.load(RUN/'hidden'/f"{r['id']}.npz");assert set(h.files)=={'numeric_L14','boundary_L14','preverdict_L14'};assert all(h[k].shape==(1536,) and np.isfinite(h[k]).all() for k in h.files)
  if not parsed['span']:assert np.count_nonzero(h['numeric_L14'])==0
  for a in ['Keep','Rebuild','Review']:
   if r['legacy_success'][a]!=r['actions'][a]['success']:differences.append(dict(id=r['id'],split=r['split'],action=a,legacy=r['legacy_success'][a],v3=r['actions'][a]['success']))
  results[r['split']]=results.get(r['split'],0)+1
 dump(RUN/'integrity_audit.json',dict(scoring_version='v3',passed=True,completed=results,old_files_preserved=len(old),no_old_question_overlap=True,all_hidden_position_and_label_checks_passed=True,scoring_label_differences=differences));print(results,flush=True)
if __name__=='__main__':main()
