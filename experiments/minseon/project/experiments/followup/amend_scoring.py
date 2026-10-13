"""One-time, pre-fit v2→v3 measurement amendment. Never regenerate responses."""
import sys,shutil,hashlib,time
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from collect import original_numeric,RUN,read,dump
from scoring import extract,correct

def main():
 assert not (RUN/'selection_frozen.json').exists();assert not any(read(p)['split']=='test' for p in (RUN/'records').glob('*.json'))
 archive=RUN/'measurement_v2_archive';archive.mkdir(exist_ok=True);cfg=read(RUN/'config.json')
 class TokenizerOnly:pass
 e=TokenizerOnly();e.tok=AutoTokenizer.from_pretrained(cfg['model'],revision=cfg['tokenizer_revision'])
 changed=[];reextract=[];count=0
 for p in sorted((RUN/'records').glob('*.json')):
  r=read(p);rid=r['id'];shutil.copy2(p,archive/p.name);count+=1;old=r['original']['numeric']['parsed'];new=extract(r['original']['decoded_output'],r['original']['truncated']);o=r['original'];o['numeric']=original_numeric(e,o)
  dump(RUN/'stages'/rid/'original.json',o)
  for a in ['Keep','Rebuild','Review']:
   obj=o if a=='Keep' else r['actions'][a];parsed=extract(obj['decoded_output'],obj['truncated']);prior=r['actions'][a]['success'];r['actions'][a]['answer']=parsed['answer'];r['actions'][a]['success']=int(correct(parsed['answer'],r['gold']))
   if a!='Keep':r['actions'][a]['parse']=parsed
   if prior!=r['actions'][a]['success']:changed.append(dict(id=rid,split=r['split'],action=a,v2_success=prior,v3_success=r['actions'][a]['success'],v3_answer=parsed['answer']))
  r['label_error']=1-r['actions']['Keep']['success']
  if old['span']!=new['span']:
   # Preserve every old feature, remove only its active pointer; raw generations stay.
   for f in [RUN/'stages'/rid/'verification.json',RUN/'hidden'/f'{rid}.npz']:
    if f.exists():shutil.move(str(f),str(archive/(rid+'_'+f.name)))
   p.unlink();reextract.append(rid)
  else:
   r['verification']['numeric_parse']=new;dump(RUN/'stages'/rid/'verification.json',r['verification']);dump(p,r)
 # Update staged originals from a partially completed group, with no new generation.
 for f in (RUN/'stages').glob('*/original.json'):
  rid=f.parent.name
  if not (RUN/'records'/f'{rid}.json').exists():
   o=read(f);old=o['numeric']['parsed'];new=extract(o['decoded_output'],o['truncated']);o['numeric']=original_numeric(e,o);dump(f,o)
   vp=f.parent/'verification.json';hp=RUN/'hidden'/f'{rid}.npz'
   if vp.exists():
    if old['span']!=new['span']:
     shutil.move(str(vp),str(archive/(rid+'_partial_verification.json')))
     if hp.exists():shutil.move(str(hp),str(archive/(rid+'_partial_hidden.npz')))
    else:v=read(vp);v['numeric_parse']=new;dump(vp,v)
 cfg['scoring']='scoring.py v3; pre-fit measurement amendment fixes next-line/prose marker and contextual-number regressions; legacy and strict sensitivities retained'
 dump(RUN/'config.json',cfg);dump(Path(__file__).resolve().parent/'config.json',cfg)
 dump(RUN/'scoring_v3_amendment.json',dict(at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='Three unambiguous parser regressions found during training-only measurement audit before any classifier fitting. No fresh-test response exists.',archived_records=count,changed_labels=changed,hidden_reextraction_ids=reextract,raw_original_and_repair_generations_reused=True,prior_collection_protocol=read(RUN/'collection_protocol_frozen.json')))
 dump(RUN/'collection_protocol_frozen_v3.json',dict(config_sha256=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest(),scoring_sha256=hashlib.sha256((Path(__file__).resolve().parent/'scoring.py').read_bytes()).hexdigest(),no_models_fit=True,no_test_generated=True))
 print('Archived',count,'records; changed labels',changed,'reextract',reextract,flush=True)
if __name__=='__main__':main()
