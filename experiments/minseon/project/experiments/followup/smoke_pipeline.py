"""End-to-end CPU wiring test with duplicated excluded-pilot fixtures; NOT research results."""
import sys,tempfile,shutil
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from learn import Features,Constant
import learn
from pipeline import read,dump

def main():
 original=learn.RUN;pilot=[read(p) for p in sorted((original/'records').glob('*.json')) if read(p)['split']=='pilot'];assert len(pilot)==40
 with tempfile.TemporaryDirectory(prefix='endoscope_smoke_') as folder:
  root=Path(folder);learn.RUN=root;(root/'records').mkdir();(root/'hidden').mkdir();cfg=read(original/'config.json');cfg.update(sizes=[32],training_seeds=[101],split_counts={'train':32,'validation':8,'test':8});dump(root/'config.json',cfg);manifest=[];tests=[]
  for split,rows in [('train',pilot[:32]),('validation',pilot[32:]),('test',pilot[:8])]:
   for r in rows:
    rid=r['id']+'_'+split;src=original/'hidden'/f"{r['id']}.npz";shutil.copyfile(src,root/'hidden'/f'{rid}.npz');r=dict(r,id=rid,split=split);manifest.append(dict(id=rid,split=split))
    if split=='test':tests.append(r)
    else:dump(root/'records'/f'{rid}.json',r)
  dump(root/'manifest.json',manifest)
  with threadpool_limits(limits=2):
   learn.fit()
   for r in tests:dump(root/'records'/f"{r['id']}.json",r)
   learn.evaluate()
  assert read(root/'selection_frozen.json')['test_records_present']==0
  result=read(root/'test_results.json');assert len(result['n32_seed101']['B'])==8
  dump(original/'smoke_test.json',dict(passed=True,scope='CPU wiring only; duplicated excluded-pilot fixtures; no scientific claims',policies=list(result['n32_seed101']['B'])))
 print('End-to-end wiring test passed',flush=True)
if __name__=='__main__':main()
