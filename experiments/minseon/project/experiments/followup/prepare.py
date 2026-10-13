import sys,hashlib,platform,subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from pipeline import read,dump
from common import normalize
RUN=ROOT/'runs/20261012_followup';OLD=ROOT/'runs/20261012_AB'
def main():
 from datasets import load_dataset
 if (RUN/'manifest.json').exists():return
 old=read(OLD/'config.json');cfg={k:old[k] for k in ['model','model_revision','tokenizer_revision','dataset','data_revision','data_config','system_prompt','user_instruction','rebuild_prompt','review_prompt','verification_prompt','verification_system','max_new_tokens','max_repair_tokens','max_verdict_tokens']}
 cfg.update(seed=20261014,dtype='float32',do_sample=False,num_beams=1,repetition_penalty=1.1,batch_size=8,repair_batch_size=16,blocks=[14],hidden_candidates=['numeric_L14','boundary_L14','preverdict_L14'],representations=['scaled','standard','pca32'],C_grid=[.01,.1,1.],sizes=[200,500,1000],training_seeds=[101,202],split_counts={'pilot':40,'train':1200,'validation':200,'test':300},scoring='scoring.py v2; old extraction saved as sensitivity; strict format separate',calculator='unchanged legacy calculator; no extensions',candidate_verdicts=['CORRECT','INCORRECT'],verdict_score='sum raw autoregressive log probability of all label tokens, no EOS; normalize two full label sequence probabilities; no length normalization',hidden_positions='last token overlapping reference-free extracted numeric span in verification prompt; dedicated boundary; final pre-verdict prompt token. Raw post-block14, pre model.norm.',missing_numeric_hidden='zero vector + shared availability scalar; no substitution with another position',representation_details={'scaled':'train coordinate StandardScaler / sqrt(1536), original approach','standard':'train coordinate StandardScaler, no block shrinkage','pca32':'train coordinate StandardScaler then train-fit randomized PCA32, no whitening'},selection='A validation AUROC. B heads validation mean available-head logloss; policies validation accuracy, ties fewer interventions. Fixed dev; no test access before freeze.',margin_grid=[0.,.025,.05,.1,.2,1.01],detector_thresholds=[0.,.25,.5,.75,1.01],bootstrap_iterations=2000,primary_training_seed=101,group_predictor_minority_min=20)
 canonical=ROOT/'experiments/followup/config.json'
 if canonical.exists():cfg=read(canonical)
 # Fresh official-train pool avoids every previous local sampled question, including pilot/dev.
 prior=read(ROOT/'runs/20261009T0542Z/manifest.json')+read(OLD/'manifest.json');seen={r['question_sha256'] for r in prior};rows=[]
 for partition,counts in [('train',[('pilot',40),('train',1200),('validation',200)]),('test',[('test',300)])]:
  ds=load_dataset(cfg['dataset'],'main',split=partition,revision=cfg['data_revision']);indices=iter(np.random.default_rng(cfg['seed']+(partition=='test')).permutation(len(ds)))
  for split,n in counts:
   added=0
   while added<n:
    i=int(next(indices));r=ds[i];h=hashlib.sha256(' '.join(r['question'].split()).encode()).hexdigest()
    if h in seen:continue
    seen.add(h);rows.append(dict(id=f'gsm8k_{partition}_{i}',source_index=i,source_partition=partition,split=split,question=r['question'],question_sha256=h,gold_answer=r['answer'],gold=normalize(r['answer'].split('####')[-1])));added+=1
 dump(RUN/'config.json',cfg);dump(RUN/'manifest.json',rows)
 dump(RUN/'environment.json',dict(python=sys.version,gpu=subprocess.getoutput('nvidia-smi'),disk=subprocess.getoutput('df -h . /tmp')))
 dump(RUN/'protocol_initial.json',dict(config_sha256=hashlib.sha256((RUN/'config.json').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((RUN/'manifest.json').read_bytes()).hexdigest(),old_test_excluded=True,all_old_questions_excluded=True,test_not_generated=True,primary_comparison='internal minus external at n=1000 seed101; other sizes/seeds/representations descriptive secondary',feasibility='1200 training pool enables two distinct nested 200/500/1000 samples, dev200 and new test300; pilot40 gates main run'))
 print('Prepared',len(rows),'fresh examples')
if __name__=='__main__':main()
