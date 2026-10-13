"""Mark complete only after artifacts, frozen settings and preservation checks pass."""
import hashlib,time,shutil,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/ab'))
from pipeline import read,dump
RUN=ROOT/'runs/20261012_followup'
def main():
 for f in ['REPORT.md','test_results.json','test_predictions.json','deployment_benchmark.json','integrity_audit.json','selection_frozen.json','smoke_test.json','application_smoke_check.json']:
  assert (RUN/f).exists(),f
 assert read(RUN/'smoke_test.json')['passed'];assert read(RUN/'application_smoke_check.json')['passed']
 for name,h in read(RUN/'provenance.json')['source_hashes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 audit=read(RUN/'integrity_audit.json');cfg=read(RUN/'config.json');assert audit['passed'];assert audit['completed']==cfg['split_counts'];frozen=read(RUN/'selection_frozen.json');assert frozen['test_records_present']==0
 for name,s in frozen['settings'].items():assert hashlib.sha256((RUN/'models'/f'{name}.joblib').read_bytes()).hexdigest()==s['sha256']
 for name in ['learn.py','collect.py','scoring.py']:assert hashlib.sha256((ROOT/'experiments/followup'/name).read_bytes()).hexdigest()==frozen['code_sha256'][name]
 snapshot=RUN/'source_code';snapshot.mkdir(exist_ok=True)
 for f in (ROOT/'experiments/followup').iterdir():
  if f.is_file():shutil.copy2(f,snapshot/f.name)
 dump(RUN/'completion.json',dict(status='complete',completed_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),A='completed',B='completed',C='deferred',FinQA='deferred',larger_models='deferred',activation_steering='deferred',records=audit['completed'],preserved_old_files=audit['old_files_preserved']))
 (RUN/'STATUS.md').write_text('# Follow-up completed\n\nExperiments A and B, original-run diagnostics, scoring audit, learning curves, fresh held-out evaluation and deployment benchmark are complete. See [REPORT.md](REPORT.md), completion.json and the saved validation/test artifacts. Experiment C, FinQA, larger models and activation steering are deferred.\n')
 dump(RUN/'artifact_sha256.json',{str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*') if p.is_file() and p.name!='artifact_sha256.json' and p.suffix not in ('.log','.pid')})
 print('Follow-up complete',flush=True)
if __name__=='__main__':main()
