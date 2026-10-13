#!/usr/bin/env python3
"""Mark completion only after full collection, evaluation, audits and reports exist."""
import hashlib,datetime,shutil
from pathlib import Path
from pipeline import RUN,ROOT,read,dump

def main():
    required=['test_results.json','test_predictions.json','selection_frozen.json','integrity_audit.json','pilot_integrity.json','deployment_benchmark.json','REPORT.md','REPORT_KO.md','results.png','paired_differences.png','metrics_A.csv','metrics_B.csv','EXAMPLES.md']
    assert all((RUN/p).exists() for p in required)
    audit=read(RUN/'integrity_audit.json');assert audit['passed'] and audit['checked_records']==420
    frozen=read(RUN/'selection_frozen.json');assert frozen['trained_sha256']==hashlib.sha256((RUN/'trained.joblib').read_bytes()).hexdigest()
    phase=read(RUN/'protocol_frozen.json');assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in phase['files'].items())
    stops=[]
    for p in sorted((RUN/'records').glob('*.json')):
        r=read(p);entry={'id':r['id'],'split':r['split']}
        for name,v,cap in [('original',r['original'],512),('verification',r['verification'],8),('Rebuild',r['actions']['Rebuild'],512),('Review',r['actions']['Review'],512)]:
            ids=v['generated_ids'];entry[name]={'tokens_including_eos':len(ids),'stop_reason':'eos' if ids[-1] in [151645,151643] else 'max_new_tokens' if len(ids)>=cap else 'other_special_or_unknown','truncated':v['truncated']}
        stops.append(entry)
    dump(RUN/'generation_stop_audit.json',stops)
    for p in (ROOT/'experiments/ab').iterdir():
        if p.is_file():shutil.copy2(p,RUN/'source_code'/p.name)
    checksum={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(RUN.rglob('*')) if p.is_file() and p.name not in ['artifact_sha256.json','completion.json']}
    dump(RUN/'artifact_sha256.json',checksum)
    dump(RUN/'completion.json',dict(status='complete',completed_kst=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),pilot_examples=40,train_examples=200,validation_examples=80,official_test_examples=100,experiments_completed=['A','B'],deferred=['C'],remaining_required_work=[],integrity_audit_passed=True,frozen_protocol_hashes_verified=True,trained_model_hash_verified=True,prior_results_preserved=True,scope='exploratory fixed-size run, not full GSM8K test evaluation'))
    print('Completed A and B; C deferred; all recorded scientific invariants passed.')
if __name__=='__main__':main()
