"""Recheck genuine local backups and export selected errors for later annotation."""
import argparse
from collections import Counter
import json
import io
from pathlib import Path
import sys
import zipfile

import numpy as np

from protocol import verify,write_json

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/gsm8k-logit-study'))
from gsm8k_lens.answers import label_answer

def loads_jsonl(data):
    # JSONL separates records with LF. Unicode paragraph separators may occur
    # literally inside JSON string values and must not split a record.
    return [json.loads(line) for line in data.decode('utf-8').split('\n') if line.strip()]

def audit(final=False):
    backup=ROOT/'backup'
    rows=[]
    accepted={}
    seen=set(json.loads((ROOT/'input/existing_pairs.json').read_text(encoding='utf-8')))
    for receipt_path in sorted((backup/'outbox').glob('generations_*.json')):
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        verify(receipt_path.with_suffix('.zip'),receipt)
        with zipfile.ZipFile(receipt_path.with_suffix('.zip')) as archive:
            current=loads_jsonl(archive.read('records.jsonl'))
        assert [r['id'] for r in current]==receipt['ids']
        for r in current:
            expected=label_answer(r['generation']['text'],r['reference'],r['generation']['truncated'])
            assert expected==r['label'],r['id']
            pair=r['question_group']+':'+r['response_sha256']
            if r['selected_new_error']:
                assert pair not in seen and expected['analysis_eligible'] and not expected['answer_correct'],r['id']
                assert r['id'] not in accepted
                accepted[r['id']]=r
            seen.add(pair)
        rows+=current
    assert len({r['id'] for r in rows})==len(rows)
    groups={}
    for r in rows:
        assert groups.setdefault(r['question_group'],r['fold'])==r['fold']
    featured=set()
    qa=Counter()
    for receipt_path in sorted((backup/'outbox').glob('features_*.json')):
        receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
        verify(receipt_path.with_suffix('.zip'),receipt)
        with zipfile.ZipFile(receipt_path.with_suffix('.zip')) as archive:
            records=loads_jsonl(archive.read('records.jsonl'))
            for r in records:
                assert r['id'] in accepted and r['id'] not in featured
                assert r['response_sha256']==accepted[r['id']]['response_sha256']
                # Read each outer ZIP member once. Seeking inside a nested
                # ZipExtFile otherwise repeatedly decompresses the entire NPZ.
                with io.BytesIO(archive.read('features/'+r['id']+'.npz')) as file:
                    with np.load(file,allow_pickle=False) as arrays:
                        assert arrays['hidden'].shape[1:]==(29,1536)
                        assert arrays['pooled_hidden'].shape==(28,3,1536)
                        assert arrays['confidence'].shape==(26,)
                        assert arrays['length'].shape==(4,)
                        for name in ['posthoc_question','posthoc_end','posthoc_mean','posthoc_number']:
                            assert arrays[name].shape==(1536,) and np.isfinite(arrays[name]).all()
                        np.testing.assert_array_equal(arrays['posthoc_question'],arrays['hidden'][0,-1])
                        np.testing.assert_array_equal(arrays['posthoc_end'],arrays['pooled_hidden'][-1,0])
                featured.add(r['id'])
                qa['prefix_checks']+=r['feature_quality'].get('independent_prefix_checks',0)
                if len(featured)%100==0:
                    write_json(ROOT/'AUDIT_PROGRESS.json',{'checked_features':len(featured),'total':len(accepted)})
    assert featured<=set(accepted)
    result={'generated':len(rows),'selected_errors':len(accepted),'verified_features':len(featured),
            'datasets':dict(Counter(r['dataset'] for r in accepted.values())),
            'raw_outcomes':dict(Counter(r['status'] for r in rows)),
            'unique_selected_questions':len({r['question_group'] for r in accepted.values()}),
            'duplicate_response_and_grade_checks_passed':True,'feature_shape_and_link_checks_passed':True,
            'qa':dict(qa),'complete':final,'taxonomy_status':'unannotated'}
    if final:
        assert len(accepted)==len(featured)==4000
        with (ROOT/'new_errors.jsonl').open('w',encoding='utf-8') as stream:
            for r in accepted.values():
                exported={key:r[key] for key in ['id','question_id','question_group','dataset','fold','question',
                    'reference','reference_equation','response_sha256','label','math_error_type','human_review_status']}
                exported['response']=r['generation']['text']
                stream.write(json.dumps(exported,ensure_ascii=False)+'\n')
    write_json(ROOT/('FINAL_AUDIT.json' if final else 'PARTIAL_AUDIT.json'),result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--final',action='store_true')
    args=p.parse_args()
    print(json.dumps(audit(args.final)))
