#!/usr/bin/env python3
"""Independent artifact audit; does not change experimental choices."""
import hashlib,json,collections
from pathlib import Path
import numpy as np
from common import answer,correct,recalculate
from pipeline import RUN,OLD,read,dump

def audit(pilot_only=False):
    manifest=read(RUN/'manifest.json');expected=[r for r in manifest if not pilot_only or r['split']=='pilot'];checks=[]
    hashes={r['question_sha256'] for r in expected};assert len(hashes)==len(expected)
    assert all(r['source_partition']==('test' if r['split']=='test' else 'train') for r in manifest)
    bysplit=collections.defaultdict(list)
    for m in expected:
        r=read(RUN/'records'/f"{m['id']}.json");assert r['split']==m['split'];bysplit[r['split']].append(r)
        pred,status=answer(r['original']['decoded_output'],r['original']['truncated']);assert pred==r['actions']['Keep']['answer']
        assert r['label_error']==int(not correct(pred,m['gold']));assert r['actions']['Keep']['success']==1-r['label_error']
        calc=recalculate(r['original']['decoded_output'],r['original']['truncated']);assert calc['available']==r['actions']['Recalculate']['available'];assert calc['answer']==r['actions']['Recalculate']['answer']
        for action in ['Rebuild','Review']:
            a=r['actions'][action];assert answer(a['decoded_output'],a['truncated'])[0]==a['answer'];assert a['success']==int(correct(a['answer'],m['gold']))
        if m['reuse_original']:
            old=read(m['reuse_original'])
            for k in ['prompt_ids','generated_ids','decoded_output']: assert old[k]==r['original'][k],(m['id'],k)
        v=r['verification'];p=v['positions'];assert p['answer']<p['boundary']<p['preverdict']==len(v['prompt_ids'])-1
        for k,i in p.items(): assert v['position_tokens'][k]['token_id']==v['prompt_ids'][i]
        h=np.load(RUN/'hidden'/f"{m['id']}.npz");assert len(h.files)==12;assert all(h[k].shape==(1536,) and np.isfinite(h[k]).all() for k in h.files)
        checks.append(m['id'])
    summary={}
    for s,rows in bysplit.items():
        summary[s]=dict(n=len(rows),errors=sum(r['label_error'] for r in rows),strict_errors=sum(r['strict_error'] for r in rows),parse_counts=dict(collections.Counter(r['parse_status'] for r in rows)),verdict_counts=dict(collections.Counter(r['verification']['verdict'] for r in rows)),recalculate_available=sum(r['actions']['Recalculate']['available'] for r in rows),original_truncated=sum(r['original']['truncated'] for r in rows),repair_truncated={a:sum(r['actions'][a]['truncated'] for r in rows) for a in ['Rebuild','Review']},action_successes={a:sum(r['actions'][a]['success'] or 0 for r in rows) for a in ['Keep','Recalculate','Rebuild','Review']})
    result=dict(passed=True,checked_records=len(checks),split_summary=summary,checks=['official partition isolation','exact question grouping','original token IDs/text preserved','Keep=1-error','calculator extraction availability and outcome','all action scoring','12 finite 1536-dimensional vectors per example','pre-verdict positions belong only to prompt IDs'])
    dump(RUN/('pilot_integrity.json' if pilot_only else 'integrity_audit.json'),result);print(json.dumps(result,indent=2))
if __name__=='__main__':
    import sys
    audit('--pilot' in sys.argv)
