#!/usr/bin/env python3
"""Export completed results and deterministic illustrative examples, without tuning."""
import csv,json,hashlib,shutil,subprocess,datetime
from pathlib import Path
from pipeline import RUN,ROOT,read,dump
from evaluate import ACTIONS

def main():
    results=read(RUN/'test_results.json');pred=read(RUN/'test_predictions.json')
    grid=read(RUN/'validation_grid.json');position_summary=[]
    for pos in ['answer','boundary','preverdict']:
        for experiment in ['A','B']:
            candidates=[x for x in grid if x['experiment']==experiment and x['key'] and x['key'].startswith(pos+'_')]
            score=(lambda x:x['validation_auroc']) if experiment=='A' else (lambda x:(x['validation_accuracy'],-x['validation_mean_log_loss']))
            position_summary.append(dict(position=pos,**max(candidates,key=score)))
    dump(RUN/'validation_position_summary.json',position_summary)
    for name in ['A','B']:
        metrics=results[name];keys=[k for k,v in next(iter(metrics.values())).items() if not isinstance(v,dict)]
        with (RUN/f'metrics_{name}.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=['method']+keys);w.writeheader()
            for method,row in metrics.items():w.writerow(dict(method=method,**{k:row[k] for k in keys}))
    examples=[];covered=set()
    for i,rid in enumerate(pred['ids']):
        r=read(RUN/'records'/f'{rid}.json');keep=r['actions']['Keep']['success'];ic=pred['B_choices']['internal'][i];ec=pred['B_choices']['external'][i];isuccess=r['actions'][ACTIONS[ic]]['success'];esuccess=r['actions'][ACTIONS[ec]]['success']
        categories=[]
        if keep==0 and isuccess==1:categories.append('internal_successful_repair')
        if keep==1 and isuccess==0:categories.append('internal_harmful_revision')
        if isuccess!=esuccess:categories.append('external_internal_outcome_disagreement')
        if keep==1 and isuccess==1:categories.append('original_correct_preserved')
        if keep==0 and isuccess==0:categories.append('original_error_not_repaired')
        fresh=[k for k in categories if k not in covered]
        if fresh:
            covered.update(fresh);examples.append(dict(id=rid,categories=fresh,question=r['question'],gold=r['gold'],original_response=r['original']['decoded_output'],original_answer=r['actions']['Keep']['answer'],verification=r['verification']['decoded_output'],external_choice=ACTIONS[ec],internal_choice=ACTIONS[ic],action_answers={a:v['answer'] for a,v in r['actions'].items()},action_successes={a:v['success'] for a,v in r['actions'].items()},repair_responses={a:r['actions'][a]['decoded_output'] for a in ['Rebuild','Review']},calculator=r['actions']['Recalculate']))
        if len(covered)==5:break
    dump(RUN/'examples.json',dict(selection_rule='First official-test example in manifest order for each reported category; illustrative only, no tuning.',represented_categories=sorted(covered),examples=examples))
    text=['# Illustrative official-test records','', 'These examples are selected after evaluation for explanation only. Full responses are in examples.json.','']
    for x in examples:
        text.extend([f"## {x['id']}: {', '.join(x['categories'])}",'',x['question'],'',f"Reference: {x['gold']}; original: {x['original_answer']}; verifier: {x['verification']}.",'',f"External chooses {x['external_choice']}; internal chooses {x['internal_choice']}.",'','| Action | Numeric answer | Success |','| --- | --- | --- |'])
        if x['id']=='gsm8k_test_823':
            text.extend(['', '**Reference issue:** The question implies 18, but the official reference mistakenly gives 14. This is an official-reference mismatch, not evidence that the model’s 18 is mathematically wrong. See reference_quality.json.',''])
        for a in ACTIONS:text.append(f"| {a} | {x['action_answers'][a]} | {x['action_successes'][a]} |")
        text.extend(['','Original response:', '```text',x['original_response'],'```',''])
    (RUN/'EXAMPLES.md').write_text('\n'.join(text))
    dump(RUN/'storage.json',dict(workspace_disk=shutil.disk_usage(ROOT)._asdict(),temporary_disk=shutil.disk_usage('/tmp')._asdict(),persistent_run_bytes=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file()),temporary_environment_and_cache=subprocess.getoutput('du -sh /tmp/yaicon-ab'),prior_results_preserved=True))
    print('Exported metrics_A.csv, metrics_B.csv, EXAMPLES.md, examples.json, storage.json')
if __name__=='__main__':main()
