#!/usr/bin/env python3
"""Post-evaluation sensitivity for a manually observed reference inconsistency.
Not a complete dataset-label audit; never retunes or changes primary artifacts.
"""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from pipeline import RUN,read,dump
p=read(RUN/'test_predictions.json');rid='gsm8k_test_823'
if rid not in p['ids']:
    dump(RUN/'reference_quality.json',dict(flagged_example_not_in_this_run=True));raise SystemExit()
i=p['ids'].index(rid);mask=np.arange(len(p['ids']))!=i;y=np.array(p['error_labels']);z=np.array(p['action_success_labels']);row=read(RUN/'records'/f'{rid}.json')
report=dict(review_scope='Non-exhaustive qualitative review after all models and primary metrics were frozen.',flagged_id=rid,official_gold='14',question_implied_answer='18',reason='Sasha first game = 14. Julie first game = 14-4 = 10. Sasha second game = 10-6 = 4. Sasha total = 14+4 = 18. The reference solution mistakenly sums Julie first game 10 with Sasha second game 4.',primary_results_unchanged=True,sensitivity_kind='Exclude this one flagged example; retain all trained models, probabilities and choices.',n_after_exclusion=int(mask.sum()),A={},B={})
for name in ['D3','D4']:
 pred=np.array(p['A'][name]);report['A'][name]=dict(AUROC=float(roc_auc_score(y[mask],pred[mask])),AP=float(average_precision_score(y[mask],pred[mask])))
report['A']['D4_minus_D3_AUROC']=report['A']['D4']['AUROC']-report['A']['D3']['AUROC']
for name,choices in p['B_choices'].items():report['B'][name]=float(z[np.arange(len(z)),choices][mask].mean())
dump(RUN/'reference_quality.json',report)
print(report)
