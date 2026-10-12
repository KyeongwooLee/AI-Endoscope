import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('classifier', Path(__file__).resolve().parents[1] / 'scripts/train_error_classifier.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ClassifierTests(unittest.TestCase):
    def test_feature_groups_exclude_labels_and_output_from_internal(self):
        rows = []
        fields = ['entropy', 'margin', 'generated_logprob', 'lower_tail_logprob', 'js_final',
                  'adjacent_js', 'top1_agreement', 'log_rank', 'support_change_abs',
                  'stabilization_depth', 'late_stabilization_depth', 'top1_switches']
        for layer in (7, 14, 21, 27, 28):
            for region in ('reasoning', 'final_number', 'final_first_token'):
                rows.append(dict(id='a', layer=layer, region=region, generated_tokens=10,
                                 answer_correct=True, **{f: 1. for f in fields}))
        frame, groups = module.design(pd.DataFrame(rows), ['a', 'missing'])
        self.assertNotIn('answer_correct', frame.columns)
        self.assertTrue(all(not c.startswith('L28_') for c in groups['internal']))
        self.assertTrue(all(c.startswith('L28_') for c in groups['confidence']))
        self.assertEqual(groups['combined'], groups['confidence'] + groups['internal'])
        self.assertTrue(frame.loc['missing'].isna().all())

    def test_perfect_error_ranking(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([.1, .8, .2, .9])
        threshold = module.threshold_for_f1(y, score)
        metrics = module.metrics(y, score, threshold)
        self.assertEqual(metrics['auroc'], 1.)
        self.assertEqual(metrics['pr_auc_ap'], 1.)
        self.assertEqual(metrics['f1'], 1.)
        self.assertEqual(metrics['confusion_tn_fp_fn_tp'], [2, 0, 0, 2])

    def test_paired_bootstrap_same_scores_has_zero_difference(self):
        y = np.array([0, 1, 0, 1])
        score = np.array([.1, .8, .2, .9])
        result = module.paired_bootstrap(y, score, score, replicates=20)
        self.assertEqual(result['delta_auroc_ci95'], [0., 0.])
        self.assertEqual(result['delta_ap_ci95'], [0., 0.])
