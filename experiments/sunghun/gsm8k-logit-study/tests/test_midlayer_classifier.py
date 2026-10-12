import importlib.util
from pathlib import Path
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location('midlayer', Path(__file__).resolve().parents[1] / 'scripts/train_midlayer_classifier.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic():
    readouts = pd.DataFrame([{'id': 'a', 'layer': layer, 'token_index': token,
                             'entropy': float(layer + token), 'margin': .1,
                             'adjacent_js': .01 * layer}
                            for layer in (6, 7, 9, 10, 13, 14) for token in range(3)])
    top = pd.DataFrame([{'id': 'a', 'layer': layer, 'token_index': token,
                        'top1': 42 if layer % 2 else 43}
                       for layer in range(1, 15) for token in range(3)])
    return readouts, top


class MidlayerTests(unittest.TestCase):
    def test_future_tokens_and_layer15_comparison_do_not_change_features(self):
        frame, top = synthetic()
        expected, groups = module.prefix_design(frame, top, 1)
        frame.loc[frame.token_index > 1, ['entropy', 'margin', 'adjacent_js']] = 999.
        frame.loc[frame.layer == 14, 'adjacent_js'] = 999.
        top.loc[top.token_index > 1, 'top1'] = 999
        actual, actual_groups = module.prefix_design(frame, top, 1)
        pd.testing.assert_frame_equal(expected, actual)
        self.assertEqual(groups, actual_groups)
        self.assertNotIn('JS_L14_L15_current', actual.columns)
        self.assertEqual(actual.loc['a', 'JS_L13_L14_current'], .13)

    def test_prompt_only_checkpoint_has_no_answer_history(self):
        frame, top = synthetic()
        result, groups = module.prefix_design(frame, top, 0)
        self.assertEqual(result.loc['a', 'L14_entropy_current'], 14.)
        self.assertEqual(result.loc['a', 'L14_entropy_prefix_std'], 0.)
        self.assertEqual(result.loc['a', 'L14_candidate_prefix_change_rate'], 0.)
        self.assertEqual(len(groups['mid14']), 6)

    def test_safe_reader_has_no_target_dependent_columns(self):
        forbidden = {'token_id', 'region', 'included', 'generated_logprob',
                     'generated_rank', 'js_final', 'top1_agreement', 'support_change'}
        self.assertFalse(set(module.SAFE_COLUMNS) & forbidden)
