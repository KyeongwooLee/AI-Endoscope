import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('ablation', Path(__file__).resolve().parents[1] / 'scripts/prepare_feature_ablation.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FeatureAblationTests(unittest.TestCase):
    def test_no_later_layer_information(self):
        rng = np.random.default_rng(42)
        states = rng.normal(size=(4, 16))
        curve = rng.uniform(.1, 1, size=(28, 5))
        before = module.state_features(states, curve)
        curve[14:] = 99999
        self.assertEqual(before, module.state_features(states, curve))

    def test_scale_and_direction(self):
        states = np.tile([1., 0.], (4, 1))
        states[3] = [2., 0.]
        result = module.state_features(states, np.ones((14, 5)))
        self.assertAlmostEqual(result['adjacent']['delta_norm'], 1)
        self.assertAlmostEqual(result['adjacent']['relative_delta'], 1)
        self.assertAlmostEqual(result['adjacent']['cosine_distance'], 0)

    def test_readouts_exclude_future_and_later_layers(self):
        frame = pd.DataFrame([{'layer': layer, 'token_index': token, 'entropy': token + .1, 'margin': .2}
            for layer in (7, 10, 14, 28) for token in range(40)])
        before = module.readout_features(frame, 32)
        frame.loc[(frame.token_index > 32) | (frame.layer > 14), ['entropy', 'margin']] = 9999
        self.assertEqual(before, module.readout_features(frame, 32))
        self.assertAlmostEqual(before['L14_entropy_last8_slope'], 1)


if __name__ == '__main__':
    unittest.main()
