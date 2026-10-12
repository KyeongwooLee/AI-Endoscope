import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1] / 'scripts'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HiddenProbeTests(unittest.TestCase):
    def test_features_do_not_access_later_layers(self):
        module = load('train_hidden_probes')
        h = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
        np.testing.assert_array_equal(module.make_features(h, 'h14'), h[:, 3])
        np.testing.assert_array_equal(module.make_features(h, 'h14_delta13'),
            np.concatenate([h[:, 3], h[:, 3] - h[:, 2]], axis=1))
        np.testing.assert_array_equal(module.make_features(h, 'h7_h10_h14'), h[:, [0, 1, 3]].reshape(2, -1))

    def test_residual_metrics_separate_scale_and_direction(self):
        module = load('extract_hidden_probes')
        states = torch.tensor([[[1., 0.]], [[2., 0.]], [[0., 2.]]])
        result = module.residual_metrics(states).numpy()
        np.testing.assert_allclose(result[0, 0], [1., 2., 1., 1., 0.])
        np.testing.assert_allclose(result[1, 0], [2., 2., np.sqrt(8), np.sqrt(2), 1.], rtol=1e-6)

    def test_model_sizes(self):
        module = load('train_hidden_probes')
        self.assertEqual(sum(p.numel() for p in module.build_model(1536, 'mlp').parameters()), 49217)
        self.assertEqual(sum(p.numel() for p in module.build_model(3072, 'linear').parameters()), 3073)

    def test_inverse_shape_fit(self):
        module = load('analyze_hidden_trajectory')
        mean = 4 / np.arange(1, 29)
        fit = module.fit_shapes(mean, mean)
        self.assertAlmostEqual(fit['power_exponent'], -1)
        self.assertAlmostEqual(fit['test_mean_r_squared']['inverse_1_over_layer'], 1)


if __name__ == '__main__':
    unittest.main()
