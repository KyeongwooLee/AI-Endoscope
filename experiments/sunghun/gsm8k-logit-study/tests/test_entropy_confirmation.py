import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('confirmation', Path(__file__).resolve().parents[1] / 'scripts/confirm_entropy_features.py')
# The standalone script is normally placed beside its existing training helpers.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ConfirmationTests(unittest.TestCase):
    def test_candidate_definition(self):
        seven = np.arange(33, dtype=float) + 4
        ten = np.arange(33, dtype=float) + 5
        np.testing.assert_allclose(module.candidate_values(seven, ten), [21, 20, 16])

    def test_short_prefix_is_not_silently_used(self):
        with self.assertRaises(ValueError):
            module.candidate_values([1.] * 32, [1.] * 33)

    def test_quality_failure_blocks_resume(self):
        rows = [{'id': str(i)} for i in range(8)]
        ledger = {row['id']: {'quality': {'positions': 33, 'cached_argmax_matches': 33, 'replay_argmax_matches': 30}}
                  for row in rows}
        with self.assertRaises(AssertionError):
            module.check_gate(ledger, rows, 'test')

    def test_quality_pass_commits_fingerprint(self):
        rows = [{'id': str(i)} for i in range(8)]
        ledger = {row['id']: {'quality': {'positions': 33, 'cached_argmax_matches': 33, 'replay_argmax_matches': 33}}
                  for row in rows}
        scratch = (Path(__file__).resolve().parents[1] / '.test-tmp/confirmation').resolve()
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            old = module.ROOT
            try:
                module.ROOT = Path(temp)
                module.check_gate(ledger, rows, 'fixed')
                self.assertTrue((Path(temp) / 'quality_gate.json').exists())
            finally:
                module.ROOT = old


if __name__ == '__main__':
    unittest.main()
