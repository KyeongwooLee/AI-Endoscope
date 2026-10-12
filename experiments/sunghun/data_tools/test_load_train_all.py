import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from load_train_all import (feature_matrix, iter_examples, scalar_features,
                            SCALAR_NAMES, sha256, verify_checksums)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        batch = self.root / "batch_0000"
        batch.mkdir()
        records = [{"id": key, "label": {"answer_correct": correct, "analysis_eligible": eligible}}
                   for key, correct, eligible in (("train-00000", True, True),
                                                  ("train-00001", False, False))]
        (batch / "generations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        hidden = np.arange(3 * 4 * 1536, dtype=np.float32).reshape(3, 4, 1536)
        readouts = np.arange(34 * 9, dtype=np.float32).reshape(34, 3, 3)
        np.savez_compressed(batch / "features.npz", ids=np.array(["train-00000"] * 2 + ["train-00001"]),
                            checkpoints=np.array([0, 32, 0]), hidden=hidden,
                            trajectory=np.ones((3, 28, 5), dtype=np.float32),
                            readout_ids=np.array(["train-00000", "train-00001"]),
                            readout_offsets=np.array([0, 33, 34]), readouts=readouts)
        self.hidden = hidden

    def test_prefix_and_missing_checkpoint(self):
        rows = list(iter_examples(self.root, 32))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["readouts_prefix"].shape, (33, 3, 3))
        self.assertEqual(rows[0]["trajectory_early"].shape, (14, 5))
        self.assertEqual(len(list(iter_examples(self.root, 0, True))), 1)
        with self.assertRaises(ValueError):
            list(iter_examples(self.root, 31))

    def test_representations_and_labels(self):
        ids, x, y = feature_matrix(self.root, 0, "h14_delta")
        self.assertEqual(ids.tolist(), ["train-00000", "train-00001"])
        self.assertEqual(y.tolist(), [0, 1])
        self.assertEqual(x.shape, (2, 3072))
        np.testing.assert_array_equal(x[0, :1536], self.hidden[0, 3])
        np.testing.assert_array_equal(x[0, 1536:], self.hidden[0, 3] - self.hidden[0, 2])
        self.assertEqual(feature_matrix(self.root, 32, "multi")[1].shape, (1, 4608))

    def test_scalar_order_and_population_std(self):
        values = np.stack((np.ones((3, 3)), np.full((3, 3), 3.)))
        result = scalar_features(values)
        self.assertEqual(len(SCALAR_NAMES), 36)
        np.testing.assert_array_equal(result[:4], [3., 2., 1., 2.])
        np.testing.assert_array_equal(scalar_features(values[:1])[2::4], 0.)

    def test_integrity_and_path_escape(self):
        path = self.root / "example.txt"
        path.write_text("original", encoding="utf-8")
        sums = self.root / "checksums.sha256"
        sums.write_text(f"{sha256(path)}  example.txt\n", encoding="utf-8")
        self.assertEqual(verify_checksums(self.root), 1)
        path.write_text("modified", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_checksums(self.root)
        sums.write_text(f"{'0' * 64}  ../escape.txt\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_checksums(self.root)


if __name__ == "__main__":
    unittest.main()
