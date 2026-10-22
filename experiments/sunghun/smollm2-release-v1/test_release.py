import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np

from build_release import COMPACT_KEYS, combine_examples, compact_arrays, make_schema
from load_data import SCALAR_NAMES, feature_matrix, iter_compact, iter_full, scalar_features, verify_checksums
from publish_release import verified_assets


def synthetic(length):
    times = np.array([t for t in [0, 32, 64, 128] if t < length], np.int16)
    return dict(checkpoints=times,
        hidden=np.arange(len(times) * 25 * 2048, dtype=np.float32).reshape(len(times), 25, 2048),
        trajectory=np.arange(length * 24 * 5, dtype=np.float32).reshape(length, 24, 5),
        prefix_readouts=np.arange(min(length, 129) * 6 * 5, dtype=np.float32).reshape(min(length, 129), 6, 5))


class ReleaseTests(unittest.TestCase):
    def test_exact_subsets(self):
        source = synthetic(65)
        out = compact_arrays(source)
        np.testing.assert_array_equal(out["hidden"], source["hidden"][:, [6, 9, 11, 12]])
        np.testing.assert_array_equal(out["trajectory"], source["trajectory"][[0, 32, 64]])
        np.testing.assert_array_equal(out["readouts"], source["prefix_readouts"][:, [0, 1, 3]][:, :, [1, 2, 3]])
        self.assertEqual(out["hidden"].dtype, np.float32)
        source["hidden"][:] = 0
        self.assertNotEqual(float(out["hidden"].sum()), 0)

    def test_ragged_loader_and_availability(self):
        examples = []
        for name, length in [("short", 32), ("long", 33)]:
            record = dict(id=name, label=dict(answer_correct=name == "short", analysis_eligible=True))
            examples.append((record, compact_arrays(synthetic(length))))
        binary, lines = combine_examples(examples)
        with np.load(io.BytesIO(binary), allow_pickle=False) as z:
            self.assertEqual(set(z.files), COMPACT_KEYS)
            np.testing.assert_array_equal(z["readout_offsets"], [0, 32, 65])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch_0000"
            batch.mkdir()
            (batch / "features.npz").write_bytes(binary)
            (batch / "generations.jsonl").write_bytes(lines)
            self.assertEqual(len(list(iter_compact(root, 0))), 2)
            rows = list(iter_compact(root, 32))
            self.assertEqual([r["id"] for r in rows], ["long"])
            self.assertEqual(rows[0]["trajectory_early"].shape, (12, 5))
            for kind, width in [("middle", 2048), ("middle_delta", 4096), ("multi", 6144), ("scalar", 36)]:
                ids, x, y = feature_matrix(root, 32, kind)
                self.assertEqual(x.shape, (1, width))
                self.assertEqual(y.tolist(), [1])
                self.assertEqual(ids.tolist(), ["long"])

    def test_scalars_match_qwen_definition(self):
        old = runpy.run_path(str(Path(__file__).resolve().parent.parent / "data_tools/load_train_all.py"))
        value = np.random.default_rng(42).normal(size=(33, 3, 3)).astype(np.float32)
        np.testing.assert_array_equal(scalar_features(value), old["scalar_features"](value))
        self.assertEqual(len(SCALAR_NAMES), 36)
        self.assertEqual(SCALAR_NAMES[0], "l6.entropy.current")
        self.assertEqual(SCALAR_NAMES[-1], "l12.top1_probability.change_from_t0")

    def test_full_loader_preserves_arrays(self):
        record = dict(id="example", dataset="gsm8k")
        value = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        stream = io.BytesIO()
        np.savez_compressed(stream, record=np.asarray(json.dumps(record)), trajectory=value)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batches").mkdir()
            with zipfile.ZipFile(root / "batches/batch_0000.zip", "w") as z:
                z.writestr("features/example.npz", stream.getvalue())
            rows = list(iter_full(root))
            self.assertEqual(rows[0][0], record)
            np.testing.assert_array_equal(rows[0][1]["trajectory"], value)
            self.assertEqual(list(iter_full(root, "svamp")), [])

    def test_checksum_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checksums.sha256").write_text("0" * 64 + "  ../outside.txt\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_checksums(root)

    def test_schema_no_old_shape_or_layer_names(self):
        old = json.loads((Path(__file__).resolve().parent.parent / "DATA_SCHEMA.json").read_text(encoding="utf-8"))
        new = make_schema(old)
        self.assertEqual(new["npz"]["hidden"]["shape"], ["C", 4, 2048])
        self.assertEqual(new["npz"]["trajectory"]["shape"], ["C", 24, 5])
        self.assertEqual(new["primary_layer"], 12)
        self.assertEqual(old["primary_layer"] if "primary_layer" in old else old["raw_layers"][-1], 14)
        self.assertEqual(old["dimensions"]["D"], 1536)

    def test_publication_rejects_partial_before_any_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "READY_TO_PUBLISH.json").write_text(json.dumps(dict(complete=False, examples=1744, full_original_batches=109)))
            with patch("publish_release.subprocess.run", side_effect=AssertionError("Must not use network")):
                with self.assertRaisesRegex(ValueError, "Incomplete release"):
                    verified_assets(root)

    def test_recovery_requires_completed_snapshot(self):
        from build_release import recovery_inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                recovery_inputs(root, {}, [], root / "incomplete-recovery")

    @unittest.skipUnless(os.environ.get("SMOLL_BACKUP"), "Set SMOLL_BACKUP for original batch integration tests")
    def test_real_original_batch(self):
        source = Path(os.environ["SMOLL_BACKUP"])
        with zipfile.ZipFile(source / "incoming/batch_0000.zip") as z:
            name = next(n for n in z.namelist() if n.startswith("features/") and n.endswith(".npz"))
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as arrays:
                result = compact_arrays(arrays)
                for t_index in range(len(result["checkpoints"])):
                    for layer_index, layer in enumerate([6, 9, 11, 12]):
                        np.testing.assert_array_equal(result["hidden"][t_index, layer_index], arrays["hidden"][t_index, layer])
                record = json.loads(str(arrays["record"]))
        data, lines = combine_examples([(record, result)])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "batch_0000").mkdir()
            (root / "batch_0000/features.npz").write_bytes(data)
            (root / "batch_0000/generations.jsonl").write_bytes(lines)
            x = next(iter_compact(root, 0))
            np.testing.assert_array_equal(x["hidden"], result["hidden"][0])

    @unittest.skipUnless(os.environ.get("SMOLL_BACKUP"), "Set SMOLL_BACKUP for missing-source guard")
    def test_missing_source_never_creates_release(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "must-not-exist"
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("build_release.py")),
                os.environ["SMOLL_BACKUP"], str(target)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source batches not backed up", result.stderr)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
