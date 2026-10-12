import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location('train_all', PROJECT / 'scripts/extract_train_all.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrainAllTests(unittest.TestCase):
    def test_only_official_train_is_selected(self):
        scratch = PROJECT / '.test-tmp/train-all-selection'
        scratch.mkdir(parents=True, exist_ok=True)
        fake = [{'question': str(i), 'answer': '#### 1'} for i in range(7473)]
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            config = {'dataset': {'id': 'openai/gsm8k', 'name': 'main', 'revision': 'fixed'}}
            with patch.object(module, 'load_dataset', return_value=fake) as load:
                rows = module.select_train(Path(temp), config, {})
            load.assert_called_once_with('openai/gsm8k', 'main', revision='fixed', split='train')
            self.assertEqual(len({r['id'] for r in rows}), 7473)
            self.assertTrue(all(r['id'].startswith('train-') for r in rows))

    def test_checkpoint_boundary(self):
        self.assertEqual(module.checkpoint_times(32), [0])
        self.assertEqual(module.checkpoint_times(33), [0, 32])
        self.assertEqual(module.checkpoint_times(129), [0, 32, 64, 128])
        with self.assertRaises(ValueError):
            module.checkpoint_times(0)

    def test_residual_metrics(self):
        states = torch.tensor([[[1., 0.]], [[2., 0.]], [[2., 2.]]])
        curve = module.residual_metrics(states)
        np.testing.assert_allclose(curve[0, 0].numpy(), [1, 2, 1, 1, 0])
        self.assertEqual(tuple(curve.shape), (2, 1, 5))

    def test_gate_requires_agreement_and_causality(self):
        rows = [{'id': 'train-00000', 'origin': 'new_train',
                 'quality': {'positions': 4, 'matches': 3, 'causal_max_abs_error': 0.}}]
        with self.assertRaises(ValueError):
            module.gate(rows)
        rows[0]['quality']['matches'] = 4
        self.assertTrue(module.gate(rows)['passed'])
        rows[0]['quality']['causal_max_abs_error'] = None
        with self.assertRaises(ValueError):
            module.gate(rows)

    def test_batch_roundtrip_and_ack_resume(self):
        scratch = PROJECT / '.test-tmp/train-all'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            for name in ('pending', 'outbox', 'acks'):
                (root / name).mkdir()
            paths = []
            for index, length in enumerate((33, 65)):
                identifier = f'train-{index:05d}'
                times = module.checkpoint_times(length)
                record = {'id': identifier, 'fingerprint': 'frozen',
                          'summary': {'id': identifier, 'incorrect': index}}
                path = root / 'pending' / (identifier + '.npz')
                np.savez_compressed(path, record=np.asarray(json.dumps(record)),
                    checkpoints=np.asarray(times), hidden=np.ones((len(times), 4, 1536)),
                    trajectory=np.ones((len(times), 28, 5)), readouts=np.ones((length, 3, 3)))
                paths.append(path)
            receipt = module.pack_batch(root, 0, paths, 'frozen')
            ids = ['train-00000', 'train-00001']
            module.validate_committed(root, 'batch_0000', receipt, 'frozen', ids)
            archive = root / 'outbox/batch_0000.zip'
            with zipfile.ZipFile(archive) as packed:
                with np.load(io.BytesIO(packed.read('batch_0000/features.npz'))) as features:
                    np.testing.assert_array_equal(features['readout_offsets'], [0, 33, 98])
                    self.assertEqual(features['hidden'].shape, (5, 4, 1536))
            archive.unlink()
            with self.assertRaises(ValueError):
                module.validate_committed(root, 'batch_0000', receipt, 'frozen', ids)
            module.write_json(root / 'acks/batch_0000.json', {
                'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes']})
            module.validate_committed(root, 'batch_0000', receipt, 'frozen', ids)
            with self.assertRaises(ValueError):
                module.validate_committed(root, 'batch_0000', receipt, 'changed', ids)


if __name__ == '__main__':
    unittest.main()
