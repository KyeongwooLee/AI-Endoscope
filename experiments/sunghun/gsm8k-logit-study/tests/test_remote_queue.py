import importlib.util
from pathlib import Path
import unittest

from gsm8k_lens.storage import source_digest, write_json
from test_core import temporary_directory

spec = importlib.util.spec_from_file_location('remote_queue', Path(__file__).resolve().parents[1] / 'scripts/remote_queue.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class QueueTests(unittest.TestCase):
    def test_ready_bundles_do_not_block_production(self):
        with temporary_directory() as run:
            records = [{'phase': 'test', 'state': 'ready', 'sample_ids': ['a', 'b']}]
            self.assertEqual(module.next_batch(run, records, ['a', 'b', 'c', 'd'], 'test', 1), ['c'])

    def test_incomplete_active_batch_is_resumed(self):
        with temporary_directory() as run:
            write_json(run / 'offload/active.json', {'phase': 'test', 'sample_ids': ['a', 'b'],
                                                    'source_hash': source_digest()})
            self.assertEqual(module.next_batch(run, [], ['a', 'b', 'c'], 'test', 1), ['a', 'b'])
            records = [{'phase': 'test', 'state': 'ready', 'sample_ids': ['a', 'b']}]
            self.assertEqual(module.next_batch(run, records, ['a', 'b', 'c'], 'test', 1), ['c'])

    def test_changed_source_fails_closed(self):
        with temporary_directory() as run:
            write_json(run / 'offload/active.json', {'phase': 'test', 'sample_ids': ['a'], 'source_hash': 'wrong'})
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                module.next_batch(run, [], ['a'], 'test', 1)
