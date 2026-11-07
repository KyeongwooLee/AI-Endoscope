"""Synthetic, offline CPU tests. No downloads or pretrained model loading."""
import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sae_error_clustering.common import atomic_json, atomic_npz, digest
from sae_error_clustering.prepare import assign_splits, join_records
from sae_error_clustering.cluster import select_k, transform_features
from sae_error_clustering.extract import aggregate, decode, encode, hidden_chunks, read_shard, reasoning_tokens
from sae_error_clustering.report import choose_cases
from sae_error_clustering.tokens import byte_level_offsets


class DataTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(id='a', split='train', pid=4, sample=1, mode='sampled', finish='stop',
                     response='2+2=5\n#### 5', rule_type='ARITH')]
        tokens = [dict(id='a', prompt_token_ids=[1], output_token_ids=[2, 3])]
        return rows, tokens

    def test_id_join_and_label_isolation(self):
        rows, tokens = self.fixture()
        first = join_records(rows, tokens, ['a'])
        rows[0]['rule_type'] = 'NOEQ'
        rows[0]['gold'] = 'arbitrary privileged reference'
        self.assertEqual(first, join_records(rows, tokens, ['a']))
        with self.assertRaises(ValueError):
            join_records(rows, tokens * 2, ['a'])
        with self.assertRaises(ValueError):
            join_records(rows, [], ['a'])
        with self.assertRaises(ValueError):
            join_records(rows, tokens, ['wrong-order'])

    def test_grouping_and_duplicate_tracking(self):
        rows = [dict(id=str(i), pid=i//3, response=str(i % 2)) for i in range(60)]
        result = assign_splits(rows, 42, .2)
        for pid in range(20):
            self.assertEqual(len({r['partition'] for r in result if r['pid'] == pid}), 1)
        self.assertEqual(result[0]['canonical_id'], result[2]['canonical_id'])

    def test_confirmation_never_changes_fitted_pca(self):
        x = np.random.default_rng(42).uniform(size=(20, 8)).astype('float32')
        discovery = np.arange(15)
        before, a = transform_features(x, discovery, 4, True)
        x[15:] *= 1000
        after, b = transform_features(x, discovery, 4, True)
        np.testing.assert_allclose(a['pca'].components_, b['pca'].components_)
        np.testing.assert_allclose(before[:15], after[:15])

    def test_candidate_tie_prefers_smaller_k(self):
        self.assertEqual(select_k([{'k': 3, 'mean_silhouette': .3},
                                   {'k': 2, 'mean_silhouette': .3}]), 2)
        with self.assertRaises(ValueError):
            select_k([{'k': 2, 'mean_silhouette': None}])

    def test_review_uses_unique_problems(self):
        rows = [dict(pid=i//2) for i in range(40)]
        distances = np.random.default_rng(1).random((40, 2))
        chosen = choose_cases(list(range(40)), rows, distances, 0)
        self.assertEqual(len(chosen), 15)
        self.assertEqual(len({rows[i]['pid'] for _, i in chosen}), 15)

    def test_shard_integrity_and_resume_signature(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'shard.npz'
            atomic_npz(p, ids=np.array(['a']), sae_mean=np.zeros((1, 24576), 'float32'),
                       hidden_mean=np.zeros((1, 1536), 'float32'), peaks=np.full((1, 24576), -1, 'int32'))
            atomic_json(p.with_suffix('.json'), dict(signature='fixed', sha256=digest(p), rows=[dict(id='a')]))
            read_shard(p, ['a'], 'fixed')
            with self.assertRaises(ValueError):
                read_shard(p, ['a'], 'changed')
            with self.assertRaises(ValueError):
                read_shard(p, ['other'], 'fixed')
            with p.open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaises(ValueError):
                read_shard(p, ['a'], 'fixed')

    def test_token_mask_excludes_boundary_special_and_final(self):
        row = dict(output_token_ids=[1, 2, 3, 4], response='text\n#### 3', finish='stop')
        regions = [dict(token_index=i, char_start=0, char_end=4, region=region,
                        included=included, boundary_overlap=overlap)
                   for i, (region, included, overlap) in enumerate([
                       ('reasoning', True, False), ('reasoning', True, True),
                       ('excluded', False, False), ('final_number', True, False)])]
        with patch('sae_error_clustering.extract.token_regions', return_value=(row['response'], regions)):
            selected, _ = reasoning_tokens(None, row)
            self.assertEqual(selected, [0])
        with patch('sae_error_clustering.extract.token_regions', return_value=('changed', regions)):
            with self.assertRaises(ValueError):
                reasoning_tokens(None, row)

    def test_byte_offsets_when_unicode_character_spans_tokens(self):
        decoder = SimpleNamespace(__getstate__=lambda: b'{"type":"ByteLevel"}')
        tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=decoder),
                                    convert_ids_to_tokens=lambda ids: ['\u00e2', '\u0124', '\u00ac'])
        self.assertEqual(byte_level_offsets(tokenizer, [1, 2, 3], '\u20ac'), [(0, 1)] * 3)
        with self.assertRaises(ValueError):
            byte_level_offsets(tokenizer, [1, 2, 3], 'different')


class TensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.torch = importlib.import_module('torch')

    def test_sae_bias_and_streaming_pooling(self):
        t = self.torch
        weights = dict(W_enc=t.eye(2), W_dec=t.eye(2), b_dec=t.tensor([1., 2.]), b_enc=t.tensor([-1., 1.]))
        t.testing.assert_close(encode(t.tensor([[3., 5.]]), weights), t.tensor([[1., 4.]]))
        t.testing.assert_close(decode(t.tensor([[1., 4.]]), weights), t.tensor([[2., 6.]]))
        h = t.arange(10, dtype=t.float32).reshape(5, 2)
        a = aggregate([(0, h)], [0, 2, 4], 2, 2, weights)
        b = aggregate([(0, h[:2]), (2, h[2:4]), (4, h[4:])], [0, 2, 4], 2, 2, weights)
        for left, right in zip(a[:3], b[:3]):
            np.testing.assert_allclose(left, right)
        self.assertEqual(a[3], b[3])

    def test_cached_chunks_match_full_causal_forward(self):
        from transformers import Qwen2Config, Qwen2Model
        t = self.torch
        t.manual_seed(42)
        model = Qwen2Model(Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            attention_dropout=0.)).eval()
        prompt, output = [1, 2, 3], [4, 5, 6, 7, 8]
        captured = {}
        def hook(_m, _a, value):
            captured['h'] = value[0] if isinstance(value, tuple) else value
        with t.inference_mode():
            handle = model.layers[0].register_forward_hook(hook)
            model(input_ids=t.tensor([prompt + output]), use_cache=False)
            reference = captured['h'][0, len(prompt):].clone()
            handle.remove()
            count = len(model.layers[0]._forward_hooks)
            actual = t.cat([h.clone() for _, h in hidden_chunks(model, 0, prompt, output, 2, 'cpu')])
            t.testing.assert_close(reference, actual, rtol=1e-4, atol=1e-5)
            self.assertEqual(len(model.layers[0]._forward_hooks), count)


if __name__ == '__main__':
    unittest.main()
