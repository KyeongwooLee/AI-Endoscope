import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from common import write_json
from control import acknowledge
from extract import extract, js_divergence, residual_metrics
from run import commit, quality_gate


class Tokenizer:
    all_special_ids = [0, 1, 2]
    pieces = {3: "2", 4: "+", 6: "\n", 7: "####", 8: " ", 9: "4", 11: "5"}

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids if i not in self.all_special_ids)

    def __call__(self, text, **kwargs):
        ids, offsets, start = [], [], 0
        while start < len(text):
            matches = [(i, piece) for i, piece in self.pieces.items() if text.startswith(piece, start)]
            i, piece = max(matches, key=lambda item: len(item[1]))
            ids.append(i)
            offsets.append((start, start + len(piece)))
            start += len(piece)
        return {"input_ids": ids, "offset_mapping": offsets}


class DetailedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cls.model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=4, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=128, pad_token_id=2)).eval()
        cls.config = {"extraction": {"checkpoints": [0, 2, 4, 8], "readout_prefix_tokens": 5,
                     "readout_layers": [1, 2, 4], "projection_chunk_tokens": 3, "top_k": 3, "primary_layer": 2}}
        cls.tokenizer = Tokenizer()
        ids = [3, 4, 3, 6, 7, 8, 9, 6, 2]
        cls.generation = {"prompt_token_ids": [20, 21, 22], "generated_token_ids": ids,
                          "text": cls.tokenizer.decode(ids)}

    def test_shapes_native_readout_and_masks(self):
        f, qa, names = extract(self.model, self.tokenizer, self.generation, self.config, quality=True)
        self.assertEqual(f["hidden"].shape, (4, 5, 16))
        self.assertEqual(f["pooled_hidden"].shape, (4, 3, 16))
        self.assertEqual(f["trajectory"].shape, (9, 4, 5))
        self.assertEqual(f["prefix_readouts"].shape, (5, 3, 5))
        self.assertEqual(f["checkpoint_readouts"].shape, (4, 4, 7))
        self.assertEqual(f["checkpoint_top_ids"].shape, (4, 4, 3))
        self.assertEqual(f["token_readouts"].shape, (9, 5))
        self.assertEqual(f["token_masks"][:, 2].sum(), 1)
        self.assertEqual(len(names), 26)
        self.assertEqual(qa["independent_prefix_checks"], 4)
        np.testing.assert_allclose(f["prefix_readouts"][:, -1], f["token_readouts"][:5], atol=1e-6)
        np.testing.assert_allclose(f["checkpoint_readouts"][:, -1, 5], 0, atol=1e-6)
        self.assertEqual(f["token_offsets"][-1].tolist(), [-1, -1])

    def test_future_token_invariance(self):
        a, _, _ = extract(self.model, self.tokenizer, self.generation, self.config)
        changed = dict(self.generation)
        changed["generated_token_ids"] = [3, 4, 3, 6, 7, 8, 11, 6, 2]
        changed["text"] = self.tokenizer.decode(changed["generated_token_ids"])
        b, _, _ = extract(self.model, self.tokenizer, changed, self.config)
        np.testing.assert_allclose(a["hidden"][:3], b["hidden"][:3], atol=1e-6)
        np.testing.assert_allclose(a["token_readouts"][:6], b["token_readouts"][:6], atol=1e-6)

    def test_zero_pools_and_nan_confidence_without_final_number(self):
        changed = dict(self.generation, generated_token_ids=[3, 4, 3, 2], text="2+2")
        f, _, _ = extract(self.model, self.tokenizer, changed, self.config)
        self.assertTrue(np.isnan(f["confidence"][13:]).all())
        self.assertTrue((f["pooled_hidden"][:, 2] == 0).all())

    def test_js_and_trajectory(self):
        a = torch.tensor([[1., 2., 3.]]).log_softmax(-1)
        np.testing.assert_allclose(js_divergence(a, a).numpy(), 0, atol=1e-6)
        metric = residual_metrics(torch.tensor([1., 0.]), torch.tensor([2., 0.]))
        np.testing.assert_allclose(metric.numpy(), [1, 2, 1, 1, 0])

    def test_commit_verify_ack_and_bad_ack(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ["pending", "outbox", "acks"]:
                (root / directory).mkdir()
            row = dict(id="train-00000")
            record = dict(**row, dataset="gsm8k", fingerprint="pinned", label={"answer_correct": True}, quality={},
                          generation={"effective_generation_config": {"do_sample": False}, "generated_tokens": 2})
            np.savez_compressed(root / "pending/train-00000.npz", record=np.asarray(json.dumps(record)), hidden=np.ones((1, 2, 3)))
            receipt = commit(root, "batch_0000", [row], "pinned")
            self.assertFalse((root / "pending/train-00000.npz").exists())
            with self.assertRaises(ValueError):
                acknowledge(root, "batch_0000", "wrong")
            self.assertTrue((root / "outbox/batch_0000.zip").exists())
            acknowledge(root, "batch_0000", receipt["archive_sha256"])
            self.assertFalse((root / "outbox/batch_0000.zip").exists())
            ack = json.loads((root / "acks/batch_0000.json").read_text())
            self.assertTrue(ack["verified_local"])
            with self.assertRaises(ValueError):
                acknowledge(root, "../bad", "wrong")

    def test_gate_is_checked_on_resumed_summaries(self):
        cfg = {"quality": {"pilot_examples": 2, "minimum_greedy_checkpoint_agreement": .99}}
        rows = [dict(do_sample=False, quality={"positions": 4, "matches": 4, "independent_prefix_checks": 1},
                     label={"parse_status": "ok"}) for _ in range(3)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            quality_gate(root, cfg, rows, "pinned")
            self.assertTrue(json.loads((root / "quality_gate.json").read_text())["passed"])
            rows[0]["quality"]["matches"] = 1
            with self.assertRaises(ValueError):
                quality_gate(root, cfg, rows, "pinned")


if __name__ == "__main__":
    unittest.main()
