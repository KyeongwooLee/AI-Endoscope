import copy
from contextlib import contextmanager
import math
import os
from pathlib import Path
import shutil
import unittest
import uuid

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from gsm8k_lens.answers import parse_answer, label_answer, token_regions
from gsm8k_lens.config import load_config, manifest_for_run
from gsm8k_lens.engine import (FinalAnswerLineStop, generate_example, make_generation_config,
                              prediction_positions, replay)
from gsm8k_lens.features import (check_quality, distribution_features, js_divergence,
                                 lower_tail, project_features, stabilization_depth,
                                 stable_topk, summarize_example)
from gsm8k_lens.analysis import bootstrap_mean, difference_ci
from gsm8k_lens.storage import commit_files, complete, digest, write_json
from gsm8k_lens.pipeline import record_environment


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def temporary_directory():
    # Avoid restrictive mkdtemp ACLs in sandboxed Windows test environments.
    root = Path(os.environ.get("GSM8K_TEST_TEMP_ROOT", ROOT / ".test-tmp")).resolve()
    folder = root / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        yield folder
    finally:
        if folder.resolve().parent != root:
            raise RuntimeError("Unsafe test cleanup path")
        shutil.rmtree(folder)


class CharacterTokenizer:
    all_special_ids = [0, 1, 2]
    eos_token_id, pad_token_id, bos_token_id = 0, 1, 2
    chat_template = "synthetic-test-only"

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return "".join(chr(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        return [2, 65, 66]


def tiny_config():
    cfg = load_config(ROOT / "configs/rtx3090.json")
    cfg["model"].update(device="cpu", dtype="float32", expected_layers=3,
                        expected_hidden_size=32, expected_vocab_size=128,
                        attention_backend="sdpa")
    cfg["dataset"].update(pilot_size=4, exploration_size=2, expected_train_size=6, expected_test_size=2)
    cfg["generation"].update(max_new_tokens=24, max_prompt_tokens=64, max_sequence_tokens=128)
    cfg["extraction"].update(projection_chunk_tokens=4, minimum_free_disk_gib=0)
    cfg["quality"].update(generation_replay_examples=2, native_lens_max_logprob_error=1e-5,
                          generation_replay_max_logprob_error=1e-5)
    cfg["analysis"].update(primary_layer=2, late_start_layer=2, trajectory_bins=4,
                           bootstrap_replicates=20, minimum_group_size=1)
    return cfg


def tiny_model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         bos_token_id=2, eos_token_id=0, pad_token_id=1,
                         attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(0.5, 1.5, 32))
    return model


class AnswerTests(unittest.TestCase):
    def test_final_answer_stop_waits_for_completed_line(self):
        stop = FinalAnswerLineStop(CharacterTokenizer(), 2)
        for text in ("reasoning\n", "reasoning\n#### 12", "reasoning\n#### 12 apples\n"):
            ids = torch.tensor([[2, 65] + [ord(c) for c in text]])
            self.assertFalse(stop(ids).item())
        self.assertFalse(stop.finished)
        ids = torch.tensor([[2, 65] + [ord(c) for c in "reasoning\n#### 12\n"]])
        self.assertTrue(stop(ids).item())
        self.assertTrue(stop.finished)
    def test_exact_numeric_comparison(self):
        self.assertTrue(label_answer("work\n#### +1,234.00\n", "#### 1234", False)["answer_correct"])
        self.assertTrue(label_answer("#### -.5", "#### -0.50", False)["answer_correct"])
        self.assertFalse(label_answer("#### 3.0001", "#### 3", False)["answer_correct"])

    def test_no_fallback_and_truncation(self):
        for text in ("answer is 3", "#### 3 apples", "#### 1,23", "#### 3\nextra", "#### 3/1", "#### NaN"):
            self.assertNotEqual(parse_answer(text)["status"], "ok")
        self.assertEqual(parse_answer(" \n")["status"], "missing")
        record = label_answer("#### 3", "#### 3", True)
        self.assertTrue(record["answer_correct"])
        self.assertFalse(record["analysis_eligible"])

    def test_regions_and_special_tokens(self):
        text = "2+2=4\n#### 4"
        decoded, rows = token_regions(CharacterTokenizer(), [ord(c) for c in text] + [0])
        self.assertEqual(decoded, text)
        self.assertEqual(rows[-2]["region"], "final_number")
        self.assertFalse(rows[-1]["included"])
        self.assertEqual(sum(r["included"] and r["region"] == "reasoning" for r in rows), 5)
        self.assertEqual(sum(r["region"] == "final_number" for r in rows), 1)

    def test_token_roundtrip_is_verified(self):
        class BadTokenizer(CharacterTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [55], "offset_mapping": [(0, 1)]}
        with self.assertRaisesRegex(ValueError, "roundtrip"):
            token_regions(BadTokenizer(), [52])

    def test_byte_level_noncanonical_generated_tokens(self):
        class Decoder:
            def __getstate__(self):
                return b'{"type":"ByteLevel"}'
        class Backend:
            decoder = Decoder()
        class ByteTokenizer(CharacterTokenizer):
            backend_tokenizer = Backend()
            pieces = {10: '/c', 11: 'ar', 12: '\u010a####\u0120', 13: '24'}
            def decode(self, ids, **kwargs):
                return ''.join(self.pieces[i] for i in ids).replace('\u010a', '\n').replace('\u0120', ' ')
            def convert_ids_to_tokens(self, ids):
                return [self.pieces[i] for i in ids]
            def __call__(self, text, **kwargs):
                return {"input_ids": [99], "offset_mapping": [(0, len(text))]}
        text, rows = token_regions(ByteTokenizer(), [10, 11, 12, 13, 0])
        self.assertEqual(text, '/car\n#### 24')
        self.assertEqual([r['text'] for r in rows], ['/c', 'ar', '\n#### ', '24', ''])
        self.assertEqual(rows[3]['region'], 'final_number')
        self.assertEqual([r['token_id'] for r in rows], [10, 11, 12, 13, 0])

    def test_byte_level_split_unicode_and_invalid_bytes(self):
        class Decoder:
            def __getstate__(self):
                return b'{"type":"ByteLevel"}'
        class Backend:
            decoder = Decoder()
        class ByteTokenizer(CharacterTokenizer):
            backend_tokenizer = Backend()
            def decode(self, ids, **kwargs):
                return '\u00e9'
            def convert_ids_to_tokens(self, ids):
                return ['\u00c3', '\u00a9'][:len(ids)]
            def __call__(self, text, **kwargs):
                return {"input_ids": [99], "offset_mapping": [(0, 1)]}
        _, rows = token_regions(ByteTokenizer(), [10, 11])
        self.assertEqual([(r['char_start'], r['char_end']) for r in rows], [(0, 1), (0, 1)])
        with self.assertRaisesRegex(ValueError, 'Invalid UTF-8'):
            token_regions(ByteTokenizer(), [10])


class FeatureTests(unittest.TestCase):
    def test_distribution_and_ties(self):
        logits = torch.zeros(2, 6)
        metrics, logp, _, ids, _ = distribution_features(logits, torch.tensor([0, 5]), logits.log_softmax(-1))
        torch.testing.assert_close(metrics["entropy"], torch.full((2,), math.log(6)))
        self.assertTrue(torch.equal(ids[0], torch.arange(5)))
        self.assertEqual(metrics["generated_rank"].tolist(), [1, 1])
        self.assertEqual(metrics["margin"].tolist(), [0, 0])
        torch.testing.assert_close(js_divergence(logp, logp), torch.zeros(2))
        p = torch.tensor([[0., -100.]])
        q = p.flip(-1)
        self.assertAlmostEqual(js_divergence(p, q).item(), math.log(2), places=6)
        torch.testing.assert_close(js_divergence(p, q), js_divergence(q, p))

    def test_stabilization_requires_staying_stable(self):
        top1 = np.array([[1, 4, 1], [2, 4, 3], [1, 4, 3], [1, 4, 3]])
        np.testing.assert_array_equal(stabilization_depth(top1), [3, 1, 2])
        np.testing.assert_array_equal(stabilization_depth(top1, 3), [3, 3, 3])

    def test_lower_tail_and_problem_bootstrap(self):
        self.assertEqual(lower_tail(range(11)), 0.5)
        self.assertTrue(math.isnan(lower_tail([])))
        self.assertEqual(bootstrap_mean([3, 3], 20, 42), (3, 3, 3))
        self.assertEqual(difference_ci([1, 1], [4, 4], 20, 42), {"difference": 3, "low": 3, "high": 3})
        self.assertIsNone(difference_ci([], [1], 20, 42)["difference"])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.cfg = tiny_config()

    def test_prediction_alignment_and_causality(self):
        prompt, generated = [2, 65, 66], [52, 53, 54]
        residual, native, positions = replay(self.model, prompt, generated)
        self.assertEqual(positions.tolist(), [2, 3, 4])
        self.assertEqual(list(residual.shape), [3, 3, 32])
        with torch.inference_mode():
            direct = self.model(torch.tensor([prompt + generated])).logits[0, positions]
            reconstructed = self.model.lm_head(self.model.model.norm(residual[-1]))
            torch.testing.assert_close(direct, reconstructed)
            torch.testing.assert_close(self.model.lm_head(native), reconstructed)
            changed, _, _ = replay(self.model, prompt, [52, 99, 100])
            torch.testing.assert_close(residual[:, :2], changed[:, :2])
        self.assertTrue(all(len(layer._forward_hooks) == 0 for layer in self.model.model.layers))
        with self.assertRaises(ValueError):
            prediction_positions(0, 2)

    def test_greedy_config_ignores_checkpoint_sampling_defaults(self):
        old = self.model.generation_config
        self.model.generation_config = copy.deepcopy(old)
        self.model.generation_config.do_sample = True
        self.model.generation_config.repetition_penalty = 1.1
        try:
            selected = make_generation_config(self.model, CharacterTokenizer(), self.cfg)
            self.assertFalse(selected.do_sample)
            self.assertEqual(selected.repetition_penalty, 1)
            self.assertEqual(selected.num_beams, 1)
        finally:
            self.model.generation_config = old

    def test_quality_gate_rejects_bad_alignment(self):
        quality = {"native_lens_max_logprob_error": 0.0,
                   "replay_generated_logprob": [-1.0], "replay_top1": [3]}
        reference = {"generation_quality": {"generated_logprob": [-5.0], "top1": [4]}}
        self.assertFalse(check_quality(quality, reference, self.cfg)["passed"])

    def test_generation_replay_and_chunk_invariance(self):
        generation = generate_example(self.model, CharacterTokenizer(), "test", self.cfg, True)
        residual, native, _ = replay(self.model, generation["prompt_token_ids"], generation["generated_token_ids"])
        frames = project_features(self.model, residual, native, generation["generated_token_ids"], self.cfg)
        self.assertTrue(check_quality(frames[-1], generation, self.cfg)["passed"])
        other = copy.deepcopy(self.cfg)
        other["extraction"]["projection_chunk_tokens"] = 1
        single = project_features(self.model, residual, native, generation["generated_token_ids"], other)
        np.testing.assert_allclose(frames[0].to_numpy(), single[0].to_numpy(), atol=2e-6, equal_nan=True)
        final = frames[0][frames[0].layer == 3]
        self.assertLess(final.js_final.max(), 1e-6)
        self.assertTrue(final.adjacent_js.isna().all())
        self.assertTrue(final.support_change.isna().all())

    def test_summary_weights_tokens_within_problem_only(self):
        ids = [ord(c) for c in "2+2=4\n#### 4"] + [0]
        residual, native, _ = replay(self.model, [2, 65], ids)
        tokens, _, diagnostics, _ = project_features(self.model, residual, native, ids, self.cfg)
        import pandas as pd
        _, regions = token_regions(CharacterTokenizer(), ids)
        tokens = tokens.merge(pd.DataFrame(regions), on="token_index")
        summary, trajectory = summarize_example(tokens, diagnostics, 4)
        self.assertEqual(len(summary), 9)
        self.assertEqual(summary[summary.region == "reasoning"].n_tokens.tolist(), [5, 5, 5])
        self.assertEqual(len(trajectory), 12)
        tokens["included"] = False
        empty, trajectory = summarize_example(tokens, diagnostics, 4)
        self.assertTrue(empty.empty and trajectory.empty)


class StorageTests(unittest.TestCase):
    def test_runtime_drift_is_rejected(self):
        with temporary_directory() as tmp:
            path = Path(tmp) / "environment.json"
            record_environment(path, {"torch": "a"})
            record_environment(path, {"torch": "a"})
            with self.assertRaisesRegex(ValueError, "environment changed"):
                record_environment(path, {"torch": "b"})

    def test_atomic_completion_and_corruption(self):
        with temporary_directory() as tmp:
            root = Path(tmp)
            self.assertFalse(complete(root, "v1"))
            write_json(root / "part.json", {"value": 1})
            commit_files(root, ["part.json"], "v1")
            self.assertTrue(complete(root, "v1", verify=True))
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                complete(root, "v2")
            write_json(root / "part.json", {"value": 2})
            with self.assertRaisesRegex(ValueError, "corrupt"):
                complete(root, "v1", verify=True)

    def test_config_is_immutable(self):
        with temporary_directory() as tmp:
            cfg = tiny_config()
            write_json(Path(tmp) / "manifest.json", {"config_hash": digest(cfg)})
            manifest_for_run(tmp, cfg)
            cfg["seed"] += 1
            with self.assertRaisesRegex(ValueError, "Config changed"):
                manifest_for_run(tmp, cfg)


if __name__ == "__main__":
    unittest.main()
