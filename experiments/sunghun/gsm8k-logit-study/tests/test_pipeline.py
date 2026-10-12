import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from gsm8k_lens import pipeline
from gsm8k_lens.analysis import analyze
from gsm8k_lens.engine import replay
from gsm8k_lens.storage import read_json
from test_core import CharacterTokenizer, tiny_config, tiny_model, temporary_directory


class PipelineTests(unittest.TestCase):
    def test_offline_end_to_end_resume_cache_and_gates(self):
        cfg, model, tokenizer = tiny_config(), tiny_model(), CharacterTokenizer()
        cfg["extraction"]["cache_mode"] = "full"
        dataset = {
            "train": [{"question": f"synthetic {i}", "answer": "#### 4"} for i in range(6)],
            "test": [{"question": f"synthetic {i}", "answer": "#### 4"} for i in range(2)],
        }
        api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="model-sha"),
                              dataset_info=lambda *a, **kw: SimpleNamespace(sha="dataset-sha"))

        def fake_generation(_model, _tokenizer, question, _cfg, quality):
            text = "2+2=4.\n#### " + ("4" if int(question.rsplit(" ", 1)[1]) % 2 == 0 else "5")
            prompt, ids = [2, 65], [ord(c) for c in text] + [0]
            record = {"prompt_token_ids": prompt, "generated_token_ids": ids, "text": text,
                      "generated_tokens": len(ids), "prompt_tokens": len(prompt),
                      "stop_reason": "eos", "truncated": False}
            if quality:
                _, native, _ = replay(model, prompt, ids)
                with torch.inference_mode():
                    logp = model.lm_head(native).float().log_softmax(-1)
                record["generation_quality"] = {"top1": logp.argmax(-1).tolist(),
                    "generated_logprob": logp.gather(-1, torch.tensor(ids)[:, None]).squeeze(-1).tolist()}
            return record

        requested = os.environ.get("GSM8K_TEST_OUTPUT")
        with temporary_directory() as temporary:
            run = Path(requested) if requested else Path(temporary)
            with patch("datasets.load_dataset", return_value=dataset), \
                 patch("huggingface_hub.HfApi", return_value=api), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
                 patch("gsm8k_lens.engine.load_model", return_value=(model, tokenizer)), \
                 patch("gsm8k_lens.engine.generate_example", side_effect=fake_generation) as generating:
                pipeline.prepare(cfg, run)
                pipeline.prepare(cfg, run)
                with self.assertRaises(FileNotFoundError):
                    pipeline.generate(cfg, run, "test")
                pipeline.generate(cfg, run, "pilot", limit=2)
                pipeline.label(cfg, run, "pilot", limit=2)
                pipeline.extract(cfg, run, "pilot", limit=2)
                with self.assertRaises(ValueError):
                    pipeline.freeze(cfg, run)
                partial = analyze(cfg, run, "pilot", allow_partial=True)
                self.assertTrue(partial["partial"])
                pipeline.generate(cfg, run, "pilot")
                calls = generating.call_count
                pipeline.generate(cfg, run, "pilot")
                self.assertEqual(generating.call_count, calls)
                pipeline.label(cfg, run, "pilot")
                pipeline.extract(cfg, run, "pilot")
                with patch("gsm8k_lens.engine.replay", side_effect=AssertionError("Cache was not reused")):
                    pipeline.extract(cfg, run, "pilot", recompute=True)
                pipeline.freeze(cfg, run)
                summary = analyze(cfg, run, "pilot")
                self.assertFalse(summary["partial"])
                self.assertGreater(summary["n_correct"], 0)
                self.assertGreater(summary["n_incorrect"], 0)
                for phase in ("exploration", "test"):
                    pipeline.generate(cfg, run, phase)
                    pipeline.label(cfg, run, phase)
                    pipeline.extract(cfg, run, phase)
                    summary = analyze(cfg, run, phase)
                    self.assertEqual(summary["completed"], 2)
                    if phase == "exploration":
                        pipeline.freeze(cfg, run, analysis=True)
                self.assertTrue((run / "analysis/test/report.md").exists())
                self.assertTrue((run / "analysis/test/complete.json").exists())
                self.assertIsNotNone(read_json(run / "analysis_lock.json")["source_hash"])


if __name__ == "__main__":
    unittest.main()
