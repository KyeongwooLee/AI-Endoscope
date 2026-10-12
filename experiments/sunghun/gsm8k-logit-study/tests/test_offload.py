import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

from gsm8k_lens import pipeline
from gsm8k_lens.engine import replay
from gsm8k_lens.offload import acknowledge, receive_bundle, run_batch, safe_path
from gsm8k_lens.storage import file_digest, read_json, write_json
from test_core import CharacterTokenizer, tiny_config, tiny_model, temporary_directory


class OffloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = temporary_directory()
        self.root = self.temp.__enter__()
        self.addCleanup(self.temp.__exit__, None, None, None)
        self.remote, self.local = self.root / "remote", self.root / "local"
        self.cfg, self.model, self.tokenizer = tiny_config(), tiny_model(), CharacterTokenizer()
        self.assertEqual(self.cfg["extraction"]["cache_mode"], "features_only")
        dataset = {split: [{"question": str(i), "answer": "#### 4"} for i in range(n)]
                   for split, n in (("train", 6), ("test", 2))}
        api = SimpleNamespace(model_info=lambda *a, **kw: SimpleNamespace(sha="model-sha"),
                              dataset_info=lambda *a, **kw: SimpleNamespace(sha="dataset-sha"))
        for target, value in (("datasets.load_dataset", dataset),
                              ("huggingface_hub.HfApi", api),
                              ("transformers.AutoTokenizer.from_pretrained", self.tokenizer),
                              ("gsm8k_lens.engine.load_model", (self.model, self.tokenizer))):
            mock = patch(target, return_value=value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch("gsm8k_lens.engine.generate_example", side_effect=self.generate)
        self.generating = mock.start()
        self.addCleanup(mock.stop)
        pipeline.prepare(self.cfg, self.remote)

    def generate(self, model, tokenizer, question, cfg, quality):
        text = "2+2=4\n#### " + ("4" if int(question) % 2 == 0 else "5")
        prompt, ids = [2, 65], [ord(c) for c in text] + [0]
        record = {"prompt_token_ids": prompt, "generated_token_ids": ids, "text": text,
                  "generated_tokens": len(ids), "prompt_tokens": len(prompt),
                  "truncated": False, "stop_reason": "eos"}
        if quality:
            _, native, _ = replay(model, prompt, ids)
            with torch.inference_mode():
                logp = model.lm_head(native).float().log_softmax(-1)
            record["generation_quality"] = {"top1": logp.argmax(-1).tolist(),
                "generated_logprob": logp.gather(-1, torch.tensor(ids)[:, None]).squeeze(-1).tolist()}
        return record

    def receive(self, result):
        # Model the SCP copy with an independent local file, not the remote archive itself.
        self.local.mkdir(parents=True, exist_ok=True)
        import shutil
        copied = self.local / (result["batch_id"] + ".zip")
        shutil.copyfile(result["archive"], copied)
        return receive_bundle(copied, result["sha256"], self.local)

    def test_two_batches_no_residual_no_regeneration_and_local_freeze(self):
        first = run_batch(self.cfg, self.remote, "pilot", 2)
        self.assertFalse((self.remote / "activation_cache").exists())
        calls = self.generating.call_count
        self.assertEqual(run_batch(self.cfg, self.remote, "pilot", 2), first)
        self.assertEqual(calls, self.generating.call_count)
        receipt = self.receive(first)
        acknowledge(self.remote, receipt)
        self.assertFalse(Path(first["archive"]).exists())
        self.assertTrue((self.remote / "manifest.json").exists())
        self.assertEqual(acknowledge(self.remote, receipt)["status"], "already_offloaded")
        with self.assertRaisesRegex(ValueError, "offloading"):
            pipeline.generate(self.cfg, self.remote, "pilot")
        with self.assertRaises(ValueError):
            pipeline.freeze(self.cfg, self.local)
        second = run_batch(self.cfg, self.remote, "pilot", 2)
        self.assertNotEqual(first["batch_id"], second["batch_id"])
        acknowledge(self.remote, self.receive(second))
        self.assertEqual(self.generating.call_count, 4)
        self.assertEqual(run_batch(self.cfg, self.remote, "pilot", 2)["status"], "phase_complete")
        self.assertFalse(list((self.remote / "features").rglob("*.parquet")))
        self.assertEqual(len(list((self.local / "features/pilot").glob("*/complete.json"))), 4)
        pipeline.freeze(self.cfg, self.local)
        from gsm8k_lens.analysis import analyze
        self.assertEqual(analyze(self.cfg, self.local, "pilot")["completed"], 4)

    def test_bad_checksum_receipt_and_local_conflict_never_delete_remote(self):
        result = run_batch(self.cfg, self.remote, "pilot", 2)
        with self.assertRaisesRegex(ValueError, "checksum"):
            receive_bundle(result["archive"], "0" * 64, self.local)
        receipt = self.receive(result)
        bad = read_json(receipt)
        bad["sha256"] = "0" * 64
        bad_path = self.root / "bad-receipt.json"
        write_json(bad_path, bad)
        with self.assertRaisesRegex(ValueError, "Invalid receipt"):
            acknowledge(self.remote, bad_path)
        self.assertTrue(Path(result["archive"]).exists())
        write_json(self.local / "manifest.json", {"changed": True})
        with self.assertRaisesRegex(ValueError, "Conflicting local"):
            self.receive(result)
        self.assertTrue(Path(result["archive"]).exists())

    def test_cleanup_interruption_and_unrelated_file_preservation(self):
        result = run_batch(self.cfg, self.remote, "pilot", 2)
        receipt = self.receive(result)
        record_path = self.remote / f"offload/records/{result['batch_id']}.json"
        record = read_json(record_path)
        target = self.remote / record["payload_files"][0]
        unrelated = target.parent / "user-notes.json"
        write_json(unrelated, {"keep": True})
        original_unlink = Path.unlink

        def fail_once(path, *args, **kwargs):
            if path == target:
                raise OSError("simulated disconnect")
            return original_unlink(path, *args, **kwargs)
        with patch.object(Path, "unlink", fail_once):
            with self.assertRaises(OSError):
                acknowledge(self.remote, receipt)
        self.assertEqual(read_json(record_path)["state"], "deleting")
        with self.assertRaisesRegex(ValueError, "Interrupted cleanup"):
            run_batch(self.cfg, self.remote, "pilot", 2)
        acknowledge(self.remote, receipt)
        self.assertTrue(unrelated.exists())
        self.assertEqual(read_json(record_path)["state"], "offloaded")

    def test_member_integrity_and_traversal_are_rejected(self):
        for name in ("../outside", "/tmp/outside", "C:/outside", "features/../../outside", "a\\b"):
            with self.assertRaises(ValueError):
                safe_path(self.local, name)
        result = run_batch(self.cfg, self.remote, "pilot", 2)
        damaged = self.root / "damaged.zip"
        with ZipFile(result["archive"]) as source, ZipFile(damaged, "w") as target:
            for name in source.namelist():
                target.writestr(name, b"changed" if name == "data/gold.json" else source.read(name))
        with self.assertRaisesRegex(ValueError, "Corrupt archive member"):
            receive_bundle(damaged, file_digest(damaged), self.local)
        self.assertFalse((self.local / "receipts").exists())
        self.assertTrue(Path(result["archive"]).exists())


if __name__ == "__main__":
    unittest.main()
