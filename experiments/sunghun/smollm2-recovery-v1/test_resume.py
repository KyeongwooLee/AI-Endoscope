import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import prepare_resume


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        prepared, backup = root / "prepared", root / "backup"
        prepared.mkdir()
        incoming = backup / "incoming"
        incoming.mkdir(parents=True)
        plan = json.dumps(dict(missing_batches=[dict(batch="batch_0109", ids=["example"])])).encode()
        hashes = {"input/recovery_plan.json": hashlib.sha256(plan).hexdigest()}
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        with zipfile.ZipFile(prepared / "input-bundle.zip", "w") as z:
            z.writestr("input/recovery_plan.json", plan)
            z.writestr("RUN_MANIFEST.json", json.dumps(dict(fingerprint=fingerprint, files=hashes)))
        ready = dict(fingerprint=fingerprint, bundle_sha256=prepare_resume.sha(prepared / "input-bundle.zip"))
        (prepared / "READY.json").write_text(json.dumps(ready))
        record = dict(id="example", fingerprint=fingerprint, label=dict(answer_correct=False))
        body = (json.dumps(record) + "\n").encode()
        archive = incoming / "batch_0109.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("generations.jsonl", body)
        receipt = dict(name="batch_0109", ids=["example"], count=1, fingerprint=fingerprint,
            archive_sha256=prepare_resume.sha(archive), bytes=archive.stat().st_size,
            members={"generations.jsonl": hashlib.sha256(body).hexdigest()}, summaries=[record])
        (incoming / "batch_0109.json").write_text(json.dumps(receipt))
        return prepared, backup, fingerprint

    def test_restore_receipt_and_ack_only_after_hash_check(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, backup, fp = self.fixture(Path(directory))
            with patch.object(prepare_resume, "RECOVERY_FINGERPRINT", fp):
                files, audit = prepare_resume.resume_state(prepared, backup)
            self.assertEqual(audit["verified_recovery_responses"], 1)
            self.assertEqual(audit["combined_verified_responses"], 1745)
            ack = json.loads(files["acks/batch_0109.json"])
            self.assertTrue(ack["verified_local"])
            self.assertTrue(ack["restored_from_verified_local"])
            self.assertNotIn("workflow_status.json", files)

    def test_corrupted_local_archive_is_never_acknowledged(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, backup, fp = self.fixture(Path(directory))
            with (backup / "incoming/batch_0109.zip").open("ab") as stream:
                stream.write(b"corrupt")
            with patch.object(prepare_resume, "RECOVERY_FINGERPRINT", fp):
                with self.assertRaises((ValueError, AssertionError)):
                    prepare_resume.resume_state(prepared, backup)

    def test_changed_summary_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prepared, backup, fp = self.fixture(Path(directory))
            path = backup / "incoming/batch_0109.json"
            receipt = json.loads(path.read_text())
            receipt["summaries"][0]["label"]["answer_correct"] = True
            path.write_text(json.dumps(receipt))
            with patch.object(prepare_resume, "RECOVERY_FINGERPRINT", fp):
                with self.assertRaisesRegex(ValueError, "Summary labels"):
                    prepare_resume.resume_state(prepared, backup)


if __name__ == "__main__":
    unittest.main()
