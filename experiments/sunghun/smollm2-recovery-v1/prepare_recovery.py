"""Freeze missing-only recovery inputs while preserving original source bytes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "smollm2-release-v1"))
from audit_backup import audit
from build_release import FINGERPRINT, METADATA_SHA, encode, require
from load_data import sha256


def prepare(source, output):
    require(not output.exists(), "Never overwrite a frozen recovery input")
    checked = audit(source)
    require(checked["invalid_batches"] == 0, "Invalid preserved batches")
    require(checked["verified_responses"] == 1744 and checked["missing_responses"] == 6729, "Unexpected recovery scope")
    with zipfile.ZipFile(source / "final-metadata.zip") as z:
        old = {name: z.read(name) for name in z.namelist()}
    config = json.loads(old["input/config.json"])
    questions = json.loads(old["input/questions.json"])
    by_id = {q["id"]: q for q in questions}
    missing_ids = [identifier for batch in checked["missing"] for identifier in batch["ids"]]
    require(len(set(missing_ids)) == 6729, "Duplicate missing IDs")
    config["study"] = "smollm2-recovery-v1"
    config["storage"].update(maximum_outbox_gib=12, maximum_study_gib=22)
    plan = dict(original_fingerprint=FINGERPRINT, original_metadata_sha256=METADATA_SHA,
        reason="Original ephemeral /tmp study lost after server restart; only missing examples are regenerated",
        preserved=checked["verified"], missing_batches=checked["missing"], total_missing=6729,
        preserved_responses=1744, new_run_outputs_must_not_be_presented_as_original_bytes=True)
    files = {name: value for name, value in old.items() if name.startswith("src/")}
    for name in ["recovery_worker.py", "launch.py", "bootstrap.sh"]:
        files["src/" + name] = Path(__file__).with_name(name).read_bytes()
    files.update({"input/config.json": encode(config), "input/questions.json": encode([by_id[i] for i in missing_ids]),
        "input/recovery_plan.json": encode(plan), "input/original-final-metadata.zip": (source / "final-metadata.zip").read_bytes()})
    controls = []
    for batch_name in ["batch_0000", "batch_0002"]:
        with zipfile.ZipFile(source / "incoming" / (batch_name + ".zip")) as z:
            records = [json.loads(line) for line in z.read("generations.jsonl").splitlines()][:2]
            for record in records:
                identifier = record["id"]
                path = "input/controls/" + identifier + ".npz"
                files[path] = z.read("features/" + identifier + ".npz")
                controls.append(dict(row=by_id[identifier], npz=path, original_batch=batch_name))
    require([q["row"]["dataset"] for q in controls] == ["gsm8k", "gsm8k", "svamp", "svamp"], "Control selection")
    files["input/controls.json"] = encode(controls)
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = dict(study=config["study"], fingerprint=fingerprint, files=hashes, total=6729,
        model_revision=config["model"]["revision"], original_fingerprint=FINGERPRINT,
        original_metadata_sha256=METADATA_SHA, official_gsm_test_loaded=False,
        preserved_responses=1744, recovery_batches=421, controls_excluded_from_release=True)
    files["RUN_MANIFEST.json"] = encode(manifest)
    output.mkdir(parents=True)
    bundle = output / "input-bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for name, value in files.items():
            z.writestr(name, value)
    ready = dict(bundle_sha256=sha256(bundle), fingerprint=fingerprint, total=6729,
        original_fingerprint=FINGERPRINT, original_metadata_sha256=METADATA_SHA)
    (output / "READY.json").write_bytes(encode(ready))
    (output / "recovery_plan.json").write_bytes(encode(plan))
    print(json.dumps(ready))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
