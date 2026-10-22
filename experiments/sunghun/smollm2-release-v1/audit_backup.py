"""Audit surviving original batches without altering the source backup."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from build_release import FINGERPRINT, METADATA_SHA, require
from load_data import sha256


def audit(source):
    source = Path(source)
    require(sha256(source / "final-metadata.zip") == METADATA_SHA, "Final metadata mismatch")
    with zipfile.ZipFile(source / "final-metadata.zip") as z:
        receipts = [json.loads(z.read(name)) for name in sorted(z.namelist())
            if name.startswith("outbox/batch_") and name.endswith(".json")]
    valid, missing, invalid, dataset_counts = [], [], [], Counter()
    for receipt in receipts:
        path = source / "incoming" / (receipt["name"] + ".zip")
        if not path.exists():
            missing.append(dict(batch=receipt["name"], ids=receipt["ids"]))
            continue
        try:
            require(path.stat().st_size == receipt["bytes"], "Size mismatch")
            require(sha256(path) == receipt["archive_sha256"], "ZIP checksum")
            with zipfile.ZipFile(path) as z:
                require(set(z.namelist()) == set(receipt["members"]), "ZIP member set")
                for name, expected in receipt["members"].items():
                    require(hashlib.sha256(z.read(name)).hexdigest() == expected, "Member checksum: " + name)
                records = [json.loads(line) for line in z.read("generations.jsonl").splitlines()]
            require([r["id"] for r in records] == receipt["ids"], "ID list mismatch")
            require(all(r["fingerprint"] == FINGERPRINT for r in records), "Run fingerprint")
            dataset_counts.update(r["dataset"] for r in records)
            valid.append(dict(batch=receipt["name"], ids=receipt["ids"], bytes=receipt["bytes"],
                sha256=receipt["archive_sha256"]))
        except (ValueError, OSError, zipfile.BadZipFile) as exc:
            invalid.append(dict(batch=receipt["name"], error=str(exc), ids=receipt["ids"]))
    return dict(source_fingerprint=FINGERPRINT, expected_batches=len(receipts),
        expected_responses=sum(r["count"] for r in receipts), verified_batches=len(valid),
        verified_responses=sum(len(r["ids"]) for r in valid), verified_dataset_responses=dict(dataset_counts),
        missing_batches=len(missing), missing_responses=sum(len(r["ids"]) for r in missing),
        invalid_batches=len(invalid), verified=valid, missing=missing, invalid=invalid,
        publication_complete=not missing and not invalid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = audit(args.source)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ["verified", "missing", "invalid"]}))


if __name__ == "__main__":
    main()
