"""Freeze the model revision and package verified local dataset text, not Qwen outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from huggingface_hub import HfApi

from common import sha, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source = Path(__file__).resolve().parent
    output = workspace / "outputs/smollm2-detailed-v1-prepared"
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    previous = workspace / "outputs/gsm8k-train-all-v1"
    manifest = json.loads((previous / "manifest.json").read_text())
    assert sha(previous / "questions.json") == manifest["questions_sha256"]
    gsm = json.loads((previous / "questions.json").read_text())
    assert len(gsm) == len({q["id"] for q in gsm}) == 7473
    assert {q["id"] for q in gsm} == {f"train-{i:05d}" for i in range(7473)}
    svamp_path = workspace / "outputs/svamp-transfer-v1-prepared/input/SVAMP.json"
    assert sha(svamp_path) == "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
    svamp = json.loads(svamp_path.read_text())
    assert len(svamp) == 1000
    config["model"]["revision"] = HfApi().model_info(config["model"]["id"]).sha
    assert len(config["model"]["revision"]) == 40
    config["prompt"] = manifest["config"]["prompt"]
    gsm_rows = [dict(id=q["id"], dataset="gsm8k", dataset_index=i, question=q["question"], gold=q["gold"])
                for i, q in enumerate(gsm)]
    svamp_rows = [dict(id="svamp-" + q["ID"], dataset="svamp", dataset_index=i,
                       question=q["Body"].strip() + " " + q["Question"].strip(),
                       gold="#### " + str(q["Answer"]), reference_equation=q["Equation"])
                  for i, q in enumerate(svamp)]
    # Inspect both domains early without selecting by correctness or difficulty.
    rows = gsm_rows[:32] + svamp_rows[:32] + gsm_rows[32:] + svamp_rows[32:]
    write_json(output / "input/config.json", config)
    write_json(output / "input/questions.json", rows)
    files = {"src/" + p.relative_to(source).as_posix(): p.read_bytes()
             for p in source.rglob("*") if p.is_file() and "__pycache__" not in p.parts
             and p.suffix in {".py", ".json", ".md", ".sh"}}
    files.update({"input/config.json": (output / "input/config.json").read_bytes(),
                  "input/questions.json": (output / "input/questions.json").read_bytes()})
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    run_manifest = dict(study=config["study"], fingerprint=fingerprint, files=hashes,
        model_revision=config["model"]["revision"], total=len(rows), official_gsm_test_loaded=False,
        gsm_questions_sha256=sha(previous / "questions.json"), svamp_sha256=sha(svamp_path),
        previous_features_or_responses_used=False, prior_qwen_prompt_reused=True)
    write_json(output / "RUN_MANIFEST.json", run_manifest)
    files["RUN_MANIFEST.json"] = (output / "RUN_MANIFEST.json").read_bytes()
    with zipfile.ZipFile(output / "input-bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    write_json(output / "READY.json", dict(bundle_sha256=sha(output / "input-bundle.zip"),
               fingerprint=fingerprint, model_revision=config["model"]["revision"], total=len(rows)))
    print(json.dumps(json.loads((output / "READY.json").read_text())))


if __name__ == "__main__":
    main()
