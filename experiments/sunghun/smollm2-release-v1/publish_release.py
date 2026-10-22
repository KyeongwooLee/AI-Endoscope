"""Publish only complete, hash-verified compact/full assets; verify GitHub digests before publication."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import zipfile

from build_release import FINGERPRINT, RECOVERY_FINGERPRINT, require
from load_data import sha256

REPO = "KyeongwooLee/AI-Endoscope"
TAG = "smollm2-features-v1"


def verified_assets(root):
    ready = json.loads((root / "READY_TO_PUBLISH.json").read_text())
    require(ready["complete"] and ready["examples"] == 8473 and ready["full_original_batches"] == 530, "Incomplete release")
    require(ready["source_fingerprint"] == FINGERPRINT and ready["recovery_fingerprint"] == RECOVERY_FINGERPRINT, "Wrong release sources")
    hashes = ready["asset_sha256"]
    required = {"smollm2-gsm8k-compact-v1.zip", "smollm2-svamp-compact-v1.zip", "smollm2-run-metadata.zip",
        "smollm2-recovery-metadata.zip", "DATA_README.md", "RELEASE_SCOPE.md", "FULL_INDEX.json", "COMPACT_SCHEMA.json", "FULL_SCHEMA.json", "AI_STARTER.zip"}
    require(required <= set(hashes), "Missing required compact/full documentation assets")
    require(any(n.startswith("smollm2-full-part-") and n.endswith(".zip") for n in hashes), "Missing full data")
    require(set(p.name for p in root.iterdir()) == set(hashes) | {"SHA256SUMS.txt", "READY_TO_PUBLISH.json"}, "Unexpected output files")
    listed = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines():
        expected, name = line.split("  ", 1)
        require(Path(name).name == name and re.fullmatch("[0-9a-f]{64}", expected), "Unsafe checksum entry")
        listed[name] = expected
    require(listed == hashes, "Checksum list disagrees with readiness marker")
    for name, expected in hashes.items():
        require(Path(name).name == name, "Unsafe asset filename")
        require(sha256(root / name) == expected, "Asset changed after verification: " + name)
        require((root / name).stat().st_size < 2 * 1024**3, "Asset exceeds GitHub limit")
    secret = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{24,}")
    for path in root.glob("*metadata.zip"):
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith((".py", ".json", ".md", ".sh", ".txt")):
                    require(not secret.search(z.read(name)), "Credential-like text in metadata; publication blocked: " + name)
    result = dict(hashes)
    result["SHA256SUMS.txt"] = sha256(root / "SHA256SUMS.txt")
    return result


def notes(root):
    summary = json.loads((root / "dataset_summary.json").read_text())
    lines = ["# SmolLM2: Compact / Full Feature Data", "", "## 다운로드", "",
        "- **기존 Qwen train Release와 feature 종류를 맞춘 버전:** `smollm2-gsm8k-compact-v1.zip` (7,473문제)",
        "- **SVAMP compact:** `smollm2-svamp-compact-v1.zip` (1,000문제, 별도 제공)",
        "- **이번 추출의 모든 저장 feature:** `smollm2-full-part-*.zip` 전부. 각각 독립 ZIP이며 같은 폴더에 압축 해제한다. 바이너리 연결은 하지 않는다.",
        "- **AI/팀원에게 먼저 전달:** `RELEASE_SCOPE.md`, `DATA_README.md`, `AI_STARTER.zip`. Raw hidden 분석에는 실제 데이터 ZIP도 필요하다.",
        "- 계산식/축 순서: `COMPACT_SCHEMA.json`, `FULL_SCHEMA.json`. 검증: `SHA256SUMS.txt`, `FULL_INDEX.json`.", "",
        "## 범위와 복구 이력", "",
        "SmolLM2-1.7B-Instruct, 24 blocks, hidden 2048. GSM8K 공식 train은 greedy 1회/문제, SVAMP는 T0.7/top-p0.95 sampling 1회/문제다. 공식 GSM8K test는 포함하지 않는다.",
        "**보존 원본 1,744응답 + 명시적 복구 재실행 6,729응답 = 총 8,473응답.** 서버 재시작으로 미백업 `/tmp` 원본이 유실돼 누락분만 동일한 고정 protocol로 재실행했다.",
        "재실행은 유실 원본 바이트의 복원이 아니다. 각 응답 fingerprint/provenance로 구분하며 새 응답의 숫자 label을 다시 계산했다. 과거 원본 metadata label을 새 응답에 붙이지 않는다.",
        "보존된 GSM8K 2개/SVAMP 2개의 token 경로 exact match와 feature 허용 오차 대조를 통과해야 공개된다. 이것이 모든 유실 응답의 bitwise 동일성을 보장하지는 않는다.", "",
        "| 데이터 | 응답 | 정답 | 오답 | 형식 실패 |", "|---|---:|---:|---:|---:|"]
    for domain, count in summary["datasets"].items():
        lines.append(f"| {domain} | {count['answers']} | {count['correct']} | {count['wrong']} | {count['parse_failures']} |")
    lines += ["", "위는 **이번 공개 응답 기준 집계**다. 과거 유실 전 집계와 다를 수 있다. Label은 엄격한 최종 숫자 일치 판정이며 풀이 의미의 인간 검수나 오류 유형 주석이 아니다.",
        "", "## Compact 대응", "",
        "[Qwen train features v1](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1)와 checkpoint/feature 정의/7-key NPZ 구조를 맞췄다.",
        "상대 깊이 대응: Qwen raw layers 7/10/13/14 → SmolLM2 6/9/11/12, readout 7/10/14 → 6/9/12. 벡터 차원·tokenizer·값은 다르며 Qwen probe 가중치를 직접 재사용하지 않는다.",
        "Full에는 전층 checkpoint hidden, 종료/평균 pooling, 모든 token의 24층 trajectory, logit-lens/JS/top10, native token confidence/길이·mask가 포함된다. 전체 vocabulary logits나 full-token dense hidden/attention은 저장하지 않았다.",
        "", "## 재현 자료", "", f"Original fingerprint: `{FINGERPRINT}`", f"Recovery fingerprint: `{RECOVERY_FINGERPRINT}`",
        "", "두 metadata ZIP에는 각각 원본/복구의 고정 코드와 설정, receipts, QA가 있다. 실제 현재 응답/label은 compact/full을 사용한다.",
        "Torch2.3 top-p strict-determinism 충돌 때문에 누적합만 CPU로 계산한 정책도 명시돼 있다.",
        "모델/probe 가중치와 새 verifier 성능 결과는 포함하지 않는다. `sunghun` 브랜치의 실험/데이터 안내를 참고한다."]
    return "\n".join(lines) + "\n"


def gh(arguments, retry=3):
    for attempt in range(retry):
        result = subprocess.run(["gh", *arguments], capture_output=True, text=True, encoding="utf-8", timeout=2400)
        if result.returncode == 0:
            return result.stdout
        if attempt + 1 < retry:
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("GitHub command failed: " + result.stderr[-2500:])


def publish(root, target):
    require(re.fullmatch("[0-9a-f]{40}", target), "Pin a pushed sunghun commit SHA, not default main")
    assets = verified_assets(root)
    body = notes(root)
    notes_path = root.parent / (root.name + "-release-notes.md")
    notes_path.write_text(body, encoding="utf-8")
    status_path = root.parent / (root.name + "-publication.json")
    def status(state, **extra):
        status_path.write_text(json.dumps(dict(state=state, tag=TAG, updated_unix=time.time(), **extra), indent=2), encoding="utf-8")
    lookup = subprocess.run(["gh", "api", f"repos/{REPO}/releases/tags/{TAG}"], capture_output=True, text=True, encoding="utf-8", timeout=60)
    if lookup.returncode == 0:
        release = json.loads(lookup.stdout)
        require(release["draft"] and RECOVERY_FINGERPRINT in release["body"], "Existing release is not this editable recovery draft")
    else:
        require("404" in lookup.stderr or "Not Found" in lookup.stderr, "Cannot safely determine if release exists")
        gh(["release", "create", TAG, "-R", REPO, "--target", target, "--draft", "--latest=false",
            "--title", "SmolLM2 GSM8K/SVAMP features v1 (compact + full, recovered)", "--notes-file", str(notes_path)])
        release = json.loads(gh(["api", f"repos/{REPO}/releases/tags/{TAG}"]))
    require(release["target_commitish"] == target, "Existing draft targets another commit")
    for name, expected in assets.items():
        current = {a["name"]: a for a in json.loads(gh(["api", f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"]))}
        old = current.get(name)
        if old is not None and old.get("digest") == "sha256:" + expected and old["size"] == (root / name).stat().st_size:
            continue
        status("uploading", asset=name)
        print("UPLOAD", name, flush=True)
        gh(["release", "upload", TAG, "-R", REPO, str(root / name), "--clobber"])
    current = {a["name"]: a for a in json.loads(gh(["api", f"repos/{REPO}/releases/{release['id']}/assets?per_page=100"]))}
    require(set(current) == set(assets), "Remote asset set differs; draft remains unpublished")
    for name, expected in assets.items():
        require(current[name]["size"] == (root / name).stat().st_size, "Remote size mismatch")
        require(current[name].get("digest") == "sha256:" + expected, "Remote SHA256 mismatch: " + name)
    gh(["release", "edit", TAG, "-R", REPO, "--draft=false", "--latest=false"])
    published = json.loads(gh(["api", f"repos/{REPO}/releases/tags/{TAG}"]))
    require(not published["draft"], "Release is still a draft")
    status("published", url=published["html_url"], verified_assets=len(assets))
    print("RELEASE_PUBLISHED", published["html_url"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--target", required=True)
    args = p.parse_args()
    publish(args.root.resolve(), args.target)
