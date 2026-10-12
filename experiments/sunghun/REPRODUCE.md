# Reproduction and Data Dependencies

## Environment

Linux CUDA 실행은 Python 3.10~3.12, RTX 3090 24GB, 호환 NVIDIA driver를 기준으로 한다.
CPU에서도 offline unit tests와 일부 classifier 분석은 가능하지만 hidden probe 학습과
실제 Qwen inference는 CUDA가 필요하다.

```bash
cd experiments/sunghun/gsm8k-logit-study
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements.txt -r requirements-probes-gpu.txt
python -m unittest discover -s tests -v
```

원래 `requirements-classifier.txt`는 로컬 Python 3.12 분석 환경용이며,
일부 전이 dependency가 Python 3.10과 맞지 않을 수 있다. 위 GPU용 파일을 구분해서 사용한다.
CPU 테스트는 첫 줄의 CUDA wheel 대신 PyTorch CPU wheel을 설치한다.
테스트는 작은 합성 모델과 fixture를 사용하며 실제 Qwen/GSM8K를 다운로드하지 않는다.

## Original Generation and Logit-Lens Study

```bash
python -m gsm8k_lens --help
python -m gsm8k_lens doctor --config configs/rtx3090.fp32-pinned.json --run /tmp/gsm8k-study/run-new
```

pilot 생성, labeling, feature 추출, gate, exploration, test, 분석과 checksum 기반
회수 명령은 [원본 실행 안내](gsm8k-logit-study/README.md)에 있다.
실제 과거 run 이름이 등장하는 부분은 이력 설명이며, 새 실행은 새 경로와 pinned config를 사용한다.
gold는 채점에만 사용하고, 저장한 자기 생성 token IDs를 causal replay한다.

## Classifiers and Probes

아래 명령은 `gsm8k-logit-study/`에서 실행한다. 대문자 경로는 사용자가 제공해야 할 데이터다.

```bash
# Original run: generations, labels, analysis/example_features.parquet, etc.
python scripts/train_error_classifier.py --run ORIGINAL_RUN --output NEW_CLASSIFIER_OUTPUT
python scripts/train_midlayer_classifier.py --run ORIGINAL_RUN --splits SPLIT_IDS_JSON --output NEW_MIDLAYER_OUTPUT

# JSONL with id, phase, prompt_token_ids, generated_token_ids, incorrect.
python scripts/extract_hidden_probes.py --input EXAMPLES_JSONL --output NEW_HIDDEN_OUTPUT
python scripts/train_hidden_probes.py --data HIDDEN_OUTPUT --splits SPLIT_IDS_JSON --output NEW_PROBE_OUTPUT
python scripts/analyze_hidden_trajectory.py --help

python scripts/prepare_feature_ablation.py --hidden HIDDEN_OUTPUT --run ORIGINAL_RUN --splits SPLIT_IDS_JSON --output NEW_ABLATION_INPUT
python scripts/train_feature_ablation.py --input ABLATION_INPUT --output NEW_ABLATION_OUTPUT
```

과거 split/feature/학습된 artifact가 필요한 실험은 입력을 임의로 바꾸고 같은 실험으로 취급하지 않는다.
In particular, saved probe/scaler files are required for the frozen confirmation baseline.
Do not load `.pt` or `.joblib` artifacts from untrusted sources.

## Fresh Confirmation

원본 코드는 `/tmp/gsm8k-confirmation-v1` staging layout을 사용한다.
`work/package_confirmation.py`가 필요한 입력을 구성하고, `work/run_confirmation.py`와
`work/bootstrap_confirmation.sh`가 실행한다. 기존 exploration ablation 입력,
inner split, frozen initial probe/scaler, 제외할 ID/question hashes가 필요하다.
`scripts/confirm_entropy_features.py`와 import 대상들을 packager의 layout에 맞게 staging한다.
[Protocol](gsm8k-logit-study/ENTROPY_CONFIRMATION.md)을 먼저 확인한다.

## Full Train Compact Extraction

```bash
# Staged root must already contain the pinned inputs and gsm8k_lens sources.
PYTHONPATH=. python scripts/extract_train_all.py --root /tmp/gsm8k-train-all-v1
```

이 명령만으로 입력이 자동 생성되는 것은 아니다. 다음 입력이 필요하다.

- `input/config.json`: pinned original inference configuration.
- `input/known.jsonl`: exactly 2,128 verified historical generations, with question and gold.
- `input/provenance.json`: source provenance used in final metadata backup.
- `gsm8k_lens/*.py` and `extract_train_all.py`: original versions copied into the root.
- Optional `questions.json`: pinned official train selection/order. If absent, train is downloaded.

`work/package_train_all.py` builds this package from the verified original and confirmation outputs.
The frozen extractor intentionally requires those historical generations; it is not a generic
"generate all from scratch" entry point. Reusing receipts also requires the corresponding verified
local archives or remote acknowledgements, not just a copied progress count.

각 batch ZIP에는 `features.npz`, `generations.jsonl`, `summary.json`이 들어간다.
Raw vectors are saved at four checkpoints only; prefix readouts retain layers 7/10/14 through t128.
Full-token residuals and vocabulary-wide logits are not retained.
After inference has started independently on the server, receive verified batches locally:

```bash
python ../work/collect_train_all.py --host USER@HOST --port PORT --local-run LOCAL_OUTPUT
python ../work/audit_train_all_backup.py --host USER@HOST --port PORT --local-run LOCAL_OUTPUT
```

Server control code is `work/train_all_control.py`, staged at `/tmp/gsm8k-train-all-v1`.
Register/verify the SSH host key in advance; SSH authentication is configured outside this repository.
The collector uses system `ssh`/`scp`, or explicit `--ssh-bin`/`--scp-bin` paths.
Only verified batches are acknowledged for remote deletion. Keep the local ZIPs after remote ACK.
SSH disconnect does not stop a detached inference process, but stopping VESSL may erase `/tmp`.

## Historical Helper Layout

Some `work/` scripts intentionally retain the workspace layout used during the experiments:

```text
WORKSPACE/
  outputs/
    gsm8k-logit-study/       # copy the project code here
    vessl-qwen-gsm8k-fp32-v6/
    error-classifier-v1/
    hidden-probes-v1/
    feature-ablation-v1/
    entropy-confirmation-v1/
    gsm8k-train-all-v1/
  work/                     # copy archived helper scripts here
```

These helpers are archived run code, not universally portable launchers. Inspect the required
`/tmp` staging roots, READY.json/package hashes and existing inputs before execution.
Do not launch a second producer against an active root, and do not run cleanup against unrelated data.

## Data Publication

No dataset artifacts or credentials are included in this branch. Final train batch ZIPs total
730,618,272 bytes, excluding historical study data. A separate data release would need an inventory,
SHA-256 manifest, feature schema, original dataset attribution and documented split history.
Until that is published, results cannot be independently reconstructed from this checkout alone.
