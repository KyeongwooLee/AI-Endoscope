# GSM8K Self-Generated Logit-Lens Study

Qwen2.5-1.5B-Instruct가 GSM8K 답변을 직접 생성하고, 최종 정답/오답에 따라 내부 layer의 분포가 다른지 비교한다. **모델 학습이나 fine-tuning은 하지 않는다.** RTX 3090 한 장(24 GB)에서 생성과 feature 추출을 순차 실행하는 구성이다.

상위 폴더의 `README.md`는 실험 설계 문서이며, 이 파일은 실행 안내다. 2026-10-07 VESSL의 RTX 3090에서 실제 checkpoint/GSM8K 실험을 시작했다. 토큰 정렬 수정 후 현재 실행은 `../vessl-qwen-gsm8k-fp32-v6/RUN_NOTES.md`에 기록하며, 진행 상태는 같은 폴더의 `job_status.json`과 `job.log`에서 확인한다. v5는 보존하고, 생성 답변 768개를 체크섬 그대로 v6에 이관해 feature와 gate를 재검증한다. 실행 시작은 전체 실험 완료를 의미하지 않는다.

## 1. 기본 설정

| 항목 | `configs/rtx3090.json` |
|---|---|
| 모델 | `Qwen/Qwen2.5-1.5B-Instruct`, prepare 시 commit SHA 고정 |
| 장비 / dtype | CUDA GPU 1장 / FP32, 양자화 없음 |
| Attention | PyTorch SDPA, FlashAttention 별도 설치 없음 |
| 생성 | zero-shot, greedy, batch 1, 문제당 1회 |
| 길이 | prompt 최대 3,072 + 답변 최대 1,024 = 총 4,096 tokens |
| 데이터 | train pilot 128 / 별도 train exploration 1,000 / test 1,319 |
| 추출 | 28개 decoder block의 final RMSNorm 적용 전 residual |
| Projection | layer별, 32 target tokens씩 norm + LM head |
| 보관 | `features_only`: raw residual 저장 안 함; token별 feature, top-5 IDs/logits 유지 |
| 원격 저장 | `/tmp/gsm8k-study/run-v2`, 최대 128문제씩 처리 후 로컬 전송/검증/원격 정리 |
| 주 비교 | reasoning 구간 layer 14의 mean JS-to-final |
| 통계 | 문제 단위 평균, bootstrap 2,000회, 오답 - 정답 차이의 95% CI |

VRAM 24 GB를 고려해 full `[layer, token, vocabulary]` logits를 만들지 않는다. FP32 가중치 약 6 GB 외에 KV cache, replay activation, projection 임시 tensor가 필요하다. BF16 사전 실행에서 생성/replay argmax 일치율이 기준에 미달해 FP32로 변경했으며, FP32 사전 8문제에서 수치 검증을 통과했다. 전체 입력 길이에 대한 GPU peak는 실행 중 계속 확인한다. Optimizer, gradient, training batch, learning rate 설정은 필요 없다.

CPU RAM은 16 GB 이상, 가능하면 32 GB를 권장한다. Raw residual은 계산 중 CPU/GPU 메모리에서만 사용하고 디스크에는 쓰지 않는다. 현재 형식의 합성 직렬화 측정상, 128문제가 모두 1,024 tokens라면 결과 파일 약 0.45 GB, 전송용 ZIP을 만드는 동안 약 0.9 GB다. 모델/설치 환경/cache는 별도이며, 최소 5 GiB 디스크 여유를 유지한다. 실제 크기는 pilot에서 확인한다.

원격의 문제별 파일은 한 묶음에 한정한다. 최대 약 1,280개 결과 파일 + 디렉터리가 일시적으로 생기며, 로컬 검증 뒤 삭제된다. inode가 더 부족하면 `--batch-size 32` 또는 `8`로 낮춘다. 로컬에서는 모든 결과와 검증된 ZIP을 보관하므로, 전체 결과만 약 5~18 GB를 예상한다(평균 256~최대 1,024 tokens 가정, 분석/환경 별도). 이 숫자는 실제 추론 측정치나 상한 보장이 아니다.

## 2. VESSL 준비

Linux, NVIDIA driver, Python 3.10~3.12가 필요하다. 프로젝트도 `/tmp/gsm8k-study/project` 등에 배치하면 그 아래 `.venv`도 `/tmp`에 생성된다. **`/tmp`가 별도 디스크라는 보장은 없다.** root와 같은 filesystem일 수도 있고 RAM 기반 tmpfs일 수도 있으므로 먼저 확인한다. 작업 종료/재시작 시 `/tmp`가 지워질 수 있다.

```bash
df -h / /tmp
df -i / /tmp
findmnt -T /tmp
# 위 결과에서 공간/inode/마운트 타입 확인 후 설치한다.
cd /tmp/gsm8k-study/project
bash scripts/setup_vessl.sh
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export RUN_DIR=/tmp/gsm8k-study/run-v2
export HF_HOME=/tmp/gsm8k-study/hf-cache
python -m gsm8k_lens doctor --run "$RUN_DIR"
```

설치 스크립트는 공식 CUDA 12.6 PyTorch wheel을 사용한다. `nvidia-smi` 및 doctor에서 driver/CUDA 호환성을 확인한다. 시스템 CUDA toolkit이나 `nvcc`를 직접 설치할 필요는 없지만 호환 NVIDIA driver는 필요하다. BF16/CUDA가 불가능하면 CPU나 FP16으로 조용히 전환하지 않고 실패한다.

주요 dependency 버전은 `requirements.txt`로 고정한다. 설치 스크립트는 pip 다운로드 cache를 끄고, 임시 파일/Hugging Face/XDG cache를 `/tmp/gsm8k-study` 아래로 설정한다. Python CLI도 Linux에서 이 cache 경로를 기본값으로 쓰되 명시된 환경변수는 보존한다. 라이브러리 설치 자체의 파일 수는 별도이므로 환경의 inode도 확인한다. `pip freeze > environment.lock.txt`로 전이 dependency를 기록할 수 있다.

## 3. 로컬에서 원격 실행 및 회수

로컬과 원격에 동일 버전 코드를 둔다. 로컬 Python에는 같은 requirements가 필요하지만 GPU는 필요 없다. 로컬에서 `ssh`/`scp`가 실행 가능해야 한다. SSH host key는 최초 접속 때 확인하여 known_hosts에 등록한다. 도구는 host key 검증을 끄지 않으며 비대화형 SSH 인증을 사용한다. 현재 실제 run은 `/tmp/gsm8k-study/run-fp32-v5`이며, 아래 `run-v2`는 예시 이름이다.

아래 명령은 **로컬 컴퓨터의 프로젝트 폴더에서** 실행한다. `vessl`은 직접 설정한 SSH alias 예시다. 다른 접속이면 `--host USER@HOST --port PORT --identity KEY_PATH`로 지정한다. `LOCAL_RUN`은 실제 로컬 저장 경로로 바꾼다. PowerShell에서는 아래 명령을 한 줄로 입력하거나 줄 연결 문법을 바꾼다.

```bash
python scripts/sync_from_vessl.py --host vessl \
  --remote-project /tmp/gsm8k-study/project \
  --remote-run /tmp/gsm8k-study/run-v2 \
  --local-run LOCAL_RUN --phase pilot --batch-size 8 --batches 1
python -m gsm8k_lens analyze --run LOCAL_RUN --phase pilot --allow-partial
```

동작 순서:

1. 원격에서 아직 처리하지 않은 최대 N문제를 생성 → 라벨링 → 추출한다.
2. 문제별 결과와 run 메타데이터를 ZIP 하나로 묶는다. ZIP 내부 파일 목록과 SHA-256도 기록한다.
3. 로컬 `.part` 파일로 가져와 전체 크기/SHA-256을 확인한다.
4. 로컬 run에 복원하고 **각 파일의 SHA-256을 다시 검증**한다. 기존 로컬 파일과 충돌하면 덮어쓰지 않고 멈춘다.
5. 로컬 검증 완료 receipt를 원격에 보내면 원격은 해당 묶음의 파일/ZIP만 삭제한다. 모델, 환경, 데이터 명세, 처리 이력은 삭제하지 않는다.
6. 삭제 완료 이후에만 다음 묶음을 시작한다. `sync_from_vessl.py` 단독 실행은 phase/gate를 자동 전환하지 않는다.

현재 실제 실행은 `scripts/run_vessl_experiment.py`로 관리한다. 이 wrapper는 고정된 분석 설정을 유지한 채 pilot 전체 완료 및 수치 검증, eligible generation coverage 80% 이상을 확인하고 다음 단계로 진행한다. Exploration 전체 완료 후 분석 설정을 고정하고 test를 시작한다. 결과에 따라 primary layer/feature를 선택하지 않는다. 실패 시 `job_status.json`에 오류를 기록하고 중단한다. 로컬 컴퓨터가 켜져 있고 네트워크가 연결돼 있어야 회수와 후속 실행이 계속된다. Windows에서는 `--ssh-bin`/`--scp-bin`에 실제 OpenSSH 실행 파일 경로를 지정할 수 있다.

현재 v6의 pilot/exploration은 완료됐으며, 남은 test는 `scripts/remote_queue.py --action produce`를 서버 독립 세션으로 실행하는 방식으로 전환했다. 이 test 전용 producer는 기존 gate를 요구하고, 로컬 전송을 기다리지 않고 미처리 묶음을 계산한다. `scripts/collect_detached.py`는 로컬에서 네트워크 재연결 및 회수만 담당하고, 전체 test 회수 후 분석한다. `sync_from_vessl.py --queue-only --phase test`는 서버 계산을 시작하지 않고 이미 대기 중인 결과만 회수한다. 서버가 살아 있으면 SSH/로컬 PC 연결이 끊겨도 계산은 계속된다. 대기 ZIP 8 GiB 또는 filesystem 여유 6 GiB 미만이면 다음 묶음을 기다린다. ZIP 외 원본도 보존하므로 실제 보관량은 더 크다. VESSL stop/terminate나 서버 재부팅에 대한 자동 복구는 아니다. 진행은 원격 `server_job_status.json`/`server_job.log`, 회수는 로컬 `collector.log`/`collector.stderr.log`로 확인한다.

전송 실패/체크섬 불일치/로컬 디스크 부족 시 원격 원본은 보존한다. 재실행하면 전송 대기 묶음을 먼저 처리하며 이미 전송한 답변을 다시 생성하지 않는다. 원격 삭제가 중간에 끊겨도 receipt로 재개한다. 재개 시 로컬 데이터도 다시 검증한다. 단, 전송 전 원격 `/tmp` 자체가 소실되는 장애는 방어할 수 없다. 작은 batch일수록 미전송 노출 범위가 작다.

로컬에 이미 run이 있으면 원격 manifest의 동일성을 생성 전에 확인한다. `/tmp`가 지워져 원격 이력이 사라진 경우 새 실험으로 조용히 재시작하지 않고 멈춘다. 이 경우 로컬 메타데이터와 ZIP/receipt를 사용한 원격 이력 복원이 필요하다. 자동 원격 재구축은 이번 도구의 범위에 포함하지 않는다.

## 4. Pilot → Exploration → Test

```bash
# 나머지 pilot을 회수한다. 이미 회수한 smoke 8문제는 다시 생성하지 않는다.
python scripts/sync_from_vessl.py --host vessl --remote-project /tmp/gsm8k-study/project --local-run LOCAL_RUN --phase pilot --batch-size 128 --batches 2
python -m gsm8k_lens analyze --run LOCAL_RUN --phase pilot

# 보고서/정답 파싱/품질/시간/디스크 사용량을 사람이 검토한 뒤 승인한다.
python -m gsm8k_lens freeze-pilot --run LOCAL_RUN
scp LOCAL_RUN/pilot_gate.json vessl:/tmp/gsm8k-study/run-v2/pilot_gate.json

# 고정한 train 1,000문제
python scripts/sync_from_vessl.py --host vessl --remote-project /tmp/gsm8k-study/project --local-run LOCAL_RUN --phase exploration --batch-size 128 --batches 8
python -m gsm8k_lens analyze --run LOCAL_RUN --phase exploration

# 탐색 보고서를 확인하고, test를 보기 전에 분석을 고정한다.
python -m gsm8k_lens freeze-analysis --run LOCAL_RUN
scp LOCAL_RUN/analysis_lock.json vessl:/tmp/gsm8k-study/run-v2/analysis_lock.json

# 최종 test 1,319문제
python scripts/sync_from_vessl.py --host vessl --remote-project /tmp/gsm8k-study/project --local-run LOCAL_RUN --phase test --batch-size 128 --batches 11
python -m gsm8k_lens analyze --run LOCAL_RUN --phase test
```

포트/identity가 필요하면 gate 업로드의 `scp`에도 `-P PORT -i KEY_PATH`를 지정한다. Gate는 복원된 로컬 전체 phase의 검증 결과로 만든다. 이미 결과를 지운 원격에서 analyze/freeze를 시도하지 않는다. 전송된 파일을 다시 생성하지 않도록, offload 모드의 원격에서 기존 전체 generate/label/extract 명령은 차단한다.

`scripts/run_phase.sh`도 이제 이름과 달리 **묶음 하나를 준비하고 대기**한다. 자동 전송은 로컬 `sync_from_vessl.py`를 사용한다. `--batches`는 현재 호출이 처리할 묶음 수의 상한이고, `--batch-size`는 1~128이다. CLI는 `RUN_DIR` 환경변수도 읽는다. Config 변경은 기존 run에 허용하지 않으므로 이번 features-only 실행은 새 `run-v2` 이름을 사용한다.

## 5. 생성과 Replay의 구분

1. 문제만 chat template에 넣고 모델이 풀이와 `#### <number>`를 생성한다.
2. 생성된 정확한 prompt/답변 token IDs를 저장한다.
3. 저장된 경로를 causal replay하여 block 출력들을 얻는다. **Gold 풀이 경로를 따라가는 실험이 아니다.**
4. 생성 token `y_j`는 그 token을 입력하기 직전 위치의 residual과 연결한다. 첫 답변 token은 prompt의 마지막 위치에서 예측한다.
5. 각 residual에 동일 checkpoint의 final RMSNorm과 LM head를 적용한다. 마지막 layer에도 raw residual을 사용하므로 norm이 중복 적용되지 않는다.

현재는 답변 생성 1회 + 내부 상태 추출 replay 1회다. 첫 8개 pilot 문제는 생성 시 native logits도 요약하여 replay와 비교한다. 모든 문제에서 final lens/native 분포의 log-probability 최대 오차를 확인한다. 허용치는 config에 있으며 실패하면 완료 marker를 만들지 않는다. BF16 full-sequence replay와 cached generation은 부동소수점 차이가 있을 수 있으므로 실제 3090 pilot에서 허용치를 검토해야 한다.

## 6. Feature와 정오답

- Entropy, top-1/top-2 probability margin, generated-token log-probability/rank.
- Final layer와의 full-vocabulary JS divergence 및 top-1 agreement.
- Prediction stabilization depth, late stabilization depth, layer 간 top-1 switch 수.
- Adjacent-layer JS 및 다음 layer로 갈 때 generated-token log-probability 변화량.
- 문제/구간 내 하위 10% generated-token log-probability 평균.

모든 분포는 FP32 log-softmax로 계산하고 자연로그 단위를 사용한다. Final layer의 adjacent JS/support change는 비교할 다음 layer가 없으므로 NaN이다. Top-1 tie는 작은 token ID, rank는 `1 + strictly-greater count` 규칙이다. 평균과 중앙값, `log(1+rank)`, 변화량 절댓값도 저장한다.

마지막 비어 있지 않은 줄이 `#### <number>`인 경우만 파싱한다. 쉼표/부호/소수는 Decimal로 비교하며 임의의 마지막 숫자로 fallback하지 않는다. 파싱 실패는 strict accuracy에서 오답으로 포함하지만, 주 feature 비교는 **파싱 성공 + 미절단 응답**만 사용한다. 길이 제한, 파싱 실패, 실행 실패를 별도로 보고한다. 숫자가 맞아도 truncated이면 주 비교에서 제외한다.

구간은 reasoning / final_number / final_first_token 보조 요약이다. Special token, 공백-only, `####` 형식 token은 분석에서 제외한다. Token ID를 다시 모델 입력으로 바꾸지 않으며, span 할당용 재인코딩은 원래 ID와 일치해야 한다. 불일치 시 명시적으로 중단한다.

## 7. 결과 구조

```text
RUN_DIR/
  manifest.json                  # config, immutable revisions, template, provenance
  sample_ids.json
  data/{pilot,exploration,test}.jsonl  # questions only
  data/gold.json                 # generation/extraction do not read this
  generations/PHASE/ID/generation.json
  labels/PHASE/ID.json
  features/PHASE/ID/
    token_features.parquet
    topk_tokens.parquet
    token_diagnostics.parquet
    example_features.parquet
    trajectory.parquet
    quality.json
  analysis/PHASE/
    report.md
    summary.json
    group_statistics.csv
    confidence_correlations.csv
    length_comparison.csv
    figures/
  pilot_gate.json
  analysis_lock.json
  errors/PHASE/STAGE/ID.json
  offload/records/BATCH.json    # remote: small transfer ledger, retained after cleanup
  offload/outbox/BATCH.zip     # remote: only until local verification
  incoming/BATCH.zip           # local: retained archive copy
  receipts/BATCH.json          # local: verified receipt
```

집단 평균은 token을 한꺼번에 pooling하지 않고 문제별 요약값을 같은 가중치로 비교한다. 레이어 곡선, JS trajectory heatmap, stabilization ECDF, 길이 층화 비교, 고정 seed로 선택한 정오답 사례를 출력한다. Test의 길이 bin은 exploration에서 정한 경계를 재사용한다. 실패/coverage/작은 집단 경고 및 차이가 없는 결과도 보고한다.

## 8. 후속 Feature 추가

이번 실행은 residual을 보관하지 않는다. 기존 scalar/top-5로 계산되는 집계는 로컬 parquet 후처리로 가능하지만, 새로운 full-distribution feature에는 동일 checkpoint와 저장된 token IDs를 이용한 추가 replay가 필요하다. 전송 중인 run의 코드를 변경하지 말고 별도 후속 분석으로 진행한다. 기존 답변 자체를 새로 생성할 필요는 없다. Test를 본 뒤 추가한 feature는 후속 탐색으로 표시한다.

## 9. 테스트와 제한

```bash
python -m unittest discover -s tests -v
```

테스트는 네트워크를 사용하지 않으며 임의 초기화한 작은 Qwen2 모델을 쓴다. 정답 파서, span 정렬, next-token shift, causal prefix invariance, final norm/native 일치, 생성/replay 일치, chunk 독립성, feature 수식, cache 재사용, 부분 실행/재시작, pilot/test gate 및 보고서 생성까지 검사한다. 실제 Qwen checkpoint/3090의 numerical gate와 VRAM 검증을 대체하지는 않는다.

추가 검증은 residual 미저장, 묶음별 재시작, SHA-256/멤버 검증 실패 시 원격 보존, 잘못된 receipt 거부, 삭제 중단 후 재개, unrelated 파일 보존, 복원된 로컬 데이터의 분석/gate 생성을 포함한다. 실제 SSH/SCP 연결과 VESSL GPU 실행은 접속 정보를 받은 뒤 검증해야 한다. 실제 모델 다운로드/GSM8K 추론은 아직 하지 않았다.

2026-10-07 로컬 CPU 검증: 전체 20개 테스트 통과. 전송/삭제 테스트는 로컬의 서로 다른 디렉터리를 원격/로컬 대역으로 사용했으며 실제 VESSL 파일을 삭제한 것이 아니다.

주요 한계: logit lens 자체의 layer별 보정 차이, 정오답군의 문제 난이도 차이, 파싱/길이 제외로 인한 selection bias, GSM8K 학습 노출 가능성. 여기서의 집단 차이를 첫 오류 위치, 인과 원인, 개별 오류 탐지 성능으로 해석하지 않는다.

참고: [고정 버전 Qwen2 구현](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py), [PyTorch 공식 이전 버전 설치](https://pytorch.org/get-started/previous-versions/), [Qwen 모델](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [GSM8K](https://huggingface.co/datasets/openai/gsm8k).
