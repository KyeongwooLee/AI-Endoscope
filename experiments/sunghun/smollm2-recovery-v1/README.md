# SmolLM2 Missing-Only Recovery

2026-10-22 서버 재시작으로 `/tmp`의 미백업 원본이 유실돼 누락분만 재실행하는 protocol이다.
**보존 원본 1,744응답은 변경하지 않고, 누락 6,729응답만 새 fingerprint로 추출한다.**
새 응답을 잃어버린 원본과 bitwise 동일한 데이터라고 주장하거나, 과거 label을 새 응답에 붙이지 않는다.

- 보존: GSM8K 1,712 + SVAMP 32 = 1,744응답 / 109배치.
- 재실행: GSM8K 5,761 + SVAMP 968 = 6,729응답 / 421배치.
- 결합 후: GSM8K 공식 train 7,473 + SVAMP 1,000 = 8,473응답 / 530배치.
- Original fingerprint: `dc40716a9fffcb546b478c759ba5e9a90546c9481ec830c53051f9059c342a2c`.
- Recovery fingerprint: `75d50117968c998f82ddb6aad71d55f3c56f8dfbc6037b3cfa25e497bf963603`.

## 유지한 것

Original 완료 metadata에서 **고정된 source bytes**와 model revision, prompt, seed, generation,
추출 설정 및 문제 목록을 읽는다. `prepare.py`로 최신 revision을 다시 조회하지 않는다.
`run.generate`, `extract.extract`, 숫자 label parser, FP32 저장과 top-p CPU cumsum 정책은 원래 실행 코드 그대로 사용한다.
모델은 같은 SmolLM2-1.7B-Instruct revision, RTX3090, Torch2.3.1+cu121, Transformers4.46.3이다.

보존 응답에서 사전에 고정한 GSM8K 2개 + SVAMP 2개를 재현한다. Prompt/generated token IDs, text, seed의
exact match 및 모든 feature 배열의 shape/value 비교를 통과해야 누락분을 시작한다.
실수 배열 비교 허용 오차는 `rtol=5e-4, atol=5e-4`, 정수/mask는 exact match다.
대조군은 Release 응답 수에 중복 포함하지 않는다. 결과는 `reproduction_check.json`에 기록한다.

## 파일

- `prepare_recovery.py`: 로컬 보존 원본 해시 검증, 누락 ID/대조군 고정, 새 fingerprint와 입력 ZIP 생성.
- `recovery_worker.py`: 보존 ID를 제외한 추론/feature 추출, 개별 generation과 NPZ 저장, 16응답 batch commit 및 QA.
- `bootstrap.sh`, `launch.py`: 고정 라이브러리 설치, 원래 코드 테스트 후 SSH와 독립적인 worker 실행.
- `collect_recovery.py`: 30초 간격의 로컬 백업, ZIP 및 모든 내부 member SHA-256 확인 후 해당 서버 ZIP만 ACK 정리.
- `finish_release.py`: 완료·검증된 백업이 준비되면 compact/full 빌드 후 draft에 업로드하고 원격 digest 확인 후 공개.
- [배포 설명](../smollm2-release-v1/README.md): compact/full 범위, layer 대응, feature 계산식/배열 축과 로더.

원격 결과/입력/receipt는 `/root/ai-endoscope/smollm2-recovery-v1`의 별도 볼륨에 둔다.
Model cache와 임시는 `/tmp`에 둔다. 이것만으로 볼륨 삭제까지 견딘다고 가정하지 않으며 **로컬 백업도 병행**한다.
로컬 PC와 VPN이 켜져 있어야 백업/업로드가 진행된다. SSH가 끊겨도 worker는 독립 실행되고,
백업이 불가능하면 12GiB outbox 상한과 6GiB 여유 공간 guard가 원본을 보존하며 중지시킨다.

## 상태와 공개 Gate

`progress.json`의 `completed`는 **재실행 6,729개 중 진행도**, `combined_completed`는 보존 1,744개를 더한 수다.
`workflow_status.json` / `job_complete.json` / 로컬 `collector_status.json` / `release_pipeline.json`을 구분한다.
Metadata만 받은 상태를 feature 전체 백업 완료로 해석하지 않는다.

배포 빌더는 두 실행의 source fingerprint, 보존/재실행 ID의 disjoint coverage, 설정 일치, 재현 QA,
530개 batch와 내부 member hash, 모든 배열 shape, 새 숫자 label 재판정, compact write/readback,
full 원본 batch ZIP 보존, GitHub asset 제한과 최종 SHA를 검증한다.
업로더는 전체 readiness marker가 없으면 중단하고, GitHub draft의 asset 크기/SHA-256이 모두 일치한 뒤 공개한다.
최종 공개 집계는 과거 원본 집계와 다를 수 있으므로 `RELEASE_SCOPE.md`/`dataset_summary.json`을 기준으로 한다.
