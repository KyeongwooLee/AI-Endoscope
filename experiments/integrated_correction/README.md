# GSM8K·MATH 오류 탐지 및 hidden 교정

[연구 명세](../../docs/final/IntegratedPipelinePlan.md)를 구현한다. 공식 train의
fit·tune·calibration 분할에서 probe와 reward head를 학습하고, 교정 설정을 잠근 뒤
공식 test에서 비교한다. 기반 Qwen 모델은 BF16으로 실행하고 reward head는 FP32를
사용한다. 응답 유형은 고정한 모델과 샘플링 조건에서의 다중 라벨이다.

## 실행

저장소 루트에서 Linux와 BF16을 지원하는 CUDA GPU 한 대를 사용한다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/integrated_correction/requirements.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```

기본 [config.json](config.json)은 공식 전체 분할을 사용한다. 최초 검증에서는
데이터셋당 2문제, pilot에서는 100문제를 고정 seed로 선택할 수 있다.

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("experiments/integrated_correction/config.json").read_text())
for name, limit in [("smoke", 2), ("pilot", 100)]:
    Path(f"/tmp/integrated-correction-{name}.json").write_text(
        json.dumps({**config, "limit_per_dataset": limit})
    )
PY

config=/tmp/integrated-correction-smoke.json
run_dir=/tmp/integrated-correction-smoke
set -e
for stage in prepare generate-label-bank label-distributions generate-features \
             train-probes generate-prefix-rollouts train-rewards tune-correction evaluate report; do
    python -m experiments.integrated_correction "$stage" --config "$config" --run-dir "$run_dir" --resume
done
```

pilot은 `/tmp/integrated-correction-pilot.json`과 별도 run 디렉터리를 사용한다.
전체 실행은 기본 config와 새 run 디렉터리에서 시작한다. `--resume`은 설정, 소스,
환경, 입력 manifest가 일치할 때 완료한 문제 단위를 검증하고 이어서 실행한다.
단계마다 `manifest.json`의 `outputs`와 각 문제의 `receipt.json`에서 출력 해시를
확인할 수 있다. 각 문제 디렉터리는 문제 ID 문자열의 canonical JSON SHA-256이다.
`report`는 결과 Markdown의 경로를 출력한다. 같은 디렉터리의 `report.json`에는
지표, 신뢰구간, 유형별 결과와 head 상태가 있으며 `failures.json`에는 응답 실패와
gold 누락, train/test 중복 제거 ID가 있다.

`prepare`는 공식 릴리스의 SHA-256을 확인하고 공식 parquet와 질문·해설이 일치하는
원본 행을 재사용한다. 새 실험의 공식 split과 원본 행 번호는 고정 parquet를
기준으로 한다. 중복 비교는 Unicode NFC와 연속 공백 정규화를 적용한다.
`INTEGRATED_CORRECTION_CACHE`를 지정하면 run 간 다운로드를 공유할
수 있다. `release_dir`를 지정할 경우 해당 디렉터리 아래에 release tag별로 자산과
`SHA256SUMS.txt`를 둔다. 사용한 자산과 채택 기준은 `prepare`의 `provenance.json`에
기록한다. 기존 릴리스의 생성 조건은 새 실험과 달라 이번 실행에서 응답을 다시
생성한다. 기존 Minsu와 Sunghun 코드는 계획에 적힌 커밋에서 확인했다.
현재 저장소에서 `research_base` 참조의 커밋은 확인되지 않아 provenance에
`null`로 기록한다.

## 비교 조건

주 오류 probe와 클래스 probe는 최초 응답의 `r_mean`을 사용한다. K=5 클래스
probe는 greedy 응답과 독립 표본 4개의 평균을 사용하며, K=5 답변 비교는 다수결과
`1 − P(incorrect)` 가중 투표를 함께 계산한다. Generic 재풀이는 동일 프롬프트에서
고정 seed로 독립 표본을 한 번 생성한다. Global·routed·oracle·shuffled 교정은
동일한 tune 설정을 사용한다. Oracle과 test reward 진단의 label bank 및
continuation 비용은 해당 연구용 산출물에서 확인할 수 있다. 교정 비교 표는 각
조건의 최초 응답과 추론 이후 계산량을 집계한다.

예산 비교의 기본 상한은 문제당 200,000 forward 토큰이다. 조건별 총 상한은
같은 문제 수와 상한의 곱이다. 배치의 패딩, 반복 처리한 prefix와 채택되지 않은
토큰도 비용에 포함한다. 한도 안에 재풀이 또는 투표를 완료하지 못하면 최초 응답을
반환하고 소진을 기록한다. 작은 smoke 분할은 probe나 reward 학습에 필요한 예제가
부족할 수 있다. 비활성화 사유와 기본 교정 설정 사용 여부는 체크포인트에 기록한다.

## 검증

```bash
python -m pytest -q -p no:cacheprovider experiments/integrated_correction/tests
INTEGRATED_CORRECTION_TEST_QWEN=1 python -m pytest -q -p no:cacheprovider \
    experiments/integrated_correction/tests
```

두 번째 명령은 고정 모델의 layer-20 residual 및 다음 토큰 logits 정렬, α=0 토큰
동등성, EOS와 빈 풀이 처리를 CUDA에서 확인한다.

## 공개 응답 은행으로 GSM8K probe 학습

`gsm8k_release_bank: true`를 사용하면 GSM8K만 대상으로 실행한다.
[`gsm8k-consistency-types-v1`](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-consistency-types-v1)의
숫자 답 30개와 형식 플래그를 공식 train 행 ID, 질문, gold로 대조한 뒤 파이프라인의
B/C/D/E 규칙을 다시 적용한다. 릴리스에 없는 train 64문제는 제외하고
공식 test는 별도로 유지한다. 샘플 답의 원문·종료 사유·개별 seed는 확인 불가로
저장하며 A 라벨도 미관측으로 집계한다. 공개 greedy 토큰을 layer 20에서 replay해
K=1 probe 특징을 만든다. 공개 답의 생성 조건은 `T=1.0, top_p=1.0`이다.

```bash
python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("experiments/integrated_correction/config.json").read_text())
Path("/tmp/integrated-correction-gsm8k-release.json").write_text(
    json.dumps({**config, "gsm8k_release_bank": True})
)
PY
set -e
for stage in prepare generate-label-bank label-distributions generate-features train-probes; do
    python -m experiments.integrated_correction "$stage" \
        --config /tmp/integrated-correction-gsm8k-release.json \
        --run-dir runs/gsm8k-release --resume
done
```

`generate-label-bank`는 다운로드한 결과를 변환하는 단계다. 소규모 실행은
config의 `limit_per_dataset`을 설정한다. `release_dir` 사용 시 해당 디렉터리에
`gsm8k-consistency-types-v1/{SHA256SUMS.txt,types.jsonl,greedy_token_ids.jsonl}`
및 GSM8K 원본 릴리스 자산을 둔다. 교정까지 실행할 때는 기존 순서대로
`generate-prefix-rollouts`부터 이어서 실행한다. 이때 reward용 sampled 원본과
continuation은 새로 생성한다. K=5 클래스 probe는 비활성화되며 test의 K=5 투표는
K=1 오류 probe로 각 후보를 평가할 수 있다. test oracle 은행은 로컬에서 같은
temperature와 top-p로 생성된다. 보고서에서 이 실행을 릴리스 기반 보조 실험으로
표시한다.
