# SVAMP inference와 GSM8K 양방향 verifier 전이 데이터 v1

Qwen2.5-1.5B-Instruct의 **SVAMP 1,000응답**, 비교에 사용한 **GSM8K sampling 4,000응답**,
생성 완료 후 feature, 두 방향의 verifier 평가 결과를 공유한다. 이번 배포에서 새 inference나 학습을 하지는 않았다.

## 다운로드와 읽는 순서

[전용 Release: posthoc-transfer-data-v1](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/posthoc-transfer-data-v1)
에서 아래 파일을 받는다. 이 README → `DATA_SCHEMA.json` → `dataset_summary.json` → 목적에 맞는 ZIP 순서로 읽는다.

| 파일 | 내용 / 누구에게 필요한가 |
|---|---|
| `svamp-inference-v1.zip` | SVAMP 1,000문제 전체의 실제 응답·생성 token IDs·정오답·완료 hidden/confidence/length feature. SVAMP 분석에는 이것부터 |
| `gsm8k-sampled-posthoc-v1.zip` | 500문제×8회 sampling한 GSM8K 4,000응답과 같은 종류의 feature. 비교 분석 및 오류 유형 주석 연결용 선택 다운로드 |
| `transfer-results-v1.zip` | GSM8K 내 평가, GSM8K→SVAMP, SVAMP→GSM8K의 metrics·개별 예측·학습된 작은 probe·고정 선택/보정/threshold·계산 정의 코드 사본 |
| `DATA_README.md`, `DATA_SCHEMA.json`, `dataset_summary.json` | AI와 팀원에게 먼저 전달할 설명, 기계 판독용 feature 사전, 실제 집계 |
| `svamp_answers.jsonl`, `gsm8k_answers.jsonl` | AI 전달용 전체 질문·응답·reference·label. 토큰/생성 설정을 담은 `generation`만 제외한 JSONL이며 hidden 배열은 ZIP에 별도 포함 |
| `RELEASE_AUDIT.json`, `SHA256SUMS.txt` | 원본 대조·정오답 재계산·AUROC 재계산 기록, 다운로드 파일 해시 |

각 ZIP에는 이 설명, schema/summary/audit, 출처 고지, 내부 파일별 `checksums.sha256`도 들어 있다.
모델 전체 가중치, cache/환경, SSH 키, 서버 접속용 스크립트, 모든 token의 residual/vocabulary logits는 포함하지 않는다.
데이터 JSONL/NPZ를 읽는 데 GPU나 Qwen 모델은 필요 없다. NPZ는 `numpy.load(..., allow_pickle=False)`로 읽는다.

## 7,473개와 4,000개의 차이

**기존 데이터를 줄인 것이 아니다. 별도 생성 실험이다.**

| 실험 | 정확한 범위 | 생성 방식 | 배포 |
|---|---|---|---|
| 과거 전체 train 추출, 대략 8천 개라고 부르던 자료 | 공식 GSM8K train 7,473문제×1응답 | greedy | 기존 [gsm8k-train-features-v1 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1), 그대로 보존 |
| 후속 sampling/posthoc 실험 | 공식 train 중 500문제×8응답 = 4,000응답 | T=0.7 sampling | 이번 GSM8K ZIP |
| 다른 dataset inference | SVAMP 원저자 challenge 1,000문제×1응답 | T=0.7 sampling | 이번 SVAMP ZIP |

기존 7,473개 Release는 중간 h7/h10/h13/h14 등의 자료이고, 이번 것은 생성 완료 후 h28 pool 등이다.
같은 GSM8K 질문이 있어도 생성 응답·feature가 다르므로 서로 행 순서로 연결하지 않는다.
이번 GSM8K의 `test`는 **500개 안에서 나눈 내부 평가 fold 100문제**다. 공식 GSM8K test 1,319문제 실험이 아니다.

## 대상과 정오답

| 대상 | 질문 | 응답 | 정답 | strict 오답 | 형식 실패 | 잘림 | 분석 가능 | 그중 오답 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GSM8K sampling 전체 | 500 | 4,000 | 2,976 | 1,024 | 128 | 7 | 3,872 | 896 |
| SVAMP 전체 | 1,000 | 1,000 | 762 | 238 | 34 | 0 | 966 | 204 |

잘림과 형식 실패는 겹칠 수 있으므로 단순 합산하지 않는다.
`answer_correct`는 마지막 비어 있지 않은 줄이 `#### <number>` 형식일 때 최종 숫자를 gold와 Decimal 동치 비교한 값이다.
분수·문자 설명·단위·수식은 이 엄격한 final-line parser의 허용 숫자가 아니다. 파싱 실패는 strict 오답에 포함한다.
`analysis_eligible`은 파싱 정상이고 잘리지 않았는지다. `answer_correct` 자체와 달리 잘림을 별도로 반영한다.
주 분석은 `analysis_eligible=true`에 한정하며, 형식 실패/잘림은 기권으로 별도 보고한다.

**숫자 일치는 풀이 전체의 논리적 타당성이나 reference의 무오류를 보장하지 않는다.**
이 정오답 값은 사람의 추론 검수 결과가 아니라 원본 실험의 자동 판정이며 이번 배포에서 바꾸지 않았다.

## 생성 protocol

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. 28 blocks, hidden 1,536. Qwen 본체는 학습하지 않았다.
- FP32, SDPA, TF32 off, batch 1. T=0.7, top_p=0.95, top_k=0, 최대 생성 1,024토큰. EOS 또는 완성된 final line의 줄바꿈에서 종료한다.
- SVAMP 입력은 `Body.strip() + ' ' + Question.strip()`뿐이다. Equation/Answer/Type은 generator나 feature 추출 입력에 넣지 않았다.
- SVAMP 원본 revision: `78e727689e1c1bebfc4be39c446898e8e10b0518`, 원본 SHA256 `5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e`. 원래 순서 전체 1,000문제를 사용했다.
- SVAMP seed는 `SHA256('2026101901:' + response_id)` 앞 8 hex를 정수화한 값 `% (2**31)`이다. 실제 seed·종료 사유·생성 설정은 각 `generation`에 보존했다.
- 같은 prompt/stopping을 쓰지만 GSM8K는 8후보, SVAMP는 1후보다. SVAMP 결과를 best-of-N 또는 greedy 성능이라고 부르지 않는다.
- `original_config.json`에는 초기 공통 설정도 남아 있다. 이번 sampling 설정은 `study_config.json` 및 각 응답의 `generation.effective_generation_config`를 함께 확인한다.

System prompt:

```text
You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write nothing after this final line. Do not omit this line. The final answer must contain only a number, without units, currency symbols, Markdown emphasis, or LaTeX.
```

User template:

```text
Problem:
{question}

Solve the problem step by step. After your reasoning, write the answer on the LAST line using exactly this format:
#### <number>
Replace <number> with the numerical answer. Do not write any text after that line.
```

## 파일 구조와 연결

각 dataset ZIP의 주요 파일:

- `records.jsonl`: 한 줄에 한 실제 응답. `id`, `question_id`, `dataset`, `fold`, 질문/reference, 원본 정오답·품질, `response`, `response_sha256`, 원본 `generation` 포함.
- `generation`: 실제 prompt/generated token IDs, text, seed, 종료·생성 설정 등 원본 정보를 보존했다. dataset별로 원본 필드 범위는 조금 다르다.
- `features.npz`: `ids`와 아래 feature 배열. 행 순서는 JSONL과 같지만 항상 ID로 검증한다. dtype/열 이름은 schema를 기준으로 한다.
- `feature_metadata.json`: 26개 confidence 열 순서와 응답별 pool token 수·추출 QA 등 원본 상세 metadata.
- `original_config.json`, `study_config.json`: 원본 공통 설정과 해당 study 설정. SVAMP ZIP에는 원본 `SVAMP.json`과 provenance도 있다.

**오류 유형 주석과 연결:** 이전 [error-taxonomy-full-v1-draft Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft)의
`labels.jsonl` 1,100개는 이번 GSM8K/SVAMP의 분석 가능 오답 896+204개와 정확히 대응한다.
`id`로 연결하고 `response_sha256`도 확인한다. 오류 유형은 **모두 인간 검수 전 AI 초안**이며 이번 배포로 검수 완료가 되지 않는다.
이번 `records.jsonl`의 binary label과 오류 유형 주석을 구별한다. reference/gold 의심 플래그가 있어도 binary label을 자동 수정하지 않는다.

## Feature 계산

모두 **모델 자신의 실제 생성 token 경로를 causal replay**해 추출했다. gold 풀이를 teacher forcing하지 않는다.
`h28`은 마지막 transformer block 출력이며 **최종 RMSNorm 적용 전**이다. 중간 layer probe용 데이터가 아니다.

| 배열 | shape | 계산 |
|---|---|---|
| `question` | N×1536 | prompt-only forward의 마지막 prompt token h28. R2 수정본으로, 전체 completion forward의 부동소수점 길이 효과를 피함 |
| `end` | N×1536 | 마지막 non-special 생성 token을 소비한 뒤의 h28. 공백/줄바꿈일 수도 있음 |
| `mean` | N×1536 | non-special이고 decoded 문자열이 공백만은 아닌 모든 생성 token의 h28 평균. reasoning 외 final 형식/숫자도 포함 |
| `number` | N×1536 | strict final 숫자 문자 구간과 겹치며 공백만은 아닌 생성 token의 h28 평균. 해당 mask가 비면 0벡터 |
| `confidence` | N×26 | 아래 13개 요약을 whole mask와 final-number mask에 각각 계산 |
| `length` | N×4 | `log1p([prompt token 수, generated token 수, whole mask token 수, number mask token 수])` |

생성 token y_t 직전 state의 최종 RMSNorm 출력에 LM head를 적용한 **원래 분포** p_t를 사용한다.
sampling의 temperature/top_p/top_k를 적용한 분포가 아니다. token별 logp는 `ln p_t(y_t)`,
entropy는 `-sum_v p_t(v) ln p_t(v)`(nats), margin은 top1 확률−top2 확률이다.

whole mask와 number mask 각각의 13개 요약, 이 순서:

1. `logp_mean`, `logp_q10`, `logp_min`, `logp_last16_mean`
2. `prob_lt_01_ratio`, `prob_lt_05_ratio`: 생성 token의 원래 확률이 각각 0.1/0.5 미만인 비율
3. `entropy_mean`, `entropy_q90`, `entropy_max`, `entropy_last16_mean`
4. `margin_mean`, `margin_q10`, `margin_min`

마지막 16개는 해당 mask로 고른 token 중 마지막 최대 16개다. 전체 생성의 마지막 16개와 다를 수 있다.
26열은 `whole_...` 13개 뒤 `number_...` 13개이며, mask가 비면 해당 요약을 NaN으로 보존한다.
NaN을 정답/오답 판정으로 바꾸지 않는다. imputer/scaler는 fit에서만 학습한다.
개별 token의 전체 logit/hidden 이력은 없으므로 저장하지 않은 모든 후속 feature를 재추출할 수 있는 자료는 아니다.

### Probe 입력 그룹

| 그룹 | 차원 | 순서 |
|---|---:|---|
| `question` | 1536 | question |
| `length` | 4 | length |
| `confidence` | 26 | confidence |
| `confidence_length` | 30 | confidence, length |
| `completed_hidden` | 4608 | end, mean, number |
| `combined` | 4638 | confidence, length, end, mean, number |
| `combined_question` | 6174 | confidence, length, end, mean, number, question |

**이름 충돌 주의:** `features.npz`의 `question`/`length`/`confidence`는 입력 feature 배열이다.
`results/*predictions.npz`의 같은 이름은 해당 그룹 **probe의 scalar score**다.
SVAMP `features.npz`의 `score_*` 역시 GSM8K-trained probe score다. 새 classifier 입력으로 섞지 않는다.
`wrong=1`/큰 score가 높은 오답 위험을 뜻한다. raw score는 세 seed sigmoid 출력의 평균이며 보정된 정답 확률이 아니다.
보정값과 수용 threshold는 각 학습 source의 validation에서 고정했다. 저장된 score를 재보정하려면 해당 source bundle을 구별한다.

## 학습·전이 실험과 결과

Qwen 본체는 고정하고 작은 linear 또는 width32 ReLU MLP만 학습했다.
각 그룹에서 seeds 42/43/44 ensemble, source fit-only imputer/scaler,
validation BCE early stopping과 ensemble validation AUROC architecture 선택을 사용했다.
AdamW lr=0.001, weight_decay=0.01, batch128, 최대100epoch, patience12.
선택·보정·threshold는 target에서 조정하지 않았다.

- **GSM8K source:** 500질문을 300 fit / 100 validation / 100 test로 고정, 각 8응답. 기존 질문 선택과 결과를 이미 본 탐색적 실험이다.
- **GSM8K→SVAMP:** 기존 GSM8K probe를 그대로 1,000 SVAMP 응답에 적용. SVAMP로 학습·선택·보정하지 않았다.
- **SVAMP→GSM8K:** 동일 SVAMP를 정규화 질문 hash와 seed `2026102001`로 600/200/200 분할. source fit만 학습, validation만 선택/보정하고 source test 200문제와 기존 GSM8K 내부 test 100문제/800응답에서 평가했다. 새 생성은 없다.
- SVAMP의 `fold`는 역방향 실험을 위해 나중에 추가한 것이다. 순방향 전이 당시에는 전체 1,000개가 target 평가였으며 이 split으로 target 적응하지 않았다.

아래는 **형식 정상·비잘림 응답**의 AUROC다. `completed_hidden`을 primary로 미리 정했고, combined가 더 높다고 사후 primary를 바꾸지 않았다.

| feature | GSM-trained → GSM 내부 test | GSM-trained → SVAMP 전체 | SVAMP-trained → GSM 내부 test |
|---|---:|---:|---:|
| question | 0.6726 | 0.6099 | 0.5201 |
| length | 0.7117 | 0.6807 | 0.6663 |
| confidence | 0.7947 | 0.7048 | 0.7805 |
| confidence_length | 0.8308 | 0.7864 | 0.7997 |
| completed_hidden | 0.8608 | 0.8126 | 0.8059 |
| combined | 0.8581 | 0.8228 | 0.7872 |
| combined_question | 0.8457 | 0.8089 | 0.7614 |

Primary hidden의 question-bootstrap 95% CI: GSM 내부 [0.8180,0.8990], 순방향 SVAMP [0.7787,0.8459], 역방향 GSM [0.7512,0.8612].
역방향 source heldout SVAMP 200문제에서는 hidden AUROC 0.8537(분석 가능194/오답41)이었다.
GSM 내부 test는 분석 가능771/오답187, SVAMP 전체는 분석 가능966/오답204다.

`transfer-results-v1.zip`의 `gsm8k/`, `svamp/`, `reverse/` 아래 `results/metrics.json`, `RESULTS.md`, 예측 NPZ에 상세 결과가 있다.
모델은 각 하위 `models/`에 있다. `gsm8k/models/`와 순방향 `svamp/models/`는 같은 GSM-trained bundle이고, `reverse/models/`는 SVAMP-trained bundle이다.
`reverse/provenance.json`은 정확한 SVAMP question fold를 기록한다. `methods/`는 당시 계산 정의/학습/평가 코드의 보존 사본으로, 원래 input·GPU·임시 경로 구조를 가정한다. 이 ZIP만 풀어 자동 재실행할 수 있다고 보장하지 않는다.

**체크포인트 주의:** `.pt`는 직접 만든 작은 probe지만 NumPy metadata를 포함한 pickle 기반 파일이다.
파일 해시와 출처를 확인한 신뢰하는 파일만 로드하고, 검증 없이 임의의 `.pt`를 실행하지 않는다.
score와 결과를 읽기만 할 때는 `.pt` 로딩이 필요 없다.

## 분석 시 제한

- AUROC는 구분 순위 성능이지 정답 보장이나 보정된 신뢰도가 아니다. domain shift 시 calibration/수용 오답률을 별도로 확인한다.
- 두 방향은 source 학습 응답 수·질문당 후보 수·평가 모집단이 다르다. 단순 방향 간 점수 차이를 paired 검정 또는 한 데이터셋의 우월성으로 해석하지 않는다.
- 기존 project에서 관찰한 SVAMP/GSM test를 재사용했으므로 완전히 미사용 최종검증셋이 아니다. validation을 선택/early stopping/보정에 재사용한 낙관성도 가능하다.
- 질문 단위 split/2,000회 bootstrap을 사용한다. GSM8K의 8응답을 독립 질문으로 세지 않는다. SVAMP의 의미적/템플릿 변형 관계까지 완전히 분리했다고 보장하지 않는다.
- 둘 다 산술 문장제다. 비수학 분야 전이, 더 어려운 문제에서의 안정성, Qwen 사전학습 오염 부재를 입증한 결과가 아니다.
- label/gold/reference/교정 설명/오류 유형/형식 실패 상태와 이전 probe score를 새 classifier의 입력에 섞지 않는다. 허용 feature와 baseline을 먼저 고정한다.
- 파일/숫자 label/AUROC 검사를 통과한 것과 인간 의미 검수는 다르다. 이전 오류 유형 1,100개는 여전히 `not_reviewed`다.

## 로딩 예시

```python
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path("svamp-inference-v1")
rows = [json.loads(s) for s in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
with np.load(root / "features.npz", allow_pickle=False) as data:
    assert data["ids"].tolist() == [r["id"] for r in rows]
    hidden = np.concatenate([data[k] for k in ["end", "mean", "number"]], axis=1)
eligible = np.array([r["analysis_eligible"] for r in rows])
wrong = np.array([not r["answer_correct"] for r in rows], dtype=int)
by_id = {r["id"]: i for i, r in enumerate(rows)}
# 오류 유형 Release의 labels.jsonl을 별도로 받은 경우:
for line in Path("labels.jsonl").read_text(encoding="utf-8").splitlines():
    label = json.loads(line)
    if label["dataset"] != "svamp":
        continue
    i = by_id[label["id"]]
    assert hashlib.sha256(rows[i]["response"].encode()).hexdigest() == label["response_sha256"]
    assert label["human_review_status"] == "not_reviewed"
```

## 출처와 검증

문제/reference 출처: [GSM8K / OpenAI](https://github.com/openai/grade-school-math),
[SVAMP / Arkil Patel 외](https://github.com/arkilpatel/SVAMP).
upstream MIT 고지는 `LICENSE-GSM8K.txt`, `LICENSE-SVAMP.txt`에 동봉했다. 모델 가중치는 배포하지 않는다.
원본 세 snapshot의 파일별 manifest, GSM generation archive receipt, SVAMP 원본 hash를 대조했다.
5,000개 strict label 재계산, feature ID/shape/유한값·NaN 검사, 1,100개 주석 연결, 28개 저장 AUROC 재계산,
새 ZIP 내부 체크섬 readback을 수행한다. 원본 hash와 재포장한 ZIP hash는 서로 다르다.

원본 백업이 있는 작업 공간에서 패키징 재실행:

```text
python package_data.py --workspace <original-workspace-root>
```

이 명령은 원본을 수정하지 않으며 GPU inference나 학습을 실행하지 않는다. 원본 백업 경로가 필요하다.
