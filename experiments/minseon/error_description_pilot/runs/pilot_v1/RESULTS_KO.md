# 근거 기반 오류 설명 파일럿 결과 (pilot_v1)

> 이 문서는 `run.py --stage report`가 실제 실행 산출물에서 자동 생성합니다. 사람 검수 결과가 들어오기 전까지 의미 평가 지표는 **null(human_review_pending)** 입니다.

## 0. 한 줄 요약

자동 단계(감사·분할·표본·E/G 주석·검증·임베딩·검색·블라인드 검수 패키지·SAE 브리지)는 모두 실행되었고, **의미 평가(같은 오류 이웃 비율)는 독립 사람 검수가 없어 계산하지 않았습니다.** 다음 단계는 검수 패키지의 항목 100개와 쌍 319개를 사람이 판정하는 것입니다.

## 1. 단계 상태

| 단계 | 상태 |
| --- | --- |
| `implementation_complete` | 예 |
| `data_audit_complete` | 예 |
| `protocol_frozen` | 예 |
| `annotations_complete` | 예 |
| `embedding_complete` | 예 |
| `retrieval_complete` | 예 |
| `review_package_ready` | 예 |
| `human_review_pending` | 예 |
| `semantic_evaluation_complete` | 아니오 |
| `primary_contrast_protocol_compliant` | 아니오 — E와 G를 같은 에이전트 세션이 작성(condition_context_isolated=false) |

## 2. 무엇을 시험했나

같은 주석자·같은 입력(문제, 원본 풀이, 틀릴 수 있는 참조 풀이)·같은 임베딩 모델에서, **근거 인용형 짧은 설명(G)**이 **일반 짧은 오류 설명(E)**보다 "같은 구체적 수학 오류"를 가진 이웃을 더 많이 찾는가(주 대비 G−E, 상위 5개 중 사람 검증 same 비율). 보조 대비는 G−R(문제+원본 풀이 텍스트). R은 참조 풀이를 쓰지 않으므로 G−R은 순수 임베딩 효과가 아니라 "참조 보조 주석 파이프라인"의 효용입니다. Q(문제만)와 Random은 통제 조건입니다.

설명 자체의 타당성(관찰 가능한 실수를 맞게 짚었는가)과 표현의 타당성(가까움이 같은 오류를 뜻하는가)은 별도로 평가하도록 설계했습니다. 이 파일럿은 오답 탐지 AUROC, 내부 인과 기제, 수정(repair) 효과를 측정하지 않습니다.

## 3. 데이터·모델·환경

- 원본: `qwen-gsm8k-self-errors-v1` (`errors.jsonl` SHA256 `cc54b199164cdef0…`), Qwen2.5-1.5B-Instruct @ `989aa7980e4c` 기존 오답. 새 답변 생성 없음.
- 감사: 7691응답 / 3489문제, sampled 6380 / greedy 1311, stop 7656 / length 35, 동일 (pid,response) 추가 사본 5, 문제 간 동일 응답 0, 정규화 문제 중복 pid 0. 인수인계 문서 값과의 불일치: 0건.
- 규칙 라벨 분포 {'SETUP': 4422, 'NOEQ': 1797, 'TARGET': 586, 'OTHER': 549, 'ARITH': 311, 'NEWNUM': 26} — 주석자·검수자에게 전달하지 않음. `finish=length` 35건은 모두 `####` 표식이 있었으나 잘림으로 제외.
- 주석 모델: `claude-opus-5-5` (Anthropic, Claude Code (VS Code extension), executing coding agent), 경로 `native_agent_direct_file_annotation`. 스냅샷 버전·디코딩 설정은 노출되지 않음(`exact_model_version_known=false`). API 호출·유료 엔드포인트 없음.
- 임베딩: `BAAI/bge-m3` @ `5617a9f61b028005a4858fdac845db406aefb181` dense only, pooling=true (CLS), L2 정규화, cosine, float32, sentence-transformers 5.1.2, torch 2.12.1. PCA·군집·재순위 없음.
- 환경: Python 3.10.12, 사용 가능 CPU 4개(호스트 72), 메모리 196642804 kB, GPU NVIDIA GeForce RTX 3090, 디스크 여유 87.2 GB. 임베딩 실측: 18.7초(모델 로드 포함), 프로세스 RSS 피크 3.32 GB, GPU 피크 2.36 GB, 최대 입력 669토큰(한도 8192, 잘림 없음).

## 4. 공유 분할과 표본

- Kyeongwoo의 `prepare.assign_splits`와 같은 `GroupShuffleSplit(test_size=0.2, random_state=42)`를 **필터링 전 7,691행 전체**에 적용: discovery 6114응답/2791문제, confirmation 1577응답/698문제 (문서 값과 일치). 파티션 간 정규화 문제 충돌 0건.
- 기존 주석과의 중복: Sunghun `labels.jsonl` 201문제/467행, 정확 일치 쌍 1건 (문서 값과 일치).
- 확장 제외(문서 6절 지시): Minsu 브랜치 d95116f의 AI 검수 출력 `ab_labels.tsv` 814문제, `loc_labels.tsv` 513문제(이 릴리스의 동일 ID), 로컬 Minseon 수동 판정/예시 사례 문제(질문 텍스트로 검증). 제외 문제 그룹 합계 1306.
- 적격 문제: labels.jsonl+length 제외만 적용 시 discovery 2612 / confirmation 663 (문서의 2,612/663 재현), 전체 제외 적용 후 1752 / 418.
- 표본(seed 20261108, numpy.random.default_rng([seed, stream]) numpy==1.23.5): 개발 30(균등 20 + 희귀 규칙/형식 보강 ARITH=2, NEWNUM=2, TARGET=2, OTHER=2, SHORT_RESPONSE=2), 평가 쿼리 20 + 갤러리 80, 모두 서로 다른 문제. 보강 비율은 말뭉치 유병률이 아닙니다.
- 확인하지 못한 것: Team-discussion explanation attempts not present as runnable artifacts in the inspected branches.; Dahyeon annotation work (branch tip equals main; not reconstructable).; Sunghun's qwen-extra-errors / posthoc-transfer releases (different responses; not used as corpus).

## 5. 결과

### 5.1 주석과 스키마 검증 (자동, 의미 타당성 아님)

| 단계 | 조건 | 패킷 | 수락(1차/2차) | supported_error | insufficient_evidence | ambiguous | no_error_observed | 경고 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | E | 30 | 30/0 | 26 | 2 | 1 | 1 | 1 |
| dev | G | 30 | 30/0 | 26 | 2 | 1 | 1 | 2 |
| eval | E | 100 | 100/0 | 94 | 2 | 4 | 0 | 6 |
| eval | G | 100 | 100/0 | 94 | 2 | 4 | 0 | 5 |

- G 평가: supported_error 94건 모두 `response[start:end] == evidence_quote` 통과(유효 span 비율 94/94). 반복되는 인용에서 주석자가 occurrence를 명시한 경우 3건, 유일 위치 인용 97건(기권 항목의 선택적 인용 포함). 참조/문항 검토 필요 플래그: E 8건, G 8건.
- 경고는 대부분 설명 속 피연산자가 우연히 정답 숫자와 같은 경우입니다(예: 문제의 "$28"과 정답 28). 스키마 통과는 수학적 타당성을 뜻하지 않습니다.
- 문체 점검(E, 평가 94건): 첫 두 단어 상위 [('The solution', 68), ('The question', 6), ('The reasoning', 2)]; "correctly" 포함 10건. 
- 문체 점검(G, 평가 94건): 첫 두 단어 상위 [('The response', 91), ("The response's", 2), ('After computing', 1)]; "correctly" 포함 31건. G 프롬프트는 "일반적 서두"를 빼라고 했으나 거의 모든 G 설명이 같은 주어로 시작합니다. 이 공통 문체가 이웃 순위에 섞였을 수 있으며, 사람 검수 전에는 영향 크기를 알 수 없습니다(평가 데이터를 보고 프롬프트를 고치지 않았습니다).
- **블라인딩 한계(설명 검수 Stage 2):** E 설명 68건이 "The solution", G 설명 93건이 "The response"로 시작해 무작위 A/B 라벨에도 조건을 추정할 수 있습니다. 설명 품질 검수는 완전 블라인드가 아님을 결과 해석에 반영해야 합니다. 주 지표인 쌍 검수는 원본 문제·풀이·참조만 보여주므로 이 단서의 영향을 받지 않습니다.

### 5.2 임베딩·검색 커버리지

| 표현 | 평가 임베딩 | 미임베딩(기권) | 순위 매긴 쿼리 | 평균 사용 갤러리 |
| --- | ---: | ---: | ---: | ---: |
| R | 100 | 0 | 20/20 | 80.0 |
| E | 94 | 6 | 19/20 | 75.0 |
| G | 94 | 6 | 19/20 | 75.0 |
| Q | 100 | 0 | 20/20 | 80.0 |
| Random | — | — | 20/20 | 80.0 |

상위 5개 집합의 방법 간 평균 Jaccard(서술 통계, 의미 판정 아님):

| A | B | Jaccard | 쿼리 수 |
| --- | --- | ---: | ---: |
| R | E | 0.193 | 19 |
| R | G | 0.188 | 19 |
| R | Q | 0.504 | 20 |
| R | Random | 0.028 | 20 |
| E | G | 0.341 | 19 |
| E | Q | 0.134 | 19 |
| E | Random | 0.019 | 19 |
| G | Q | 0.176 | 19 |
| G | Random | 0.037 | 19 |
| Q | Random | 0.040 | 20 |

R과 Q의 이웃이 크게 겹치면 원문 임베딩이 주로 문제 내용(주제)에 끌린다는 신호일 수 있습니다. 다만 겹침은 "같은 오류"의 증거가 아닙니다.

### 5.3 의미 평가 (주 지표)

**null — `human_review_pending`.** 사람 검수 판정이 없으므로 verified_same_error@5, G−E, G−R, 부트스트랩 구간을 계산하지 않았습니다. AI 판정이나 silhouette로 대체하지 않았습니다. 미검수 쌍은 0점도 uncertain도 아닌 결측입니다.

검수 대상: 항목 100개(설명 200개), 쌍 319개 (최대 500 슬롯 중 기권 제외 490개를 중복 제거). 2차 검수자 사전 지정 부분집합: 항목 36개, 쌍 64개.

### 5.4 개발 데이터 K-means 진단 (예시용, taxonomy 검증 아님)

| 표현 | n | K=2..5 군집 크기 (seed 42) | silhouette (서술) | seed 간 ARI 범위 |
| --- | ---: | --- | --- | --- |
| R | 30 | [16, 14]; [14, 10, 6]; [9, 8, 7, 6]; [7, 7, 6, 6, 4] | 0.035, 0.043, 0.030, 0.027 | 0.13–0.43 |
| E | 26 | [14, 12]; [12, 8, 6]; [12, 6, 5, 3]; [10, 6, 5, 3, 2] | 0.044, 0.030, 0.049, 0.036 | 0.10–0.51 |
| G | 26 | [16, 10]; [10, 9, 7]; [9, 8, 6, 3]; [10, 6, 5, 3, 2] | 0.045, 0.037, 0.039, 0.031 | 0.06–0.85 |
| Q | 30 | [19, 11]; [14, 8, 8]; [12, 7, 6, 5]; [10, 6, 5, 5, 4] | 0.031, 0.035, 0.032, 0.037 | 0.13–0.58 |

모든 K·seed를 보고했습니다(K=2–5, seed 42–44, n_init=10). 30건에서 silhouette가 0에 가깝고 seed 간 ARI가 불안정하면 뚜렷한 군집 구조가 확인되지 않은 것으로 해석합니다.

## 6. 실제 출력 예시 (검수 전)

아래는 실제 산출물이며 **사람 검수 전**입니다. "성공/불일치/근거 없는 설명" 판정은 검수 후에만 가능합니다.

#### 개발(discovery)

- `train-129-2` supported_error
  - E: The solution divides the 50 sold cones by 6, as if one sixth of the sold cones were free, but every sixth customer getting a free cone means one free cone per five paid cones, so the 50 paid cones should be divided by 5.
  - G: The response divides the 50 paid cones by 6, as if free cones were one sixth of the cones sold, although every sixth customer receiving a free cone means one free cone per five paid cones; it also reports a fractional number of cones.
  - G 인용(`[305,339)`): `50 cones / 6 = 8.333... free cones`
- `train-300-3` supported_error
  - E: The setup expressions are correct, but the solution never evaluates them and outputs 27.50 and then 5.50; neither value follows from subtracting 0.10*25*7 + 1.50*7 from the 40 dollars paid, so the reported change contradicts its own expressions.
  - G: The response writes correct cost and change expressions but never evaluates them, then outputs 27.50 and 5.50; a change of 5.50 does not follow from subtracting 0.10*25*7 plus 1.50*7 from the 2*20 paid, so the final values contradict its own setup.
  - G 인용(`[616,636)`): `#### 27.50 #### 5.50`
- `train-1083-2` insufficient_evidence (E: insufficient_evidence); reference_review_needed=False; Unexplained final number with no operands shown.
- `train-2866-1` insufficient_evidence (E: insufficient_evidence); reference_review_needed=False; Unexplained number; a unit confusion is possible but not observable.
- `train-424-4` ambiguous (E: ambiguous); reference_review_needed=True; The wording '9/10 kilograms' supports the response's reading; the reference reads it as 9/10 of the 10 kg.
- `train-7444-0` no_error_observed (E: no_error_observed); reference_review_needed=True; The reference computes 7*2 = 14 gigs for 'every other day for 2 weeks', which appears to double-count.

#### 평가(confirmation)

- `train-21-3` supported_error
  - E: When converting snakes to birds, the solution multiplies the 30 snakes by 12, the beetles-per-bird rate, instead of by 3 birds per snake, and then applies the 12 rate again, so the birds-per-snake factor is never used.
  - G: The response converts 30 snakes into birds by multiplying by 12, the beetles eaten per bird, instead of by the 3 birds each snake eats, then multiplies by 12 again for beetles, so one rate is used twice and the birds-per-snake rate is never applied.
  - G 인용(`[403,458)`): `30 snakes will eat \(30 \times 12 = 360\) birds per day`
- `train-63-2` supported_error
  - E: The solution treats 30 situps as a single total goal across all days and adds Monday's and Tuesday's counts against it, instead of treating 30 as the daily goal and making up each day's shortfall on Wednesday.
  - G: The response pools Monday's and Tuesday's situps against one total of 30, concluding the goal is already exceeded, although the question treats 30 as the daily goal and asks Shawna to also make up each day's shortfall, so Wednesday's own 30 and the missed situps are ignored.
  - G 인용(`[285,323)`): `12 (Monday) + 19 (Tuesday) = 31 situps`
- `train-1204-2` ambiguous (E: ambiguous); reference_review_needed=True; '8 more solar systems for every planet' is read as 9 per planet by the reference and as 8 per planet by the response.
- `train-2293-2` insufficient_evidence (E: insufficient_evidence); reference_review_needed=False; No operands are shown, so the source of 2.5 is not observable.
- `train-2373-1` insufficient_evidence (E: insufficient_evidence); reference_review_needed=False; Unexplained final number; no operands shown.
- `train-5612-0` ambiguous (E: ambiguous); reference_review_needed=True; The question may ask for separate counts; the reference reports their sum.
- `train-665-4` ambiguous (E: ambiguous); reference_review_needed=True; The wording allows 3 apples per teacher or 3 per child; the reference uses 3 per child.
- `train-7197-3` ambiguous (E: ambiguous); reference_review_needed=True; 'Twice more' is read as three times as many by the reference and as twice as many by the response.

#### 검색 예시: 쿼리 `train-21-3` (평가, 미검수)

쿼리 G 설명: The response converts 30 snakes into birds by multiplying by 12, the beetles eaten per bird, instead of by the 3 birds each snake eats, then multiplies by 12 again for beetles, so one rate is used twice and the birds-per-snake rate is never applied.

| 방법 | 1위 이웃 | cosine | 이웃의 G 설명(참고용) |
| --- | --- | ---: | --- |
| G | `train-183-1` | 0.627 | The response correctly computes the weekly losses of each bird type and subtracts them from the 580 birds, but evaluates 580 minus 140 minus 56 minus 35 as 359, an execution error of ten in the final subtraction. |
| E | `train-4525-4` | 0.604 | The response correctly counts 64 dog nails and 48 parrot claws, but for the parrot with an extra toe it subtracts 4 claws from the total instead of adding the single extra claw, so the extra toe lowers the count instead of raising it by one. |
| R | `train-183-1` | 0.599 | The response correctly computes the weekly losses of each bird type and subtracts them from the 580 birds, but evaluates 580 minus 140 minus 56 minus 35 as 359, an execution error of ten in the final subtraction. |
| Q | `train-6314-2` | 0.527 | The response correctly tracks 11 ripe peaches after five days but computes the unripe ones as 18 minus 11, which counts the 3 eaten peaches as still unripe in the bowl, instead of subtracting all 14 peaches that ripened from the original 18. |

## 7. 이 결과가 지지하는 것 / 모르는 것

- 지지: 파이프라인이 공유 SAE 분할을 정확히 재현하고, 원본 해시로 ID를 맞추며, 잘림·중복·기존 주석 문제를 기록된 규칙으로 제외하고, 두 조건의 설명을 같은 입력으로 만들고, 정확한 문자 span 검증을 통과한 근거를 남기고, 블라인드 검수 패키지를 만든다는 것.
- 지지: 평가 100건 중 주석자가 실수를 관찰 가능하다고 판단한 비율과 기권·참조 문제 비율(5.1). 이는 AI 판단이며 사람 검증 전입니다.
- 모름: G가 E·R보다 같은 오류 이웃을 더 많이 찾는지(주 질문). 설명이 실제로 맞는지. 이웃 근접이 주제·문체가 아닌 오류 패턴을 반영하는지.
- 제한: E/G 비격리 작성, 같은 모델이 두 조건 작성, 20개 쿼리·80개 갤러리의 작은 표본(작은 갤러리에서는 관련 이웃이 아예 없을 수 있음), GSM8K train의 이미 연구된 데이터, 참조 보조 오프라인 분석, G 설명의 공통 서두, E/G 서두 차이로 인한 Stage 2 설명 검수의 부분적 비블라인드.

## 8. 다음 결정 하나

**사람 검수를 완료할지 결정하고 배정합니다.** 검수자 1명 이상이 `review_package/`를 지침대로 채운 뒤 다음을 실행합니다:

```bash
python -I experiments/minseon/error_description_pilot/run.py --stage evaluate-reviewed --reviews /absolute/path/to/completed_reviews [--adjudication /absolute/path/to/adjudication.csv]
```

검수 없이 taxonomy나 G의 우위를 주장하지 않습니다. 원한다면 같은 패킷으로 E/G를 격리 문맥(별도 세션)에서 다시 생성해 프로토콜 준수 G−E를 만들 수 있습니다.

## 9. 산출물 경로

- [experiments/minseon/error_description_pilot/REPRODUCE.md](../../REPRODUCE.md) — 재현 명령
- [experiments/minseon/error_description_pilot/SOURCE_AUDIT.md](../../SOURCE_AUDIT.md) — 원본·브랜치 감사
- [experiments/minseon/error_description_pilot/REVIEWER_GUIDE.md](../../REVIEWER_GUIDE.md) — 검수 지침
- [experiments/minseon/error_description_pilot/runs/pilot_v1/status.json](status.json) — 단계 상태(기계 판독)
- [experiments/minseon/error_description_pilot/runs/pilot_v1/protocol_frozen.json](protocol_frozen.json) — 동결 프로토콜
- [experiments/minseon/error_description_pilot/runs/pilot_v1/sample_manifest.jsonl](sample_manifest.jsonl) — 표본 130건
- [experiments/minseon/error_description_pilot/runs/pilot_v1/split_manifest.jsonl](split_manifest.jsonl) — 7,691행 분할
- [experiments/minseon/error_description_pilot/runs/pilot_v1/exclusions.jsonl](exclusions.jsonl) — 제외 원장
- [experiments/minseon/error_description_pilot/runs/pilot_v1/overlap_audit.json](overlap_audit.json) — 중복 감사
- [experiments/minseon/error_description_pilot/runs/pilot_v1/annotation/](annotation/) — 패킷·원본 출력·시도 로그·검증 오버레이
- [experiments/minseon/error_description_pilot/runs/pilot_v1/embeddings/](embeddings/) — 벡터·입력 텍스트·해시
- [experiments/minseon/error_description_pilot/runs/pilot_v1/retrieval/](retrieval/) — 전체 순위·top5
- [experiments/minseon/error_description_pilot/runs/pilot_v1/review_package/](review_package/) — 블라인드 검수 파일
- [experiments/minseon/error_description_pilot/runs/pilot_v1/review_key_private/](review_key_private/) — 방법 키(검수자 비공개)
- [experiments/minseon/error_description_pilot/runs/pilot_v1/sae_bridge.jsonl](sae_bridge.jsonl) — SAE 연결 130행
- [experiments/minseon/error_description_pilot/runs/pilot_v1/clustering/dev_kmeans_diagnostic.json](clustering/dev_kmeans_diagnostic.json) — 개발 군집 진단

프로토콜 sha256: `57917ca59b00f225fdd78c8c623c4f2757d640625ea5924cf47ce8f8ddc43e88` (동결 시각 2026-11-07T16:34:00+00:00). 실행 디렉터리는 git에 포함되지 않습니다(`experiments/minseon/error_description_pilot/.gitignore`).
