# 오류 주석 + 전체 생성 데이터 다운로드 안내

[이 릴리즈 한 곳](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft)에서
**GSM8K sampling 4,000응답, SVAMP 1,000응답, 오류 유형 주석 1,100개**를 모두 받을 수 있다.
이 안내는 전체 데이터 자산을 기존 릴리즈에 추가하면서 작성했다. 기존 주석 데이터는 변경하지 않았다.

## 무엇을 받으면 되는가

| 목적 | 받을 파일 |
|---|---|
| GSM8K 전체 sampling 데이터 | [gsm8k-sampled-posthoc-v1.zip](https://github.com/KyeongwooLee/AI-Endoscope/releases/download/error-taxonomy-full-v1-draft/gsm8k-sampled-posthoc-v1.zip), 약 72.5MB |
| SVAMP 전체 데이터 | [svamp-inference-v1.zip](https://github.com/KyeongwooLee/AI-Endoscope/releases/download/error-taxonomy-full-v1-draft/svamp-inference-v1.zip), 약 23.3MB |
| 오류 유형 주석 | [error-taxonomy-full-v1.zip](https://github.com/KyeongwooLee/AI-Endoscope/releases/download/error-taxonomy-full-v1-draft/error-taxonomy-full-v1.zip), 약 1.6MB. 또는 개별 `labels.jsonl` |
| 텍스트를 AI에게 먼저 전달 | `gsm8k_answers.jsonl` 4,000행 / `svamp_answers.jsonl` 1,000행, 이 안내, `ANNOTATION_README.md`, `CODEBOOK.md` |
| 기존 verifier의 예측·전이 결과·작은 probe | `transfer-results-v1.zip`, 약 2.3MB |

Dataset ZIP의 `records.jsonl`에는 실제 질문·응답·reference·생성 token IDs/설정·binary label을,
`features.npz`에는 h28 question/end/mean/number, confidence 26열, length 4열을 보존했다.
텍스트 전용 `*_answers.jsonl`은 `generation` 필드만 뺀 파일로 token IDs와 hidden 배열이 없다.
GSM8K는 **500질문×8응답**, SVAMP는 **1,000질문×1응답**이다. 정답과 오답, 형식 실패를 모두 포함한다.

## 문서 구분

- `ANNOTATION_README.md`, `CODEBOOK.md`, `AUDIT.json`, `MANIFEST.json`: **오류 유형 주석 1,100개**의 기준·필드·검사. 주석 ZIP에도 당시 문서가 있다.
- `DATA_README.md`, `DATA_SCHEMA.json`, `dataset_summary.json`, `RELEASE_AUDIT.json`: **전체 inference 응답 5,000개**와 feature의 계산·필드·집계·검사.
- `SHA256SUMS.txt`: 현재 릴리즈의 모든 배포 파일(체크섬 파일 자신 제외) 검증값. 각 ZIP 내부에도 당시 파일별 검증값이 있다.

## 연결 및 해석

1. Dataset ZIP의 `records.jsonl`과 `features.npz`는 같은 `id` 순서다. `features['ids']`로 확인한다.
2. 오류 주석 `labels.jsonl`을 **응답 단위 `id`**로 연결하고 `response_sha256`도 대조한다. 질문 ID만으로 연결하면 8개 응답을 구분할 수 없다.
3. 5,000응답 중 형식 정상 오답 **GSM8K 896 + SVAMP 204 = 1,100개만 오류 유형 주석 대상**이다. 주석 미존재는 오류 미관찰 라벨이 아니다.
4. Binary `answer_correct`는 gold와 최종 숫자의 자동 비교다. 오류 유형 `math_error_type`은 별도의 AI 주석이며 **모두 인간 검수 전 초안**이다. 무결성 검사 통과가 의미 검수를 뜻하지 않는다.
5. 동일 질문의 응답은 같은 fold에 두고 gold/reference/주석/이전 probe 점수를 새 classifier 입력에 넣지 않는다.

기존에 약 8천 개라고 부른 **GSM8K 전체 train 7,473개 greedy 데이터**는 다른 실험이다.
그 데이터는 기존 `gsm8k-train-features-v1` 릴리즈에 그대로 있으며, 이번 4,000응답으로 줄인 것이 아니다.
같은 질문이어도 응답이 다르므로 기존 7,473개 feature와 이번 주석을 행 순서로 연결하지 않는다.

전체 데이터 ZIP과 전이 결과는 `posthoc-transfer-data-v1` 릴리즈의 동일 이름 파일과 **같은 바이트/해시**다.
기존 오류 주석 ZIP·`labels.jsonl`·CODEBOOK·주석 manifest/audit는 그대로 보존했다.
새 전체 데이터 자산과 이 안내를 추가했고, standalone `ANNOTATION_README.md` 및 릴리즈 설명·전체 체크섬 목록만 갱신했다.
