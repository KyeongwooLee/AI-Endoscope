# Qwen2.5 / GSM8K Error-Risk Experiments

Qwen2.5-1.5B-Instruct가 직접 생성한 답변을 정답/오답으로 나누고, 출력 분포와
중간 layer 표현으로 최종 오답 위험을 예측할 수 있는지 조사한 실험 코드다.
**Qwen 가중치는 고정했으며, 학습한 것은 작은 classifier/probe뿐이다.**

## 문서 읽기 순서

전체 MD 문서별 내용과 목적별 추천 경로는 [브랜치 첫 README의 읽기 안내](../../README.md#처음-읽는-팀원을-위한-안내)에 있다.
처음에는 이 문서의 타임라인·실제 결과를 읽고, 데이터 분석은 [DATA_README.md](DATA_README.md),
실행 재현은 [REPRODUCE.md](REPRODUCE.md)로 이동하면 된다.
개별 실험의 당시 설계는 아래 Experiment Order 표의 문서 링크를 참고한다.

### 후속 오류 유형 주석 데이터

[오류 유형 주석 README](error-taxonomy-full-v1/README.md)에 후속 sampling 실험의
GSM8K 896개 + SVAMP 204개 답변을 assistant가 직접 분류한 방법과 필드, 유형별 집계를 정리했다.
**모두 인간 검수 전 초안**이며 독립적인 의미 검증이나 라벨 정확도 평가를 거치지 않았다.
[전용 prerelease](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft)에서
질문·응답·reference·주석을 받을 수 있다. 기존 7,473문제 feature Release와 생성 경로/응답이 다르므로 구별해야 한다.
같은 오류 유형 주석 릴리즈에 **GSM8K sampling 4,000응답과 SVAMP 1,000응답 전체·feature**도 추가했다.
[통합 다운로드 안내](error-taxonomy-full-v1/DOWNLOAD_GUIDE.md)를 참고한다. 주석은 그중 1,100개 오답에만 있다.

[SVAMP inference·양방향 전이 데이터 안내](posthoc-transfer-data-v1/README.md)와
[전용 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/posthoc-transfer-data-v1)에는
SVAMP 1,000응답, 별도 GSM8K sampling 4,000응답의 완료 feature와 양방향 verifier 결과를 공유한다.
전체 train 7,473응답을 축소한 것이 아니며, 오류 유형 주석 1,100개와는 응답 ID/hash로 연결할 수 있다.

### 추가 오답 4,000개 수집 (2026-10-29)

[추가 수집 안내](extra-errors-20261029-v1/README.md)와 [전용 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/qwen-extra-errors-20261029-v1)에
새 Qwen 응답 18,656개, 선택 오답 4,000개(GSM8K 3,188 / SVAMP 812), 해당 오답의 상세 feature를 추가했다.
기존 실행·AI 주석을 보존한 별도 수집이며 **추가 4,000개의 오류 유형은 아직 미분류·인간 미검수**다.
고유 질문은 2,717개이며 유형별 균형 표본이나 새 미사용 test가 아니다.

## Contents

- `gsm8k-logit-study/`: 실제 사용한 연구 코드, 설정, 설계 문서, 테스트.
- `work/`: 실제 입력 패키징, detached 실행, 백업, 그림 생성에 사용한 보조 코드.
- [REPRODUCE.md](REPRODUCE.md): 설치, 실행 순서, 필요한 외부 데이터와 입력 형식.
- [DATA_README.md](DATA_README.md): 전체 train 데이터/feature 사전, AI 전달법, 누수 방지, CPU 로딩.
- [DATA_SCHEMA.json](DATA_SCHEMA.json): 기계 판독용 배열/필드/feature 정의.
- `data_tools/`: 원본을 변경하지 않는 Release 패키징·검증·로딩 도구와 테스트.
- `source_manifest.json`: 업로드한 코드와 원본의 해시 대조 기록.

생성 답변, feature 배열, 학습된 probe, 모델 가중치, 환경/cache, SSH 키는 포함하지 않는다.
전체 train 7,473개의 생성 답변과 compact feature는
[별도 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1)에서 받는다.
`AI_STARTER.zip`은 팀원/AI용 작은 설명·샘플·scalar feature 묶음이다.
초기 study의 전체 logit-lens 자료나 학습된 probe는 이 데이터 Release 범위에 포함하지 않는다.

## Fixed Protocol

| 항목 | 설정 |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dataset | `openai/gsm8k`, `main` |
| Dataset revision | `740312add88f781978c0658806c59bc2815b9866` |
| Device | RTX 3090 24GB 한 장 |
| Inference | FP32, SDPA, TF32 disabled, greedy, batch size 1 |
| Length budget | Prompt 3,072 / generated 1,024 / total 4,096 tokens |
| Label | 파싱된 최종 숫자를 gold 숫자와 비교; 파싱 실패는 오답, truncation은 별도 기록 |
| Early checkpoint | `t=0,32,64,128`; t는 이미 생성한 답변 토큰 수 |
| Raw vectors | final RMSNorm 적용 전 `h7,h10,h13,h14` |

실제 revision을 고정한 설정은
[`configs/rtx3090.fp32-pinned.json`](gsm8k-logit-study/configs/rtx3090.fp32-pinned.json)이다.
원본 `rtx3090.json`은 prepare 시 revision을 해석하는 초기 설정으로 보존했다.
정답 풀이를 teacher forcing하는 실험이 아니라, 모델 자신의 생성 token ID 경로를 causal replay한다.

## Experiment Order

| 순서 | 실험 | 코드 / 문서 |
|---|---|---|
| 1 | 자체 생성 답변의 정답/오답별 logit-lens 통계 비교 | `gsm8k_lens/`, [실행 안내](gsm8k-logit-study/README.md) |
| 2 | 답변 전체 confidence / internal / length로 사후 classifier | `scripts/train_error_classifier.py`, [설계](gsm8k-logit-study/CLASSIFIER.md) |
| 3 | layer 14까지만 사용하는 causal-prefix scalar classifier | `scripts/train_midlayer_classifier.py`, [설계](gsm8k-logit-study/MIDLAYER_CLASSIFIER.md) |
| 4 | h14 / [h14, h14-h13] / [h7,h10,h14] linear·MLP probe | `scripts/extract_hidden_probes.py`, `scripts/train_hidden_probes.py`, [설계](gsm8k-logit-study/HIDDEN_PROBES.md) |
| 5 | layer별 절대·상대·방향 변화량과 1/x 가설 분석 | `scripts/analyze_hidden_trajectory.py` |
| 6 | 구조·변화·entropy 안정성 feature ablation | `scripts/prepare_feature_ablation.py`, `scripts/train_feature_ablation.py`, [설계](gsm8k-logit-study/FEATURE_ABLATION.md) |
| 7 | 동결한 entropy 후보를 새로운 train 문제 1,000개에서 확인 | `scripts/confirm_entropy_features.py`, [설계](gsm8k-logit-study/ENTROPY_CONFIRMATION.md) |
| 8 | 공식 train 7,473개 전체의 compact feature 추출 | `scripts/extract_train_all.py`, [설계](gsm8k-logit-study/TRAIN_ALL_EXTRACTION.md) |

모든 `scripts/` 경로는 `gsm8k-logit-study/`를 기준으로 한다.
2차 실험의 confidence는 최종 layer 28의 entropy, margin, 생성 토큰 log-probability,
하위 10% log-probability 통계 묶음이다. 길이는 prompt를 제외한 생성 답변의 토큰 수다.
2차 실험의 internal에는 최종 layer와 비교하는 feature도 있어, 조기 사용 가능한 feature와 같지 않다.
3차 실험부터는 입력 시점과 layer 경계를 제한한다.

## Verified Findings

- 최초 고정 primary인 reasoning / layer-14 mean JS-to-final의 오답-정답 집단 차이는
  exploration과 test 모두 bootstrap 95% CI가 0을 포함했다.
- 사후 confidence+length logistic classifier의 test AUROC는 약 0.795였다.
  그 위에 internal을 추가했을 때의 개선은 확립되지 않았다.
- t32 raw hidden MLP test AUROC: h14 0.607, h14+adjacent delta 0.615,
  multi-depth 0.572. delta MLP의 h14 대비 개선 CI는 0을 포함했다.
- 첫 답변 토큰 생성 전 h14 MLP의 보조 test AUROC는 약 0.700이었다.
  문제 난이도나 모델의 문제별 취약성을 읽는 신호일 수 있다.
- layer 변화량은 단순한 단조 감소 1/x 형태를 지지하지 않았다.
- 탐색에서 선택한 entropy 후보는 새 문제 1,000개에서 AUROC 약 0.58~0.60으로 약해졌다.
  같은 평가의 frozen initial h14 MLP는 약 0.643이었다. paired 차이는 유의하지 않았다.

이 결과들은 각기 다른 입력, split, 대상군에서 얻은 것이므로 순위를 직접 비교하지 않는다.

## Full Train Extraction: Completed 2026-10-10

| 항목 | 결과 |
|---|---:|
| Official train problems | 7,473 |
| Correct | 6,091 (81.5068%) |
| Incorrect | 1,382 (18.4932%) |
| Parse failures | 78 |
| Truncated | 16; 모두 위 parse failures에 포함 |
| Parsed, nontruncated incorrect | 1,304 |
| Newly generated responses / incorrect | 5,345 / 994 |
| Reused historical responses | 2,128 |
| t0 / t32 / t64 / t128 available | 7,473 / 7,467 / 7,446 / 5,865 |
| Replay checkpoint argmax agreement | 28,251 / 28,251 |
| Verified backup batches | 234 |
| Batch ZIP bytes | 730,618,272 |

이 단계에서는 공식 test 1,319개를 로드하지 않았다. 전체 데이터 백업과 해시 검증은 완료됐지만,
**이 확대 데이터로 probe를 재학습한 결과는 아직 없다.**

## Interpretation Boundaries

- 예측 대상은 최종 답변의 오답 여부이지, 추론 오류가 처음 생긴 token이나 원인은 아니다.
- t0 표현은 생성 전의 문제 신호다. t32 신호도 난이도와 생성 경로 신호를 함께 포함할 수 있다.
- 과거 test는 이미 여러 실험에서 확인했다. 최종 untouched holdout으로 주장하지 않는다.
- 전체 train에는 과거 fit/tuning/confirmation 문제가 포함된다. 다음 학습 전에
  normalized-question 단위 split을 새로 고정하고, 과거 holdout 재사용을 명시해야 한다.
- weighted BCE 점수와 logit-lens entropy는 calibrated 정답 확률이 아니다.
- 서로 다른 t에서 답변이 끝난 문제는 관측되지 않는다. 대상군 변화와 성능 변화를 구분한다.
- 이전 token들은 전체 28개 layer로 생성됐다. 중간 probe가 곧 절반 계산량을 뜻하지 않는다.
- 같은 문제의 여러 sampling 경로 비교와 compute-matched 개입 실험은 아직 수행하지 않았다.
- 위험도 예측이 실제 정답률 향상을 입증한 것은 아니다.

## Source Preservation

`gsm8k-logit-study/`의 원래 코드·설계 파일은 바이트 그대로 복사했다.
추가한 pinned config와 GPU용 dependency 안내는 별도 파일이다.
`.gitattributes`는 원본 코드의 줄바꿈 변환을 막는다. resume fingerprint가 소스 해시를 포함하므로
과거 run에 변경한 코드를 섞지 말고, 기능을 수정하면 새 output directory에서 실행해야 한다.
보조 백업 도구 두 개만 공개본에서 SSH 실행 경로·서버 주소·로컬 저장 경로를 CLI 인자로 바꿨다.

업로드 전 검증: 2026-10-12 CPU 환경에서 offline unit tests 48개가 모두 통과했다.
이번 업로드를 위해 Qwen inference나 GPU probe 학습을 다시 실행하지는 않았다.
