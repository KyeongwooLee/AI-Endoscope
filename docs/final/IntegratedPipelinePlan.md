# GSM8K·MATH 오류 탐지 → 분포 유형 예측 → 유형별 hidden 교정 설계

## 1. 목표와 연구 가설

Paper.md의 반복 풀이 기반 분류와 RLforCorrection.md의 reward gradient 개입을 연결한다.

학습 흐름

문제별 독립 풀이 30회 → 응답 분포 클래스 산출 → 오류 탐지 probe 및 클래스별 probe 학습 → 클래스별 prefix 정답 도달률 reward head 학습.

추론 흐름

최초 풀이 → 오류 위험 탐지 → B/C/D/E 확률 예측 → 예측 확률로 reward head 결합 → 풀이 재생성 중 hidden gradient 교정 → 최종 답 반환.

연구 질문은 다음 세 가지다.

1. hidden에서 최초 응답의 오류 여부와 반복 응답 분포를 예측할 수 있는가?
2. 유형별 reward head가 단일 reward head보다 교정에 도움이 되는가?
3. 그 이득이 추가 생성·재계산 비용을 고려해도 유지되는가?

B/C/D/E는 모델·프롬프트·샘플링 조건에 종속된 응답 행동 유형이다. 관계 오류·계산 오류 같은 원인 라벨로 해석하지 않는다. 내부 표현의 오류 정보와 유형별 완화라는 배경은 Orgad et al. (https://arxiv.org/abs/2410.02707)에 근거하되, 아래의 통합 구조는 새 연구 설계로 명시한다.

사용자 선택에 따라 다음을 확정한다.

- 분포 클래스별 정답 가능성 reward head를 사용한다.
- prefix에서 실제 후속 풀이를 생성해 reward 학습 목표를 만든다.
- 기반 LLM 파라미터는 고정한다. 이번 범위는 지도학습된 reward에 따른 추론 시 hidden 교정이다. 모델을 RL 학습하는 SCoRe (https://arxiv.org/abs/2409.12917)나 InT (https://arxiv.org/abs/2601.14209)의 재현으로 부르지 않는다.

## 2. 구현 기반과 데이터 계약

### 기존 코드 재사용

구현 위치는 AI-Endoscope 저장소의 새 패키지 experiments/integrated_correction으로 한다. 기존 실험을 수정하거나 브랜치 전체를 병합하지 않고 필요한 로직을 이식한다.

| 확인한 소스 | 재사용할 내용 | 반드시 변경할 내용 |
|---|---|---|
| minsu 브랜치 GSM8K pilot | 반복 생성, exact token 저장, q_end·r_mean 추출, 문제 단위 probe 평가 | MATH 지원, 안정적인 ID, train/test 격리, 엄격한 답 파싱 |
| 로컬 research_base | manifest, 해시·재개 검증, residual hook, 토큰과 logits 정렬 검사 | 전체 레이어 activation 저장 대신 필요한 레이어와 집계만 저장 |
| sunghun 브랜치 데이터 도구 | 기존 release 로딩·무결성 검사 방식 | 새 데이터 계약으로 변환하고 호환 조건 검사 |

로컬에서 확인한 참조 commit은 minsu=d95116f0d8a6b64a7f3eb08457af11144d6f4a7c, sunghun=99b35512ba043b0ec86d809ec700fcb4ba969889다. 구현 시 이 commit을 기준으로 소스 provenance를 기록한다.

확인한 브랜치에는 RLforCorrection.md의 최신 MATH 교정 구현·체크포인트가 없었다. 문서의 실험 수치와 임계값을 재사용 가능한 모델 파일로 취급하지 않는다. prefix rollout, reward 학습, hidden 개입 엔진은 새 구현 대상이다.

### 데이터와 생성 조건

- GSM8K: openai/gsm8k, main (https://huggingface.co/datasets/openai/gsm8k).
- MATH: EleutherAI/hendrycks_math (https://huggingface.co/datasets/EleutherAI/hendrycks_math)의 7개 subset을 모두 합친다. MATH-500으로 대체하지 않는다.
- 모델: Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306.
- GSM8K revision: 기존 설정의 740312add88f781978c0658806c59bc2815b9866. MATH revision은 준비 단계에서 실제 SHA를 해석해 manifest에 고정한다.
- 최초 응답은 greedy, 반복 라벨링은 temperature=0.7, top_p=0.95, 정확히 30회다.
- 최대 새 토큰은 GSM8K 1,024, MATH 2,048로 고정한다. 잘린 응답을 성공으로 처리하지 않는다.
- GSM8K는 기존 pilot의 단계별 풀이 및 #### <number> 프롬프트를 그대로 가져온다.
- MATH는 같은 단계별 풀이 지시를 사용하되 최종 줄은 Final answer: \boxed{<answer>}로 변경한다.
- RNG seed는 전역 seed 42와 (problem_id, generation_role, sample_index)의 SHA256으로 결정한다. Python의 프로세스별 hash()를 사용하지 않는다.

원본 데이터가 연결된 release에 있으면 먼저 재사용한다. 생성 응답은 모델 revision·프롬프트·생성 조건·정확한 토큰·seed provenance가 모두 일치할 때만 재사용한다. 맞지 않는 기존 응답은 보조 분석에만 쓰고 주 실험용으로 재생성한다.

### 분할과 식별자

공식 train을 문제 단위로 fit/tune/calibration=80/10/10으로 나누고 공식 test는 최종 평가 전용으로 유지한다. MATH는 subset·level별로 분할한다.

정규화한 문제 텍스트가 완전히 같은 항목은 동일 그룹으로 묶는다. 공식 test와 중복되는 train 항목은 train에서 제거하고 기록한다. 근사 템플릿 중복까지 제거했다고 주장하지 않는다.

- problem_id: 데이터셋·subset·공식 split·원본 행 식별자를 포함한다.
- response_id: problem_id + role + sample_index.
- role: label_bank, initial, feature_sample, continuation, corrected.

한 문제의 모든 응답과 prefix는 같은 분할에 속한다. 결합은 ID로 수행하고 행 순서를 사용하지 않는다.

최소 저장 계약은 다음과 같다.

| 산출물 | 필수 내용 |
|---|---|
| 문제 | ID, 원본 위치, 질문, gold, 중복 그룹, 분할 |
| 응답 | ID, 역할, seed, prompt/output token IDs, 텍스트, 종료 원인, 파싱 결과, 정오 |
| 분포 라벨 | 30개 응답 ID, 정답·오답별 빈도, 실패 상태별 빈도, 다중 라벨 |
| 특징 | 응답 ID, 레이어·hook 정의, pooling, tensor 위치·해시 |
| prefix rollout | 원 응답 ID, prefix 길이, 후속 생성 seed·응답 ID, 성공 수/시도 수 |
| 교정 로그 | 위치, 예측 클래스 확률, head 가중치, reward 전후, gradient·변위 크기, 채택 여부, 비용 |
| manifest | 소스 commit, 모델·데이터 SHA, 환경, 설정, 입력·출력 해시 |

## 3. 반복 응답 분류와 probe 학습

### 클래스 정의

라벨링용 30회는 최초 응답 및 추론 특징 수집용 응답과 독립적으로 생성한다. 추론 입력에 사용한 풀이가 자기 라벨을 만드는 효과를 방지한다.

n_correct는 정답 수, n_wrong(a)는 정규화된 오답 a의 빈도, n_unique는 정상 추출된 서로 다른 답의 수다.

| 클래스 | 정확한 조건 |
|---|---|
| A | 명시적 답변 거부 15회 이상 |
| B | n_correct >= 15 |
| C | max_a n_wrong(a) >= 15 |
| D | 어떤 오답 a에 대해 n_correct >= 5, n_wrong(a) >= 5, 두 빈도 차이가 5 이하 |
| E | n_unique > 10 |

- B/C/D/E는 독립 boolean인 다중 라벨이다. 우선순위를 주어 하나로 축약하지 않는다.
- B1=n_correct==30, C1=동일 오답 30회, E1/E2는 정답 출현 여부에 따라 기록하되 주 probe 대상은 B/C/D/E다.
- 어느 조건에도 해당하지 않는 문제는 unclassified로 보존한다.
- A는 메타데이터와 평가에 포함하지만 주 교정 expert에는 포함하지 않는다.
- 거부 판정은 응답 시작부의 고정된 명시적 거부 표현 목록으로 수행하고 목록·판정기를 버전 관리한다. 답 누락이나 truncation을 거부로 바꾸지 않는다.
- 파싱 실패·잘림은 정오 평가에서 실패로 계산하되 동일한 오답 하나로 묶거나 n_unique에 넣지 않는다.

GSM8K는 마지막 #### 뒤 숫자를 Decimal로 비교한다. 본문 마지막 숫자로의 fallback은 금지한다. MATH는 최종 boxed 답을 추출하고 Math-Verify (https://github.com/huggingface/Math-Verify)의 고정 버전으로 동치성을 검사한다. 답끼리 군집화할 때도 동일 검증기를 사용하고, 먼저 생성된 대표 답과 순서대로 비교하는 결정적 군집화를 사용한다. timeout·판정 불가는 별도 상태로 보존한다.

### 특징과 학습 대상

주 실험의 추론 특징은 최초 greedy 응답 하나, 즉 K=1이다. K=5는 greedy 1개와 독립 sampled 응답 4개의 응답별 평균 hidden을 다시 평균하는 사전 지정 보조 실험이다.

레이어는 20번째 decoder block 출력 residual, 즉 model.model.layers[19]의 최종 normalization 전 출력으로 고정한다. hidden_states[20]과의 대응을 테스트로 확인한다.

- q_end: 프롬프트 마지막 토큰 hidden.
- r_mean: final-answer 줄과 EOS를 제외한 풀이 토큰 hidden 평균.
- 풀이가 비어 있으면 q_end를 사용하고 empty_reasoning=true를 남긴다.

학습 모델은 다음처럼 분리한다.

1. 오류 탐지 probe: 응답별 incorrect=1 예측.
2. 클래스 probe 4개: 문제별 B/C/D/E 여부를 각각 예측.
3. 비교 baseline: 질문만의 q_end, confidence+length, 응답 TF-IDF.

기본 probe는 train에서만 학습한 StandardScaler → LogisticRegression이다. C ∈ {0.01, 0.1, 1, 10}, class_weight=balanced 중 tune AUROC로 선택한다. 동률이면 작은 C를 선택한다. scaler부터 추론까지 저장하고 reward 계산과 무관하게 probe를 고정한다.

calibration 분할에서 sigmoid calibration을 적용한다. 클래스 probe는 다중 라벨이므로 softmax를 사용하지 않는다. 클래스별 train 양성·음성 문제 중 하나라도 30개 미만이면 해당 probe는 비활성화하고 이유를 기록한다.

오류 탐지 임계값은 calibration에서 **전체 문제 중 예상 오류 확률 상위 25%**에 해당하는 값으로 고정한다. test에서는 순위를 다시 계산하지 않고 그 임계값을 적용한다. test의 실제 개입 비율을 별도로 보고한다.

30회 라벨 은행과 gold는 학습·사후 평가에만 접근할 수 있다. 추론 함수에는 전달하지 않는다.

## 4. 클래스별 reward와 hidden 교정

### reward가 학습할 목표

클래스 probe의 P(C)를 낮추는 gradient를 교정 방향으로 사용하지 않는다. 문제의 행동 유형은 줄여야 할 오류 보상이 아니다.

별도로 다음을 학습한다.

\[
R_c(h_t)\approx P(\text{정답으로 완료}\mid \text{prefix}_{\le t}, c),
\quad c\in\{B,C,D,E\}
\]

모든 문제를 사용하는 global head R_global도 학습한다. 각 클래스 head는 그 클래스에 실제로 속한 fit 문제만 사용한다. 다중 라벨 문제는 여러 head에 참여할 수 있다.

### prefix 후속 생성

- 문제별로 최초 greedy 응답과 독립 sampled 응답 1개를 prefix 원본으로 사용한다.
- 각 원본에서 생성 토큰 길이의 10~80% 지점 8개를 선택한다. 위치는 4의 배수로 내림하고 중복·빈 prefix는 제거한다.
- 각 prefix에서 독립 후속 풀이 4개를 T=0.7, top_p=0.95로 생성한다.
- 최대 토큰은 원본과 같은 총 생성 길이 제한의 남은 부분이다.
- 타깃은 정답 후속 풀이 수 / 4다. 원본 완성 응답의 정오를 prefix 전체에 전파하지 않는다.
- prefix의 마지막 토큰 residual만 reward 입력으로 사용한다. 미래 토큰이나 gold를 입력에 포함하지 않는다.

정답 여부와 관계없이 원본 응답을 사용하고, 문제마다 전체 loss 가중치를 동일하게 한다. fit/tune/calibration의 후속 생성도 각 분할 안에서 독립 수행한다.

head는 1536 → 128 → 1, GELU와 sigmoid를 사용하는 MLP다. soft-target BCE, AdamW, 학습률 1e-3, weight decay 1e-4, 최대 30 epoch, tune loss 기준 patience 3으로 학습한다. 출력은 calibration에서 logistic calibration한다.

클래스 head의 fit 문제가 30개 미만이거나 전체 후속 생성 결과가 모두 성공 또는 모두 실패면 비활성화한다. C1 등에서 이런 현상이 발생해도 성공 데이터를 인위적으로 만들지 않는다.

### 클래스 확률과 reward 결합

최초 응답에서 얻은 클래스 확률 p_c는 재생성 동안 고정한다.

활성 head 중 p_c >= 0.5인 head를 선택하고 다음 가중치를 사용한다.

w_{c}=p_{c}/    ∑    pⱼ,  R(h)= ∑ w_{c}R_{c}(h)
            j∈ active           c

선택된 head가 없으면 global head를 사용한다. gradient가 없거나 비정상이면 해당 블록은 원본대로 유지한다. 활성화된 head 수와 fallback 비율을 보고한다.

### 개입 알고리즘

최초 응답의 오류 위험이 임계값 이상인 문제만 프롬프트부터 다시 생성한다. 최초 응답은 항상 보존한다.

참조 구현은 Transformers의 직접 forward와 hook을 사용하며 KV cache를 사용하지 않는다. rollback마다 전체 prefix를 다시 계산해 캐시 오염을 방지한다. 속도 최적화는 참조 구현과의 동등성 검증 이후 별도 작업으로 둔다.

기본 설정은 검사 간격 t=4, 최대 개입 32회, 블록당 재시도 1회다.

1. 현재 확정 prefix에서 무개입 greedy 토큰 4개를 임시 생성한다.
2. 임시 블록 끝의 residual로 R_end를 계산한다.
3. R_end가 개입 임계값 이상이면 블록을 확정한다.
4. 낮으면 임시 블록을 버리고 블록 시작 prefix로 돌아간다.
5. 시작 prefix의 마지막 토큰 residual h에서 g=∇h log(R(h)+1e-6)를 구한다.
6. 아래 변위를 hook으로 주입하여 대안 블록을 생성한다.

                  g
δ= α RMS(h)  ───────────
             RMS(g)+10⁻⁸

7. 대안 블록의 끝 reward가 무개입 블록보다 높을 때만 대안을 채택한다. 동률이면 무개입 블록을 채택한다.
8. 채택된 prefix에서 계속 생성한다. EOS가 나오면 남은 블록 생성을 중단한다.

변위는 해당 forward의 마지막 위치에만 적용한다. 대안 블록의 첫 토큰 logits와 이후 토큰 계산이 개입된 시작 residual의 영향을 받도록, 대안 블록을 계산하는 각 full-prefix forward에서 같은 시작 위치에 고정된 δ를 재주입한다. 블록 확정 후에는 토큰만 남기고 변위를 유지하지 않는다.

LLM·head 가중치는 모두 고정하고 residual에 대해서만 autograd를 켠다. probe 표준화·reward 전처리는 미분 가능한 Torch 연산으로 구현한다.

tune에서 α ∈ {0.02, 0.1, 0.4}, reward 임계값 ∈ {0.3, 0.5, 0.7}의 9개 조합을 평가한다. 전체 문제 최종 정확도가 높은 조합을 선택하고, 동률이면 낮은 실제 forward 처리 토큰 수, 작은 α, 낮은 임계값 순으로 선택한다.

문서의 기존 α·임계값과 정규화 정의가 다를 수 있으므로 기존 수치를 그대로 재현했다고 표현하지 않는다. reward 상승은 채택 기준일 뿐 정답 개선의 증거는 아니다.

## 5. 실행 인터페이스와 검증

새 CLI는 다음 순서로 실행할 수 있게 한다. 모든 단계는 같은 --config와 --run-dir를 받는다.

prepare
generate-label-bank
label-distributions
generate-features
train-probes
generate-prefix-rollouts
train-rewards
tune-correction
evaluate
report

- 각 단계는 입력 해시와 설정이 같을 때만 --resume을 허용한다.
- 공식 test 평가는 설정을 잠근 뒤 수행한다.
- correct(problem, probe_bundle, reward_bundle, config)는 gold·30회 응답·실제 클래스 라벨을 받지 않는다.
- evaluate만 gold를 읽어 최종 응답을 채점한다.
- 주 실행 환경은 Linux·단일 CUDA GPU·BF16 모델·FP32 head로 한다.
- 전체 activation 대신 집계 특징과 선택된 prefix residual만 저장한다. 생성 시 정확한 토큰은 모두 보존한다.

### 비교 실험

동일 test 문제와 최초 응답을 공유해 다음을 비교한다.

| 조건 | 확인하려는 효과 |
|---|---|
| 최초 greedy만 사용 | 원래 정확도 |
| 위험 탐지 후 generic 재풀이 | 재생성 자체의 효과 |
| 위험 탐지 후 단일 global reward 교정 | 유형 분리 없는 gradient 효과 |
| 예측 클래스별 reward 교정 | 제안 파이프라인의 순효과 |
| 실제 클래스별 reward 교정 | 유형 예측 오차를 제거한 oracle 진단 |
| 문제 간 클래스 확률을 섞은 reward 교정 | 유형에 맞춘 연결의 실질적 가치 |
| K=5 majority 및 probe vote | 기존 재샘플링 방식 대비 효율 |

oracle 조건은 최종 클래스 라벨을 사용하는 사후 분석이며 배포 가능한 성능으로 보고하지 않는다. 유형 확률 섞기는 데이터셋 안에서 고정 seed로 수행한다.

정확도와 함께 실제 forward 처리 토큰 수·생성 토큰·시간·최대 GPU 메모리를 기록한다. 고정 K 비교와 별도로 같은 누적 forward 토큰 예산에서 비교한다. FLOPs는 profiler로 측정한 경우만 사용한다.

### 필수 지표와 테스트

- 오류 탐지: AUROC, AUPRC, calibration, 개입 비율.
- 클래스 예측: 클래스별 양성/음성 문제 수, AUROC·AUPRC·F1. 전체와 최초 오답 하위집합을 각각 보고한다.
- reward: Brier score, BCE, calibration. 후속 결과의 성공률을 기준으로 평가한다.
- 교정: 전체 정확도, wrong→right, right→wrong, 순증가, 개입당 복구율, 비용.
- 데이터셋별 결과를 먼저 제시하고 통합 결과는 보조로 둔다.
- 신뢰구간은 문제 단위 paired bootstrap 2,000회, seed 42로 산출한다.

구현 완료를 위한 필수 검증은 다음과 같다.

1. 클래스 경계 14/15, D의 5/6, E의 10/11, 중복 클래스·미분류 사례.
2. 파싱 실패·거부·잘림을 구별하고 MATH 수식 동치성을 검사하는 사례.
3. label bank·특징 응답·continuation 간 역할 및 seed 독립성.
4. 문제·prefix의 분할 누출과 ID 중복이 없다는 검사.
5. residual과 다음 토큰 logits의 한 토큰 차이, 20번째 block 대응 검사.
6. α=0에서 참조 무개입 생성과 토큰이 동일하다는 검사.
7. finite-difference와 autograd gradient 방향 일치, 변위 크기 검사.
8. rollback 후 stale cache·미래 토큰·gold 접근이 없다는 검사.
9. 비활성 head·빈 풀이·EOS·비정상 gradient·개입 횟수 제한 처리.
10. GSM8K·MATH 각 2문제의 실제 모델 smoke 실행 후, 데이터셋별 100문제 pilot을 거쳐 전체 실행.

기능 검증 통과와 연구 가설 지지는 별개다. global reward나 재풀이보다 개선되지 않거나 비용 증가만 나타나도 그대로 결과에 포함한다.

## 6. 인계 산출물과 범위

구현 agent는 위 명세를 docs/final/IntegratedPipelinePlan.md에 보존하고 다음을 제공한다.

- 단계별 CLI와 고정 config.
- 소스·데이터·모델 provenance manifest.
- 클래스 분포 및 누락·실패 목록.
- probe·reward 체크포인트와 비활성화 사유.
- 문제별 교정 로그 및 원본/교정 응답.
- 전체·데이터셋별·유형별 결과와 비용 비교 보고서.
- 소규모 재현 명령과 필수 테스트 결과.

이번 설계에서는 SAE, LLM Judge 원인 라벨링, 기반 모델 RL/GRPO 학습을 추가하지 않는다. 기존의 관계·상태·계산 오류 head를 B/C/D/E head로 이름만 바꾸어 재사용하지 않는다.

현재 단계에서는 계획만 제시했으며 파일 작성·모델 학습·GPU 교정 실험은 수행하지 않았다. 최신 MATH 교정 코드·가중치의 부재와 full-prefix 재계산 비용이 구현 시 주요 제약이다.