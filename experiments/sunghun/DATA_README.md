# GSM8K Full-Train Feature Release v1

Qwen2.5-1.5B-Instruct가 **GSM8K 공식 train 전체 7,473문제**를 직접 풀었을 때의
생성 답변, 최종 정답/오답 label, 중간 layer hidden state와 logit-lens 통계다.
통칭한 "8천 개 데이터"의 정확한 수는 7,473개다. **공식 test 1,319개는 포함하지 않는다.**
Qwen은 학습하지 않고 고정했다. 이 데이터의 목적은 작은 probe로 최종 오답 위험을
조기에 예측할 수 있는지 연구하는 것이며, 7,473개로 새 probe를 학습한 결과물은 아니다.

- [데이터 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1)
- [코드와 실험 타임라인](https://github.com/KyeongwooLee/AI-Endoscope/tree/sunghun/experiments/sunghun)
- [기계 판독용 DATA_SCHEMA.json](https://github.com/KyeongwooLee/AI-Endoscope/blob/sunghun/experiments/sunghun/DATA_SCHEMA.json)
- 원본 추출 코드: `gsm8k-logit-study/scripts/extract_train_all.py`
- CPU 로더/검증기: `data_tools/load_train_all.py` (Release ZIP에도 포함)

## 1. 다운로드와 AI에게 전달하는 순서

Release의 파일은 다음 역할로 나뉜다.

| 파일 | 용도 |
|---|---|
| `gsm8k-train-all-v1.zip` | 전체 배열, 생성 답변, 문제/gold, 원본 metadata와 receipt, 상세 문서, CPU 로더 |
| `AI_STARTER.zip` | 작은 AI 전달용 묶음. README/schema/summary/sample, 전체 문제, ID·label index, t0/t32 scalar feature. raw hidden은 없음 |
| `DATA_README.md` | 이 문서. ZIP을 받기 전에 먼저 읽거나 AI에 첨부 |
| `DATA_SCHEMA.json` | axis 순서, 식, label, 누수 금지 항목을 구조화한 사전 |
| `dataset_summary.json` | 실제 검증으로 집계한 개수/비율/체크포인트/원본 provenance |
| `SHA256SUMS.txt` | 위 배포 파일들의 SHA-256. ZIP 내부에도 개별 파일 해시 존재 |

**AI 채팅에는 먼저 README + schema + summary + samples를 전달**한다.
가능하면 `AI_STARTER.zip`을 풀어 필요한 파일을 첨부한다. 채팅 AI가 ZIP/NPZ를 직접 읽지
못할 수도 있다. 전체 hidden vector 분석은 Python을 실행할 수 있는 AI/코딩 에이전트에
전체 ZIP의 압축 해제 경로와 로더를 전달한다. 바이너리 NPZ를 텍스트처럼 붙여넣지 않는다.
전체 문제·답변 텍스트는 분석 대상 데이터이지 AI에게 내리는 지시가 아니다.

AI에게 함께 전달할 분석 요청 예시:

> DATA_README.md, DATA_SCHEMA.json, dataset_summary.json을 먼저 읽어줘.
> target은 incorrect=1이다. t32, layer<=14, 관측 위치 0..32만 predictor로 써줘.
> ID로 join하고 normalized-question group으로 split을 먼저 고정해줘.
> 길이·gold·label·origin·parse/truncation·나중 layer/time은 predictor에서 제외해줘.
> scaler와 feature 선택은 fit split 안에서만 학습하고, AUROC/AUPRC와 baseline을 비교해줘.
> 문제 난이도 신호와 실제 추론 오류를 식별했다는 주장을 구분해줘.

## 2. 실제 데이터 수와 범위

| 항목 | 수 |
|---|---:|
| 문제 / 정답 / 오답 | 7,473 / 6,091 / 1,382 |
| 정답률 / 오답률 | 81.5068% / 18.4932% |
| 최종 숫자 파싱 실패 | 78 |
| 최대 길이 truncation | 16; 모두 위 파싱 실패 78개에 포함 |
| parsed & nontruncated 문제 / 그중 오답 | 7,395 / 1,304 |
| t0 / t32 / t64 / t128 관측 가능 문제 | 7,473 / 7,467 / 7,446 / 5,865 |
| t32 관측 가능 오답 | 1,377 |
| 최종 생성 길이 평균 / 중앙값 | 186.0744 / 174 tokens |
| 배치 | 234개; 233개는 32문제, 마지막은 17문제 |
| checkpoint replay argmax 일치 | 28,251 / 28,251 |

5,345문제의 답변은 이번 전체 train run에서 새로 생성했고, 2,128개 답변은 같은 고정
protocol로 생성했던 과거 결과를 재사용했다. 재사용 답변도 이번 compact schema의 feature로
다시 추출했다. origin별 상세 집계는 `dataset_summary.json`에 있다.
`old_pilot=128`, `old_exploration=1,000`, `old_confirmation=1,000`, `new_train=5,345`는
**생성 provenance이지 이번 데이터의 학습/검증 split이 아니다.**

## 3. 모델과 생성 protocol

| 설정 | 값 |
|---|---|
| Model / revision | `Qwen/Qwen2.5-1.5B-Instruct` / `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Dataset / config / revision | `openai/gsm8k` / `main` / `740312add88f781978c0658806c59bc2815b9866` |
| Transformer blocks / hidden size / vocab | 28 / 1,536 / 151,936 |
| 연산 | FP32, SDPA, TF32 disabled, frozen weights, no gradients |
| 생성 | greedy: `do_sample=False`, `num_beams=1`, `repetition_penalty=1`, batch size 1 |
| 길이 상한 | prompt 3,072 / new 1,024 / total 4,096 tokens |
| 종료 | EOS 또는 완성된 `#### <number>` 최종 줄의 newline; 실제 설정은 generation record 참조 |
| seed | generation 42; 전체 train 처리 순서 shuffle 20261009 |
| 완료일 | 2026-10-10 (KST) |

System prompt (정확한 원문):

```text
You are a math problem-solving assistant. Explain your reasoning step by step. You must end every answer with a separate final line formatted exactly as #### followed by a space and the numerical answer. Write nothing after this final line. Do not omit this line. The final answer must contain only a number, without units, currency symbols, Markdown emphasis, or LaTeX.
```

User template (정확한 원문):

```text
Problem:
{question}

Solve the problem step by step. After your reasoning, write the answer on the LAST line using exactly this format:
#### <number>
Replace <number> with the numerical answer. Do not write any text after that line.
```

tokenizer chat template를 적용하고 assistant generation prefix를 붙였다.
문제의 gold 풀이를 teacher forcing한 것이 아니라 **모델이 생성한 token ID 경로를 causal replay**했다.
gold는 label 계산에만 사용했다. 데이터 fingerprint는
`38d173815f312ea20a45d08cf33fe6795b9411e02c155bbd16079c7fa77d1ea4`다.
정확한 config/source hash는 `manifest.json`, Python/package/GPU 정보는 `environment.json`에 있다.
코드·chat template·dtype·revision이 달라진 run을 같은 protocol이라고 섞지 않는다.

## 4. 압축 해제 후 구조

```text
gsm8k-train-all-v1/
  README.md, DATA_SCHEMA.json, dataset_summary.json, samples.json
  problem_index.jsonl, scalar_features_t0_t32.jsonl
  load_train_all.py, checksums.sha256, release_manifest.json
  LICENSE-GSM8K.txt, ATTRIBUTION.md
  manifest.json, questions.json, environment.json, quality_gate.json, progress.json
  input/config.json, input/provenance.json
  results/generation_summary.json, results/labels.json
  outbox/batch_0000.json ... batch_0233.json
  batch_0000/ ... batch_0233/
    features.npz
    generations.jsonl
    summary.json
  extract_train_all.py, gsm8k_lens/*.py, backup_manifest.json
```

`AI_STARTER.zip`은 위 구조 중 작은 설명/파생 feature 파일만 담은 별도 묶음이다.
모든 파일 간 연결은 **문자열 ID**로 한다. `train-00000`의 숫자는 pinned GSM8K train의
zero-based 원본 row index다. `questions.json`은 shuffle된 처리 순서이므로 배열 위치로 join하면 안 된다.
`questions.json` row의 `gold`는 숫자만이 아니라 원본 풀이 전체다.

## 5. t와 layer를 정확히 해석하기

`t`는 **이미 생성해서 context에 넣은 답변 토큰 수**다. t32에서는 prompt와 generated IDs `[0:32]`를
본 상태이며, 아직 zero-based token 32, 즉 사람 기준 33번째 답변 토큰을 생성하기 전이다.
t0는 답변 첫 토큰 생성 전, 마지막 prompt 위치의 표현이다. 질문의 14번째 토큰 같은 뜻이 아니다.
각 checkpoint의 읽는 위치는 `len(prompt_token_ids)-1+t`다.

`h_l`은 l번째 transformer block 출력, **모델 마지막 RMSNorm 전 residual-stream vector**다.
block 번호는 1부터 28까지다. h14는 Python `model.model.layers[13]`의 출력이다.
h0는 embedding 출력이며 이 전체 vector는 저장하지 않았다.

생성 길이 L에 대해 `t < L`일 때만 checkpoint가 있다. L=32면 t32는 없다.
없는 checkpoint를 0으로 채우거나 앞 checkpoint로 대신하지 않는다.
t128의 성능이 t32보다 다르더라도 짧은 답변이 빠진 **대상군 변화**를 고려해야 한다.

## 6. NPZ 배열 사전

`np.load(path, allow_pickle=False)`로 연다. GPU나 Qwen 가중치는 필요 없다.
B=배치 문제 수, C=그 배치의 관측 가능한 checkpoint 총수,
R=`sum(min(129,L))`, D=1536이다. 모든 수치 feature는 float32이고 유한값이다.

| key | shape / dtype | 의미와 axis 순서 |
|---|---|---|
| `ids` | `(C,)`, Unicode | checkpoint row별 문제 ID; 한 문제가 여러 번 등장 |
| `checkpoints` | `(C,)`, int16 | 위 row의 t; 0/32/64/128 |
| `hidden` | `(C,4,1536)`, float32 | layer axis **[7,10,13,14]** |
| `trajectory` | `(C,28,5)`, float32 | layer transition 1..28; 아래 metric 순서 |
| `readout_ids` | `(B,)`, Unicode | 문제당 한 번; ragged readout segment의 ID |
| `readout_offsets` | `(B+1,)`, int64 | 문제 i의 범위 `[offsets[i]:offsets[i+1]]` |
| `readouts` | `(R,3,3)`, float32 | segment 내 위치는 t0부터 연속; layer [7,10,14], metric [entropy,margin,top1_probability] |

`hidden[row,3]`이 h14이고 `hidden[row,2]`가 h13이다. layer 번호를 그대로 axis index로 넣지 않는다.
`readouts`는 checkpoint 4개만 있는 게 아니라 **모든 위치 0..min(128,L-1)**를 저장했다.
문제 i에서 t32까지의 feature는 `readouts[offsets[i]:offsets[i]+33]`이다.

### 6.1 Raw hidden 기반 probe 입력

| 이름 | 구성 | 차원 |
|---|---|---:|
| h14 | `h14` | 1,536 |
| h14 + adjacent delta | `[h14, h14-h13]` | 3,072 |
| multi-depth | `[h7,h10,h14]` | 4,608 |

delta는 같은 문제·같은 t에서의 인접 layer 차이다. 이전 답변 토큰과의 차이가 아니다.
`[h14,h14-h13]`은 `[h13,h14]`와 정보상 동등한 재표현이지 모든 이전 layer를 요약한 값은 아니다.
이 feature들은 새 MLP/linear probe 학습에 사용할 수 있지만, split/scaler는 새로 고정해야 한다.

### 6.2 Layer trajectory 5개 metric

layer l에 대해 a=h_(l-1), b=h_l이며 `trajectory[:,l-1,:]`에 다음 순서로 저장했다.

| metric | 식 | 해석 |
|---|---|---|
| previous_norm | `||a||2` | 이전 layer 벡터 크기 |
| hidden_norm | `||b||2` | 현재 layer 벡터 크기 |
| delta_norm | `||b-a||2` | 절대 변화량 |
| relative_delta | `||b-a||2 / max(||a||2,1e-12)` | 이전 크기로 정규화한 변화 |
| cosine_distance | `1-clamp(dot(a,b)/max(||a||2*||b||2,1e-12),-1,1)` | 방향 변화, 보통 0..2 |

layer 1은 embedding→block1, layer14는 block13→block14의 변화다.
전체 28개 값은 변화 곡선 분석용으로 저장했지만 **중간층 조기 classifier에는 [:14]만 허용**한다.
절대 변화가 크다는 것과 오류 발생은 동의어가 아니다. 기존 관측은 단조 감소하는 1/x 형태를 지지하지 않았다.

### 6.3 Logit-lens readout 3개 metric

중간 h_l에 **원래 모델의 final RMSNorm과 frozen LM head**를 적용한다.
`p_l = softmax(LM_head(final_RMSNorm(h_l)))`이며 전체 vocab에서 FP32로 통계를 계산했다.

| metric | 식 / 단위 |
|---|---|
| entropy | `-sum_v p_l[v]*ln(p_l[v])`; 자연로그 **nats**, bits가 아님 |
| margin | top-1 probability - top-2 probability; **logit 차이가 아님** |
| top1_probability | `max_v p_l[v]`; 다음 토큰 분포의 최대 확률 |

여기서 confidence라는 표현은 위 분포 집중도 통계의 통칭일 뿐,
**최종 답변 정답 확률 또는 calibrated confidence가 아니다.** top1은 실제 생성 토큰의 확률과도 다르다.
이 full-train schema에는 실제 생성 토큰의 log-probability와 최종 layer의 confidence가 없다.
entropy 감소도 정답 보장이 아니다. 층마다 같은 LM head가 잘 맞는 정도가 다를 수 있다.

## 7. 생성 답변과 정답 label

`generations.jsonl`의 한 줄은 한 문제이며 핵심 필드는 다음과 같다.

- `generation`: prompt/generated token IDs, decoded text, 길이, truncation, 종료/생성 설정과 시간.
- `label`: `gold_number`, `predicted_number`, `parse_status`, `answer_correct`, `truncated`, `analysis_eligible`.
- `summary`: ID, origin, incorrect, 길이/파싱 상태, replay quality 진단.
- `fingerprint`: 추출 protocol 식별자. 일부 재사용 generation에는 과거 metadata 추가 필드가 있을 수 있다.

정답 판정은 마지막 **비어 있지 않은 줄** 전체가 `#### <number>` 형식인지 확인하고,
숫자를 Decimal로 변환해 gold 숫자와 정확히 비교한다. 부호, 소수, 올바른 천 단위 comma는 허용하며
단위·수식·분수·지수 표기·뒤쪽 설명은 허용하지 않는다. 상세 regex는 `gsm8k_lens/answers.py`다.
parse_status는 `ok`, `missing`, `invalid_final_line`이다.

`incorrect = int(not answer_correct)`다. 따라서 숫자 오답뿐 아니라 **형식 파싱 실패도 오답**에 포함한다.
truncation은 별도 flag이며 그 자체만으로 강제로 오답 처리하는 코드는 아니다.
이번 실제 16개 truncated 답변은 모두 parse failure라 오답이다.
`analysis_eligible`은 parse ok & nontruncated라는 사후 분석용 조건이다.
이를 이용한 민감도 분석은 가능하지만 deployment 시점에 미리 아는 feature나 필터라고 주장하면 안 된다.
최종 숫자가 맞았다고 풀이의 모든 단계가 논리적으로 옳다는 의미는 아니다.

## 8. AI용 파생 파일

`problem_index.jsonl`은 문제당 한 줄이며 batch 위치, label/길이/parse 상태, available checkpoints,
origin, `question_group_sha256`를 담는다. 대부분은 metadata/target이지 predictor가 아니다.
group hash는 `SHA256(UTF8(' '.join(question.split())))`다. 공백 차이만 정규화하며
의미가 같은 paraphrase까지 중복으로 식별하지는 않는다. 실제 unique group 수는 summary에 있다.

`scalar_features_t0_t32.jsonl`은 문제당 한 줄이다.

```text
{"id":"train-XXXXX","incorrect":1,"features":{"t0":[36개의 숫자],"t32":[36개의 숫자 또는 null]}}
```

위 구조 설명의 배열 내용은 의사 표기다. 실제 파일은 정상 JSON 숫자 배열을 담는다.
각 배열 순서는 layer [7,10,14] → metric [entropy,margin,top1_probability] →
stat [current,mean,std,change_from_t0]의 nested loop다. 완전한 36개 이름은 summary의
`scalar_feature_names`와 loader의 `SCALAR_NAMES`에 있다.
current=r[t], mean=mean(r[0:t+1]), std=population std(ddof=0), change=r[t]-r[0]이다.
t0에서는 std와 change가 0이다. t32를 관측하지 못한 6개 문제는 null이며 0 대체하지 않는다.
이 값들은 기존 float32 readout으로 float64 집계한 **새 배포 편의 view**다.
새 inference, 새 평가 결과, 이전 실험에서 실제 학습한 feature matrix라고 해석하지 않는다.
`samples.json`은 정답/숫자오답/파싱실패 예시와 실제 record를 소량 담아 구조 이해에 사용한다.

## 9. 누수 방지와 새 실험 권장 절차

**t32, layer14 probe의 허용 입력**: t32 raw h7/h10/h13/h14, 같은 시점 layer1..14 trajectory,
위치 0..32에서 layer7/10/14 readouts 및 그 causal 집계.
**금지 입력**: label/gold/최종 숫자/전체 답변/전체 길이/parse/truncation/analysis_eligible,
origin/ID/index/group hash/실행 시간/replay quality, layer15..28, t33 이후 값과 미래 token IDs.
group hash와 ID는 join/split용으로만 사용한다. metadata까지 통째로 NN에 넣으면 안 된다.

1. t와 포함 대상(전체 또는 parsed-only 민감도 분석)을 먼저 정한다.
2. normalized-question group을 공유하지 않는 fit/validation/test split과 seed를 먼저 고정·저장한다.
3. scaler, feature 선택, threshold, hyperparameter 탐색은 fit/validation 안에서만 수행한다.
4. t0 h14, t32 h14, delta, multi-depth, scalar baseline을 **같은 대상군/split**에서 비교한다.
5. AUROC/AUPRC, 오답 prevalence, confusion matrix와 bootstrap CI를 함께 보고한다.
6. 난이도/문제별 취약성 신호와 생성 중 오류 신호를 구분한다. t0 대비 추가 가치를 확인한다.

전체 pool에는 과거 probe fit/tuning/confirmation 문제가 포함됐다. 과거 confirmation을
새 untouched holdout으로 다시 주장하지 않는다. 이전 test도 여러 실험에서 관측됐으므로
완전히 untouched 최종 평가라는 주장을 피한다. 문제 단위 grouped split도 과거 연구자의
적응적 탐색 이력을 지우지는 않는다. 가능하면 신규 dataset/sampling 경로로 외부 확인한다.
같은 문제에서 경로를 바꿔가며 비교하는 sampling/개입 실험은 아직 하지 않았다.
이전 토큰은 전체 28개 layer로 생성했다. 중간 probe를 붙였다고 전체 계산량이 절반인 것은 아니다.

## 10. CPU 로딩과 무결성 확인

```bash
python -m pip install numpy==2.2.6
python gsm8k-train-all-v1/load_train_all.py gsm8k-train-all-v1 --verify --t 32 --kind h14
python gsm8k-train-all-v1/load_train_all.py gsm8k-train-all-v1 --t 32 --kind scalar --output t32_scalar.npz
```

첫 명령은 내부 SHA-256을 확인하고 h14 matrix shape `(7467,1536)`, 정답 6090/오답1377을 출력한다.
kind는 h14/h14_delta/multi/scalar이며 마지막의 scalar는 36차원이다.
`--parsed-only`는 사후 민감도 분석용이다. `--output`은 X/y/ids 배열을 만들 뿐 모델을 학습하지 않는다.
전체 matrix보다 배치별 처리가 필요하면 `iter_examples(root,t=32)`를 사용한다.

```python
import sys
sys.path.insert(0, "gsm8k-train-all-v1")
from load_train_all import feature_matrix, iter_examples
ids, X, y = feature_matrix("gsm8k-train-all-v1", t=32, kind="h14_delta")
# X: (7467,3072), y=1 means incorrect; IDs are for joining/splitting only.
for item in iter_examples("gsm8k-train-all-v1", t=32):
    assert item["readouts_prefix"].shape == (33,3,3)
    assert item["trajectory_early"].shape == (14,5)
    break
```

배포 ZIP은 원본 batch ZIP을 중첩하지 않고, 각 member의 bytes를 보존해 재포장했다.
원본 ZIP과 배포 ZIP의 hash는 다르다. `outbox/*.json`의 archive_sha256은 **원본 batch ZIP**의 hash이며,
그 ZIP 자체는 배포하지 않는다. receipt의 files hash는 배포 중인 같은 이름의 member에도 적용된다.
`backup_manifest.json`은 원본 metadata의 기록이며 새 설명/파생 파일은 열거하지 않는다.
배포 전체는 `checksums.sha256`과 `release_manifest.json`으로 검증한다.

## 11. 저장하지 않은 정보와 라이선스

저장된 raw hidden은 4시점×4layer뿐이므로 전체 token의 hidden pooling이나 임의 시점의 raw 차분은 얻을 수 없다.
전체 vocab logits/probability, 생성 token logprob/rank, JS-to-final/adjacent JS, 최종 layer confidence,
attention, gradient, 모델 가중치, 학습된 probe, 공식 test 출력은 포함하지 않는다.
초기 전체 답변 logit study와 이 Release는 **다른 schema**다. 초기 연구의 feature가 모두 있다고 가정하지 않는다.
재추출 가능성은 필요한 token ID/weights/관측 시점에 따라 달라서, 무엇이든 사후 복원할 수 있는 것은 아니다.

원본 GSM8K는 [OpenAI grade-school-math](https://github.com/openai/grade-school-math) /
[MIT License](https://github.com/openai/grade-school-math/blob/master/LICENSE)에 기반하며,
Copyright (c) 2021 OpenAI와 라이선스 전문을 `LICENSE-GSM8K.txt`에 동봉한다.
이는 GSM8K attribution이며 모델 가중치 배포 허가나 repo 전체의 새 라이선스를 의미하지 않는다.
모델 가중치는 동봉하지 않는다. 생성 답변에는 오류가 있으므로 gold 교사 답변으로 사용하지 않는다.
이 Release는 연구용 관측 데이터이며, 실제 오류 수정이나 정답률 향상을 실증한 것은 아니다.
