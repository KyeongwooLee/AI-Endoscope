# SmolLM2: Compact / Full Feature Data

배포 파일은 [GitHub Releases](https://github.com/KyeongwooLee/AI-Endoscope/releases)에서 확인한다.
**실제 공개 범위와 최신 정오답 집계는 asset의 `RELEASE_SCOPE.md`와 `dataset_summary.json`이 기준**이다.
추론 완료와 원본 백업 완료를 구분하며, 전체 검증 전에는 전체 Release로 공개하지 않는다.

2026-10-22 서버 재시작으로 미백업 `/tmp` 원본이 유실됐다. 해시 검증된
**109배치 / 1,744응답(GSM8K 1,712 + SVAMP 32)**은 그대로 보존하고,
나머지 421배치 / 6,729응답을 재실행하는 복구 protocol을 사용한다.
**재실행분은 유실된 원본 파일 자체의 복원이 아니며** fingerprint/provenance를 구분한다.
Metadata에 남은 과거 숫자 label을 새 응답에 그대로 붙이지 않고 다시 판정한다.

## 데이터 범위

- 모델: `HuggingFaceTB/SmolLM2-1.7B-Instruct`, revision `31b70e2e869a7173562077fd711b654946d38674`.
- Transformer block 24개, hidden dimension 2,048. 모델/probe 학습 없이 inference와 feature 추출만 수행했다.
- GSM8K **공식 train 7,473문제**, 문제당 greedy 응답 1개. 공식 test 1,319문제는 불러오지 않았다.
- SVAMP **원본 1,000문제**, 문제당 sampling 응답 1개. `temperature=0.7, top_p=0.95, top_k=0`.
- RTX 3090, FP32, SDPA, TF32 비활성화, batch size 1. Prompt 최대 3,072 / 생성 최대 1,024 / 합계 최대 4,096 tokens.
- Qwen 데이터와 의미상 같은 system/user prompt에 각 모델의 공식 chat template을 적용했다.
  정확한 prompt 및 설정은 실행 metadata의 `input/config.json`, 실제 token IDs는 generation record에 있다.
- 모델 자신의 생성 경로를 causal replay했다. Gold 풀이를 teacher forcing해서 얻은 feature가 아니다.

### 원래 완료된 실행의 집계

| 데이터 | 응답 | 숫자 정답 | 오답(형식 실패 포함) | 형식 실패 | 길이 제한 도달 |
|---|---:|---:|---:|---:|---:|
| GSM8K train | 7,473 | 3,300 | 4,173 | 425 | 37 |
| SVAMP | 1,000 | 417 | 583 | 36 | 4 |

이는 **유실 전 전체 실행의 metadata 집계**이며, 현재 보존 원본의 수를 뜻하지 않는다.
41개 길이 제한 도달 응답은 모두 형식 실패에도 포함되므로 두 수를 더하지 않는다.
정답률은 GSM8K 44.16%, SVAMP 41.70%다. 이 데이터로 새 verifier를 학습/평가한 결과는 아직 없다.
기존 Qwen 오류 유형 주석이나 Qwen probe 가중치는 이번 데이터의 label/모델이 아니다.

## Compact와 Full

아래는 asset 구성이다. 릴리즈의 실제 공개 파일과 `RELEASE_SCOPE.md` 검증 수를 확인한 뒤 사용한다.

| 파일 | 범위와 목적 |
|---|---|
| `smollm2-gsm8k-compact-v1.zip` | GSM8K train만. 이전 Qwen train Release와 같은 종류의 feature, checkpoint, ragged NPZ 구조 |
| `smollm2-svamp-compact-v1.zip` | 같은 compact 정의의 SVAMP. GSM8K 비교 대상에 섞지 않도록 별도 파일 |
| `smollm2-full-part-NNN.zip` | GSM8K+SVAMP의 이번 추출 결과 전부. 각각 독립 ZIP이며 같은 폴더에 모두 압축 해제 |
| `smollm2-run-metadata.zip` | 과거 원본 실행 설정, 코드, receipts, 과거 label 집계, runtime patch. 새 응답에 과거 label을 붙이지 않음 |
| `smollm2-recovery-metadata.zip` | 누락분 재실행의 고정 코드/입력, 새 fingerprint/receipts/labels, 보존 응답 4개 재현 검사 |
| `AI_STARTER.zip` | 이 설명, schemas, NumPy loader, 문제 index, t0/t32의 36개 scalar, 작은 응답 샘플 |
| `COMPACT_SCHEMA.json`, `FULL_SCHEMA.json` | 배열 축과 필드 사전 |
| `FULL_INDEX.json` | 원본 batch별 part 위치, ID 목록, SHA-256, 크기 |
| `SHA256SUMS.txt` | 다운로드 asset의 SHA-256 |

Compact는 원본 FP32 배열에서 **필요한 축/시점만 선택**한다. 재추론, 양자화, 반올림은 하지 않는다.
Full은 배포 대상 530개 batch ZIP의 **바이트를 그대로** 여러 part에 넣는다. 보존 원본과 명시적 재실행 batch를 함께 포함한다.
GitHub 개별 asset 크기 제한을 피하기 위한
약 960 MiB 단위의 묶음이며, `.part` 파일을 바이너리로 이어 붙이는 형식이 아니다.
유실 전 전체 원본 batch는 10,399,064,394 bytes였다. 재실행 후 Compact/Full의 실제 크기는 완성된 asset에서 확인한다.
Compact와 Full은 같은 응답의 다른 뷰이지, 서로 다른 표본이 아니다.

### Qwen과 무엇이 같은가

비교 기준은 [Qwen GSM8K train features v1](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1)이다.
**feature 정의가 같은 것이지 layer 번호, vector dimension, tokenization이나 관측값이 같은 것은 아니다.**
24층 모델에는 상대 깊이에 대응하는 정수 layer를 사용했다.

| 용도 | Qwen2.5: 28 blocks | SmolLM2: 24 blocks |
|---|---|---|
| Raw hidden | 7, 10, 13, 14 | 6, 9, 11, 12 |
| Prefix logit-lens | 7, 10, 14 | 6, 9, 12 |
| 중간 probe 기준 | h14 | h12 |
| 인접 차이 | h14-h13 | h12-h11 |
| Trajectory | 전체 28 transitions, checkpoint만 | 전체 24 transitions, checkpoint만 |
| Hidden dimension | 1,536 | 2,048 |
| Checkpoint | t=0,32,64,128 | t=0,32,64,128 |

SmolLM2 h6를 h7이라고 재명명하지 않는다. 기존 Qwen probe 가중치를 그대로 연결할 수도 없다.
Tokenizer와 어휘 크기가 달라 raw entropy나 token 길이의 절댓값 비교에도 주의한다.

## Checkpoint와 Compact 배열

`t`는 **이미 context에 들어간 생성 답변 토큰 수**다. `t=0`은 첫 답변 토큰 예측 직전,
`t=32`는 32개 생성 후 33번째 토큰 예측 직전이다. Prompt 길이와 다르다.
응답 길이 `T`에 대해 `t<T`인 checkpoint만 저장하며, 없는 시점을 0/NaN으로 패딩하지 않는다.
Raw state는 해당 위치의 **final RMSNorm 전 block output**이다. t0는 prompt-only forward로 정규화해 저장했다.

압축 해제 후 `smollm2-gsm8k-compact-v1/` 또는 `smollm2-svamp-compact-v1/` 내부:

```text
README.md, DATA_SCHEMA.json, load_data.py, questions.json, checksums.sha256
LICENSE-GSM8K.txt, LICENSE-SVAMP.txt
batch_0000/features.npz
batch_0000/generations.jsonl
...
```

한 batch는 최대 32응답이다. `B`=batch 응답 수, `C`=해당 batch의 유효 checkpoint 수 합,
`R`=응답별 `min(T,129)`의 합이다. NPZ에는 기존 Qwen compact와 같은 **7개 key만** 있다.

| Key | Dtype / shape | 의미 |
|---|---|---|
| `ids` | Unicode `[C]` | checkpoint 행의 문제 ID; 시점별 반복 |
| `checkpoints` | int16 `[C]` | 해당 행의 t |
| `hidden` | float32 `[C,4,2048]` | h6,h9,h11,h12 |
| `trajectory` | float32 `[C,24,5]` | checkpoint에서 24개 인접 layer 변화 |
| `readout_ids` | Unicode `[B]` | ragged prefix segment별 문제 ID |
| `readout_offsets` | int64 `[B+1]` | segment i=`readouts[offsets[i]:offsets[i+1]]` |
| `readouts` | float32 `[R,3,3]` | 매 위치 0..min(128,T-1), layer6/9/12, entropy/margin/top1 |

Label/생성 답변은 JSONL, 문제와 gold는 `questions.json`에 있다. **dataset+id로 join**한다.
`fingerprint`, `provenance.kind`로 보존 원본과 복구 재실행분을 구분한다. 같은 ID의 원래 label metadata를
새 응답에 덮어쓰지 않는다. Compact와 Full의 현재 응답/label이 서로 대응한다.
Qwen과 ID가 같아도 생성 응답과 label은 다르므로 Qwen label을 붙이지 않는다.

## Feature 계산

`h_l`=layer l의 raw hidden, `W`=고정 LM head, `N`=고정 final RMSNorm,
`p_l=softmax(W N(h_l))`. Norm은 정확히 한 번 적용한다. 모든 log는 자연로그다.

- Entropy: `-sum_v p_l(v) log p_l(v)`, 단위 nats.
- Margin: `p_l(top1)-p_l(top2)`. Logit 차이가 아니다.
- Top1 probability: `max_v p_l(v)`. **최종 답이 맞을 확률이 아니다.**
- Trajectory 축 순서: `||h_(l-1)||`, `||h_l||`, `||h_l-h_(l-1)||`,
  `delta_norm/max(previous_norm,1e-12)`, `1-clamp(cosine,-1,1)`.
  Embedding은 h0이며 trajectory index0은 h0→h1이다.
- Convenience scalar: layer6/9/12 × entropy/margin/top1 × `current,mean,std,change_from_t0` = **36개**.
  prefix `0..t`만 사용하고, `std(ddof=0)`, 변화량=`r[t]-r[0]`다. 저장된 FP32 값에서 float64로 계산한다.

## Full에 추가된 것

`C`는 **한 응답**의 checkpoint 수, `T`는 생성 토큰 수, `P=min(T,129)`다.
각 원본 batch ZIP의 `features/<id>.npz`에는 아래 16개 배열과 `record` JSON scalar, 총 17개 key가 있다.

| Key | Shape | 설명 |
|---|---|---|
| `checkpoints` | `[C]` | 유효 t |
| `hidden` | `[C,25,2048]` | embedding h0, blocks h1..h24 전체 |
| `pooled_hidden` | `[24,3,2048]` | layer별 end / whole_mean / number_mean |
| `trajectory` | `[T,24,5]` | checkpoint뿐 아니라 모든 토큰 예측 위치의 layer 변화 |
| `checkpoint_readouts` | `[C,24,7]` | 모든 block의 5가지 token metric + 두 JS |
| `checkpoint_top_ids` | `[C,24,10]` | logit-lens top10 token IDs |
| `checkpoint_top_logprob` | `[C,24,10]` | 위 top10의 log probability |
| `prefix_readouts` | `[P,6,5]` | layer6/9/11/12/18/24에서 매 위치의 5 token metrics |
| `token_readouts` | `[T,5]` | 최종 layer의 모든 위치 token metrics |
| `token_top_ids`, `token_top_logprob` | 각각 `[T,10]` | 최종 layer top10 |
| `token_region` | `[T]` | excluded0 / reasoning1 / final_number2 / format3 / unparsed4 |
| `token_offsets` | `[T,2]` | decoded text의 Unicode codepoint `[start,end)`; special은 -1 |
| `token_masks` | `[T,3]` | non_special / whole_nonwhitespace / final_number_nonwhitespace |
| `confidence` | `[26]` | whole 13개 + number 13개, 빈 mask는 NaN |
| `length` | `[4]` | log1p(prompt token 수, 전체 생성 수, whole mask 수, number mask 수) |

실수 배열은 float32, top ID/offset은 int32, region은 int8, mask는 bool이다.
Token metric 축은 `[generated_logprob,entropy,probability_margin,top1_probability,generated_rank]`.
Generated logprob는 **실제 생성 token** y_t의 `log p_l(y_t)`이고,
rank는 `1 + count_v[p_l(v)>p_l(y_t)]`다. T/top-p 변형 **전 raw 분포**를 사용한다.
Sampling token이므로 최종 layer에서도 항상 top1/rank1일 필요는 없다.

Checkpoint metric 마지막 둘은 `JS(p_l,p_24)`와 `JS(p_l,p_(l-1))`다.
`JS(p,q)=0.5 KL(p||m)+0.5 KL(q||m)`, `m=(p+q)/2`, nats. l=1의 previous는 embedding readout이다.
`js_to_final`은 최종 layer를 필요로 하므로 중간층 조기 예측의 입력으로 쓰면 안 된다.

Pooling은 **생성 token을 소비한 뒤의 위치**를 사용한다. `end`는 마지막 non-special token,
whole_mean은 non-special/non-whitespace 생성 token 전체 평균, number_mean은 파싱된 최종 숫자 token 평균이다.
따라서 checkpoint의 다음 token 예측 직전 hidden과 위치 의미가 다르다. 빈 pooling mask는 zero vector다.
Number 위치는 gold가 아니라 **모델 응답**의 최종 줄을 파싱해 정한다.

Confidence는 각 whole/number mask에서 다음 순서의 13개 통계다. 정확한 순서는 각 `record.confidence_names`에도 있다.

```text
logp_mean, logp_q10, logp_min, logp_last16_mean,
prob_lt_01_ratio, prob_lt_05_ratio,
entropy_mean, entropy_q90, entropy_max, entropy_last16_mean,
margin_mean, margin_q10, margin_min
```

`q10/q90`은 NumPy 기본 linear quantile, `last16`은 mask를 적용한 token sequence의 마지막 최대 16개,
probability 비율은 실제 생성 token의 확률이 0.1/0.5보다 작은 비율이다. 빈 mask의 NaN은 0으로 임의 대체하지 않았다.
임퓨테이션/표준화는 train split에서만 fit한다.

**Full도 저장하지 않은 것:** 전체 vocabulary logits/probabilities, 모든 token의 dense hidden vector,
attention, gradients, 모델 weights, 학습된 probe. Full은 *이번에 저장한 모든 feature*이지 모든 내부 tensor가 아니다.

## Label과 분석 주의

마지막 nonempty line이 `#### <number>`인지를 엄격히 파싱하고 gold 숫자와 Decimal로 비교했다.
`answer_correct=true`는 최종 숫자 일치이지 풀이 전체의 논리 검증 결과가 아니다.
형식 실패는 오답이다. `analysis_eligible`은 `parse_status==ok and not truncated`이며 분석용 사후 필터다.
오류 유형 인간 주석이나 LLM judge 분류는 이번 SmolLM2 데이터에 없다.

- 사후 verifier는 pooled/end/confidence/length를 사용할 수 있으나, 조기 probe에 그대로 섞으면 미래 정보 누수다.
- 중간층 실험은 t까지의 prefix와 layer12 이하만 사용한다. Full trajectory는 `:12`로 제한한다.
- Gold, label, parse/truncation 진단, QA, 최종 길이, ID를 predictor에 넣지 않는다.
- 동일 질문의 응답이 train/test에 흩어지지 않게 question group 기준으로 split한다. 정해진 split/학습 probe는 제공하지 않는다.
- 형식 실패 포함/제외 결과를 함께 비교한다. 형식 탐지 능력을 reasoning 오류 탐지 능력으로 해석하지 않는다.
- 난이도 신호와 추론 오류 신호가 혼재할 수 있다. 두 모델을 비교할 때 같은 질문 집합을 고정한다.

## NumPy로 사용

Python+NumPy만 있으면 저장 feature를 읽을 수 있다. 모델 다운로드나 GPU는 필요 없다.
`AI_STARTER`는 설명/작은 scalar 분석용이며 raw hidden 분석에는 실제 compact/full asset이 필요하다.

```python
from load_data import verify_checksums, feature_matrix, iter_full

root = "smollm2-gsm8k-compact-v1"
verify_checksums(root)
ids, X, y = feature_matrix(root, t=32, kind="middle")  # h12, y=1 오답
# kind="middle_delta": [h12, h12-h11]
# kind="multi": [h6,h9,h12]; kind="scalar": 36개 prefix 통계

# 모든 full part를 같은 폴더에 압축 해제한 뒤, 응답 하나씩 읽는다.
for record, arrays in iter_full("smollm2-full-v1", dataset="gsm8k"):
    print(record["id"], arrays["pooled_hidden"].shape)
    break
```

Compact 내부 `checksums.sha256`로 파일 변형을 확인하고, 다운로드 asset은 `SHA256SUMS.txt`와 대조한다.
Full의 `FULL_INDEX.json`과 part별 manifest에는 원본 batch ZIP hash가 있다.
Numpy pickle은 필요 없으며 `allow_pickle=False`를 유지한다.

## 재현성과 출처

실행 fingerprint: `dc40716a9fffcb546b478c759ba5e9a90546c9481ec830c53051f9059c342a2c`.
완료 metadata ZIP SHA-256: `c71a036950c93aea79a3135f9d8fdaf1028dfb2f6ea8af7e17bc1ce908e9be2e`.
복구 재실행 fingerprint: `75d50117968c998f82ddb6aad71d55f3c56f8dfbc6037b3cfa25e497bf963603`.
새 실행은 model revision, prompt, seed, decoding, FP32 추출 정의를 유지하고 누락 문제만 재실행한다.
동일한 RTX3090 / Torch2.3.1+cu121 / Transformers4.46.3을 사용하며, 독립 fingerprint로 기록한다.
공개 전 보존된 GSM8K 2개/SVAMP 2개의 실제 token 경로 exact match와 feature 허용 오차 검사를 요구한다.
이 작은 재현 검사가 모든 유실 응답의 bitwise 동일성을 보장하는 것은 아니다.

PyTorch 2.3.1+cu121 / Transformers 4.46.3에서 SVAMP top-p sampling의 CUDA cumsum이
strict deterministic 옵션과 충돌했다. `runtime_compat.py`는 **누적합만 CPU에서 계산**하며
GPU sort/softmax, 생성 설정, 모델/feature 코드, 이미 끝난 첫 32개 greedy 응답은 유지했다.
이 정책과 검증 hash는 metadata의 `runtime_patch.json`에 있다. CPU/CUDA 누적합이 경계에서
부동소수점 차이를 낼 수 있으므로 일반 CUDA top-p와 bitwise 동일하다고 주장하지 않는다.

원래 실행에서는 greedy checkpoint argmax 일치율 1.0, 독립 prefix 확인 51건,
최초 2응답의 generation/replay logprob 최대 차이 약 1.56e-5를 확인했다.
QA는 feature/label predictor가 아니다. 재실행 복구 시 원본과의 검증 결과를 별도로 공개해야 한다.

GSM8K는 [OpenAI GSM8K](https://github.com/openai/grade-school-math),
SVAMP는 [arkilpatel/SVAMP](https://github.com/arkilpatel/SVAMP)의 데이터를 사용했다.
각 MIT license는 compact asset에 포함하며 원본 데이터의 출처/고지를 유지한다.
SmolLM2 모델은 [공식 모델 저장소](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)를 참고한다.

## 패키징 도구

`audit_backup.py`는 보존 batch ZIP과 내부 모든 파일의 SHA-256을 검증해 유실 목록을 만든다.
`build_release.py SOURCE OUTPUT --preflight`는 전체 원본의 존재 여부를 확인한다.
실제 빌드는 보존+복구 총 530개 batch와 metadata hash를 검증하고, 8,473개 숫자 label을 재계산하며,
모든 feature shape와 ZIP readback까지 통과한 뒤에만 `READY_TO_PUBLISH.json`을 만든다.
부분 백업을 전체 데이터처럼 공개하는 우회 옵션은 없다.
`--recovery RECOVERY_BACKUP`은 완료된 복구 snapshot, 승인된 fingerprint, 대조군 재현,
입력 설정 일치와 보존/복구 ID의 겹침 없는 전체 coverage를 요구한다.
