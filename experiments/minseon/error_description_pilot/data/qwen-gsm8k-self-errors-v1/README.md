# Qwen GSM8K 자체 생성 오답 v1: 규칙 유형·위치 라벨

**상태: 오류 유형은 규칙으로 붙인 자동 라벨이다. 인간 검수 전.**

모델: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (팀의 다른 Release와 같음)

## 무엇인가

- Qwen2.5-1.5B-Instruct가 GSM8K 공식 **train** 7,473문제를 직접 풀었다.
  - 문제당 greedy 1개 + sampling 4개, 총 37,365풀이
- 그중 **오답 7,691개** (문제 3,489개)를 담았다.
- 팀의 다른 Release와 같은 기준으로 맞췄다.
  - GSM8K 공식 test는 넣지 않았다.
  - `####` 줄이 없는 형식 실패는 뺐다.
- 판정: `####` 뒤 숫자와 gold의 차이가 1e-6 미만이면 정답. 숫자 비교일 뿐, 추론 과정의 정오를 뜻하지 않는다.
- 같은 문제를 다른 Release와 공유한다.
  - `train-74` = `train-00074`. pid는 GSM8K 공식 train 파일의 줄 번호(0부터)다.
  - 응답은 다르다.

## 생성 설정

- vLLM 0.5.3.post1, bfloat16, max_tokens 1024
  - 팀의 다른 Release(HF transformers, FP32)와 생성 방식이 다르다.
- greedy: temperature 0
- sampling: n=4, temperature 0.7, top_p 0.95, 엔진 seed 0
- 프롬프트: 팀의 다른 Release와 같다.
  - Qwen chat template, `add_generation_prompt=True`

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

## 파일

| 파일 | 내용 |
| --- | --- |
| `errors.jsonl` | 오답 7,691개: 질문, GSM8K 풀이, 모델 응답, 규칙 라벨 |
| `token_ids.jsonl` | 실제 prompt / output token ids. 모델을 다시 돌려 다른 feature를 뽑을 때 사용 |
| `hidden.npy` | hidden state, float16, (7691, 4 위치, 7 층, 1536). 약 660MB |
| `confidence.npy` | 확신도·길이 feature, float32, (7691, 9) |
| `features_meta.json` | 두 배열의 행 id (errors.jsonl과 같은 순서), 층·위치·feature 이름 |
| `DATA_SCHEMA.json` | 필드 설명 |
| `ATTRIBUTION.md`, `LICENSE-GSM8K.txt` | 출처 고지 |
| `SHA256SUMS.txt` | 무결성 확인 |

## Feature (`hidden.npy`, `confidence.npy`)

모델 자신의 실제 생성 토큰 경로(`token_ids.jsonl`)를 HF transformers(float16)로 다시 넣어 뽑았다.

- 층: 4, 8, 12, 16, 20, 24, 28
  - HF `output_hidden_states`의 해당 인덱스다. 28층은 최종 RMSNorm **이후** 값이다.
  - 팀의 다른 Release의 h28은 RMSNorm **이전** 값이라 그대로 섞으면 안 된다.
- 위치
  - `q_end`: 프롬프트 마지막 토큰. 답을 쓰기 전이라 문제만 아는 상태 (난이도·주제 통제용)
  - `r_end`: `####` 줄 직전, 풀이의 마지막 토큰
  - `r_mean`: 풀이 토큰 전체의 평균 (`####` 줄 제외)
  - `ans`: 최종 답 숫자의 마지막 토큰
- 확신도 (마지막 층 분포, 풀이 토큰 기준)
  - `nll_mean`, `nll_max`, `lowtail_lp` (하위 10% logprob 평균), `ent_mean`, `ent_max`, `margin_mean`, `margin_min`, `ans_lp` (답 숫자 logprob), `length` (풀이 토큰 수)
- 우리 실험에서 맞음/틀림 탐지에 가장 좋았던 것은 `r_mean` 20층이다 (전체 풀이 기준 AUROC 0.88).

## 규칙 라벨 (`rule_*`)

맞는 값 목록 = 문제에 나온 숫자 + GSM8K 풀이의 중간값 + 같은 문제를 맞힌 풀이의 값.
풀이의 식(`a op b = c`)을 순서대로 보며, 값이 목록에 없는 첫 식을 첫 오류로 본다.

| `rule_type` | 뜻 | 개수 |
| --- | --- | --- |
| `SETUP` | 첫 오류 식의 계산은 맞지만, 다른 양을 계산함 | 4,422 |
| `NOEQ` | 검사할 식이 없음 (답만 쓰거나 LaTeX·문장 풀이) | 1,797 |
| `TARGET` | 식은 다 맞는데, 최종 답이 중간값 | 586 |
| `OTHER` | 식은 다 맞는데, 최종 답이 설명되지 않음 | 549 |
| `ARITH` | 첫 오류 식의 계산 자체가 틀림 | 311 |
| `NEWNUM` | 첫 오류 식에 어디에도 없는 숫자가 나옴 | 26 |

- `rule_first_wrong_eq`: 첫 오류 식의 번호 (0부터). `ARITH`, `SETUP`, `NEWNUM`에만 있다.
- **알려진 한계:** 공개하지 않은 test 오답 1,406개를 AI 검수자 10명이 다시 라벨링해 비교했다 (일치도 Fleiss κ 0.91~0.96).
  - 검수 클래스와 같은 비율: SETUP 93%, ARITH 67%, TARGET 39%
  - TARGET은 앞에서 이미 틀렸는데 답이 우연히 중간값과 같은 경우가 많다.
  - 첫 오류 위치는 검수와 같은 줄이 50%. 검수가 더 앞인 경우가 34%로, 대부분 식이 아니라 문장 속 숫자에서 먼저 틀렸다.

## 로딩 예시

```python
import json

import numpy as np

rows = [json.loads(l) for l in open("errors.jsonl", encoding="utf-8")]
meta = json.load(open("features_meta.json"))
assert meta["ids"] == [r["id"] for r in rows]                  # 같은 순서

H = np.load("hidden.npy", mmap_mode="r")                       # (7691, 4, 7, 1536)
C = np.load("confidence.npy")                                  # (7691, 9)
X = H[:, meta["positions"].index("r_mean"), meta["layers"].index(20)].astype(np.float32)
y = np.array([r["rule_type"] == "ARITH" for r in rows])        # 예: 계산 실수 vs 나머지
groups = [r["pid"] for r in rows]                              # 같은 문제는 같은 fold로
```

`id`는 `{split}-{pid}-{sample}`이다. `sample` 0은 greedy, 1~4는 sampling이다.
같은 문제의 풀이는 같은 fold에 두어야 한다.
