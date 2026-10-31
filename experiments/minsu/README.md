# 조민수 실험 (Qwen2.5-1.5B-Instruct)

질문: 모델 내부 hidden state가 confidence(토큰 확률)보다 **오답을 더 잘 찾는가**, 그리고 **오류 유형**까지 구분하는가.
모델 가중치는 고정했고, 학습한 것은 logistic regression probe뿐이다.

| 폴더 | 내용 |
|---|---|
| [pilot/](pilot/) | EIC 데이터(GPT-4가 인위로 넣은 오류)로 단계 단위 오류 탐지·유형 분류 |
| [pilot/gsm8k/](pilot/gsm8k/) | 모델이 직접 푼 GSM8K 풀이 약 4.3만 개로 정답/오답 판별 |
| [taxonomy_relabel/](taxonomy_relabel/) | 성훈 오류 주석 885개를 EIC 유형 기준으로 다시 분류 |

## 1. EIC 파일럿 (`pilot/`)

EIC: Li et al., *Evaluating Mathematical Reasoning of LLMs: A Focus on Error Identification and Correction*.
데이터는 <https://github.com/LittleCirc1e/EIC> (commit `be8132a`)의 `data/`를 `pilot/data/EIC`에 받아 사용했다(이 브랜치에는 포함하지 않음).
GSM8K·MathQA 문제 1,558개, 9개 오류 유형.

**첫 오류 단계 찾기** (`run_steps.sh` → `results_steps.txt`, 문제 단위 5-fold, 같은 F1 기준)

| 방법 | 위치 모름 F1 | 위치 알려줌 F1 |
|---|---|---|
| 단계 NLL max (가장 낮은 토큰 확률) | 0.62 | 0.80 |
| hidden mean L28 | 0.66 | 0.79 |
| hidden mean L28 + NLL | **0.75** | **0.86** |

hidden + NLL이 NLL보다 +0.13 (bootstrap 95% CI [+0.11, +0.16]).

**오류 유형 9분류** (`run.sh` → `results.txt`): L16 macro-F1 0.56(chance 0.11).
하지만 같은 위치의 **정답 풀이**로 학습한 대조군도 0.43이 나와서, 신호 대부분이 유형이 아니라 문제·위치 맥락이다.
인위 주입 오류라 모델이 실제로 낼 법한 오류도 아니다(`results.txt` [C]: 주입 토큰이 원래 토큰보다 확률이 높은 경우 2~26%).

## 2. 직접 생성한 GSM8K (`pilot/gsm8k/`)

GSM8K train+test 8,792문제 × (greedy 1 + sampling 4) = 42,951 풀이(형식 실패 1,009 제외). 프롬프트는 성훈 실험과 동일.
생성은 vLLM(`generate.py`), feature는 모델 자신의 생성 토큰을 다시 넣어 추출(`extract_gen.py`, layer 4~28, 위치 q_end / r_end / r_mean / ans).

| 입력 | AUROC 전체 | 같은 문제 안(within) |
|---|---|---|
| confidence | 0.771 | 0.675 |
| confidence + length | 0.804 | 0.661 |
| 질문만 본 hidden (q_end L20) | 0.757 | 0.515 |
| **hidden r_mean L20** | **0.876** | **0.784** |
| hidden r_mean L20 + conf + length | 0.882 | 0.791 |

- within = 같은 문제의 정답 풀이와 오답 풀이만 비교. 문제 난이도 효과를 뺀 값이다.
- 질문만 본 hidden은 within이 0.5 근처라 난이도만 안다. 풀이를 본 hidden은 within에서도 confidence보다 +0.11 (CI [+0.10, +0.12]).
- train/test 나눠서 따로 봐도 같은 결론(`results_split.txt`).
- feature(약 7GB)는 서버 `~/yaicon_pilot/gsm8k/feat`에만 있다.

## 3. 오류 유형 재분류 (`taxonomy_relabel/`)

성훈 주석([error-taxonomy-full-v1-draft](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft))에서
첫 오류 위치가 표시된 885개를 EIC 9유형 + TG(묻는 대상을 잘못 고름)로 다시 붙이고, 풀이 단계별 4묶음으로 집계했다. 기준은 [RUBRIC.md](taxonomy_relabel/RUBRIC.md).

| 묶음 | 코드 | 개수 |
|---|---|---|
| A 정보 읽기 | CV 147, CO 86, HA 31 | 264 (30%) |
| B 식·절차 세우기 | OP 73, FC 117, MS 111, TG 133 | 434 (49%) |
| C 실행 | CA 76, PS 65 | 141 (16%) |
| D 단위 | UC 45 | 45 (5%) |

- 성훈의 산술·단위 유형은 거의 그대로 CA·UC로 옮겨졌고, 실제로 바뀐 건 `modeling_relation` 732개가 A 258 / B 421 등으로 쪼개진 것이다.
- **AI가 AI 주석을 다시 분류한 초안이다.** 4개 에이전트가 공통 30문항에 붙인 라벨의 쌍별 일치도는 코드 0.87, 묶음 0.94였다. 다만 61%가 경계 사례(conf m)라 사람 검수가 필요하다.
- 재현: release의 `labels.jsonl`(sha256 `6137ffc9…193d`)을 받아
  `python -I scripts/make_chunks.py labels.jsonl chunks` → 라벨링 결과를 `out/chunk_k.tsv`에 저장 → `python -I scripts/merge.py labels.jsonl chunks out`.
  결과는 `out/relabeled.tsv`, `out/summary.txt`.
