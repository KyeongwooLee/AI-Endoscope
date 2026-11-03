# 조민수 실험 (Qwen2.5-1.5B-Instruct)

질문: 모델 내부 hidden state가 confidence(토큰 확률)보다 **오답을 더 잘 찾는가**, 그리고 **오류 유형**까지 구분하는가.
모델 가중치는 고정했고, 학습한 것은 logistic regression probe뿐이다.

| 폴더 | 내용 |
|---|---|
| [pilot/](pilot/) | EIC 데이터(GPT-4가 인위로 넣은 오류)로 단계 단위 오류 탐지·유형 분류 |
| [pilot/gsm8k/](pilot/gsm8k/) | 모델이 직접 푼 GSM8K 풀이 약 4.3만 개로 정답/오답 판별 |
| [taxonomy_relabel/](taxonomy_relabel/) | 성훈 오류 주석 885개를 EIC 유형 기준으로 다시 분류 + AI 검수 4건 (`setup_ab/`, `loc_audit/`, `cls_audit/`, `sh_audit/`) |
| [pilot/gsm8k/types/](pilot/gsm8k/types/) | 규칙 오류 유형, 유형·위치 probe, 교정 실험 결과 |

데이터: [qwen-gsm8k-self-errors-v1](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/qwen-gsm8k-self-errors-v1)
(GSM8K train 오답 7,691개, 규칙 라벨, token ids, hidden feature).
팀 Release와 같이 **GSM8K 공식 test의 문제별 데이터·라벨은 공개하지 않는다.** 이 폴더의 문제별 라벨 파일도 train/SVAMP 행만 남겼고, test 결과는 요약 수치만 있다.

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
- 전체 feature(약 7GB)는 서버 `~/yaicon_pilot/gsm8k/feat`에 있고, train 오답분은 위 Release에 있다.

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

## 4. 규칙 오류 유형과 유형 probe (`pilot/gsm8k/auto_types.py`, `probe_types.py`)

맞는 값 목록(문제 숫자 + GSM8K 풀이 중간값 + 같은 문제의 정답 풀이 값)에 없는 첫 식을 첫 오류로 보고 유형을 붙였다.
오답 9,734개: SETUP 5,659 / NOEQ 2,195 / TARGET 728 / OTHER 707 / ARITH 414 / NEWNUM 31.

| 구분 (AUROC) | hidden r_mean L20 | 텍스트 TF-IDF | 확신도+길이 | 질문만 본 hidden |
|---|---|---|---|---|
| ARITH vs SETUP | **0.79** | 0.69 | 0.64 | 0.69 |
| TARGET vs SETUP | **0.71** | 0.61 | 0.65 | 0.54 |
| 설정 오류 안의 A 읽기 vs B 식 세우기 (`setup_ab/`) | 0.64 | 0.61 | 0.51 | 0.61 |

계산 실수와 중간값 답함은 구분되지만, 설정 오류를 더 쪼갠 유형은 문제 주제 이상으로 구분되지 않는다.

## 5. 첫 오류 위치 probe (`extract_loc.py`, `probe_loc2.py`, `check_loc_noise.py`)

식마다 hidden(L20)을 보고 풀이 안에서 첫 오류 식 하나를 고른다. 규칙 라벨로 train 문제에서 학습하고 test 문제로 평가했다.

| 방법 | 맞힌 비율 (AI 검수 라벨, 설정 오류 1,105개) | 참고: 규칙 라벨로 채점 |
|---|---|---|
| **hidden** | **51%** | 54% |
| 확신도 | 46% | 41% |
| 항상 첫 식 | 41% | 40% |

hidden이 확신도보다 +5%p(CI [+1, +9]). 규칙 라벨로 채점하면 차이가 과장된다(probe가 규칙의 버릇까지 배움).

## 6. 교정 실험 (`correct_exp.py`, `analyze_correct.py`, `analyze_repeat.py`)

test 오답 1,264개, 풀이마다 4번 시도. 고친 비율:

| 방법 | 고친 비율 |
|---|---|
| 처음부터 다시 풀기 | **32%** |
| 규칙이 찍은 위치부터 다시 풀기 | 24% |
| 예측한 위치부터 다시 풀기 | 22% |
| 스스로 검토 (refine) | 3% |

- 같은 위치에서 다시 풀면 48%가 같은 틀린 값을 또 쓴다. 다른 값이 나올 때까지 다시 뽑아도 32%로 처음부터 다시 풀기와 같다.
- 계산 실수(ARITH 99개)는 계산기로 고치면 처음부터 다시 풀기보다 +14%p (CI [+2, +26]).
- 중간값 답함에 "질문을 다시 봐라" 힌트: 10% vs 다시 풀기 44%. 모델이 힌트를 무시한다.

## 7. AI 검수 (`taxonomy_relabel/`)

검수자는 모두 LLM 에이전트이고, 규칙 라벨을 보지 않고 각 폴더의 RUBRIC 기준으로 라벨링했다. 일치도는 공통 문항의 Fleiss κ.

| 폴더 | 대상 | 검수자 | 일치도 | 주요 결과 |
|---|---|---|---|---|
| `setup_ab/` | 설정 오류 990개를 A 읽기 / B 식 세우기로 | 6 | 묶음 0.86, 세부 코드 0.77 | 61%가 스스로 "애매" |
| `loc_audit/` | 첫 오류 식 630개 | 5 | 0.95 | 규칙 위치 = 검수 68% |
| `cls_audit/` | test 오답 1,406개의 클래스 + 처음 틀린 줄 | 10 | 클래스 0.96, 줄 0.91 | 규칙 정확도 SETUP 93% / ARITH 67% / TARGET 39% |
| `sh_audit/` | 성훈 7유형 라벨 100개 재라벨 | 5 | 0.94 | 원 라벨과 93% 일치, 관계 설정 66%로 쏠림 |

클래스를 많이 나누면 애매하고(세부 유형 κ 0.77, 애매 61%), 적게 나누면 한 클래스로 쏠린다(3클래스 κ 0.96, 설정 오류 86%).

## 8. 성훈 7유형을 hidden으로 맞히기 (`probe_sh_types.py`, `probe_sh_multi.py`)

성훈 Release의 실제 token ids로 우리 feature를 다시 뽑아(`make_sh_gen.py` → `extract_gen.py`) 1,100개에 probe를 학습했다.
답 옮기기는 0개라 6개 중 하나를 맞힌다(문제 단위 5-fold × 5회).

| 입력 | 클래스별 정답률 평균 | macro-F1 |
|---|---|---|
| 전부 관계 설정으로 찍기 | 17% | 0.13 |
| 확신도 | 39% | 0.27 |
| 질문만 본 hidden | 29% | 0.28 |
| **hidden r_mean L20** | **41%** | **0.40** |

이진 비교(각 유형 vs 관계 설정)에서 질문만 본 hidden보다 확실히 나은 건 계산 실행뿐이다(0.86 vs 0.73).
