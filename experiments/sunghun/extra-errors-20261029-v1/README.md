# Qwen 추가 오답 4,000개 — 2026-10-29

**수집·로컬 무결성 검증 완료, 오류 유형 미분류.** 기존 데이터와 주석을 보존한 추가 실행이다.

[전체 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/qwen-extra-errors-20261029-v1)

| 항목 | 개수 |
|---|---:|
| 전체 생성 응답 | 18,656 |
| 새로 선택한 형식 정상·비잘림 숫자 오답 | 4,000 |
| GSM8K / SVAMP 선택 오답 | 3,188 / 812 |
| 선택 오답의 고유 질문 | 2,717 |
| 일치하는 feature | 4,000 |
| 전체 생성 중 정답 / 형식 실패·잘림 / 중복 | 13,880 / 695 / 81 |

4,000개는 **응답 수**다. 유형별 수백 개를 확보했다는 뜻이 아니며, `math_error_type=null`,
`human_review_status=not_reviewed`다. 기존 1,100개 AI 주석과 합쳐 유형별 집계를 하지 않았다.
숫자 오답은 reference 문제, 단위 등가, 모호한 전제를 포함할 수 있어 확정된 추론 오류 기제와 같지 않다.

## 다운로드

| 파일 | 용도 |
|---|---|
| `new_errors.jsonl` | 선택 오답 4,000개의 질문·응답·reference·숫자 판정·ID/hash/fold. 텍스트 검토용 |
| `AI_STARTER.zip` | 위 JSONL과 설명·schema·집계·검증·로더. raw hidden은 포함하지 않음 |
| `qwen-extra-errors-full-part-*.zip` **전부** | 전체 18,656응답의 실제 생성 token IDs 및 선택 오답 4,000개의 모든 저장 feature, 원본 batch ZIP/receipt. 약 4.8GB |
| `qwen-extra-errors-metadata.zip` | 동결 입력, 제외 대상 hash, 환경, run manifest, 정확한 producer/import 코드 |
| `FULL_INDEX.json`, `SHA256SUMS.txt` | 배포 ZIP 및 보존한 원본 checkpoint 파일별 SHA256 |
| `FINAL_AUDIT.json`, `dataset_summary.json` | 최종 검증과 집계 |

분할 파일은 **각각 독립 ZIP**이다. 모두 같은 위치에 압축 해제한다. 바이너리 연결하지 않는다.
전송 연결이 끊겨도 다시 받는 양을 줄이도록 각 ZIP을 약 200MB 이하로 나눴다.
AI_STARTER와 metadata ZIP도 같은 위치에 풀면 `qwen-extra-errors-20261029-v1/` 아래에 합쳐진다.
원본 batch ZIP은 `backup/outbox/` 안에 그대로 보존했으므로 nested ZIP을 지원하는 아래 로더를 사용한다.

```bash
gh release download qwen-extra-errors-20261029-v1 --repo KyeongwooLee/AI-Endoscope --dir downloaded
```

`SHA256SUMS.txt`를 확인한 뒤 ZIP들을 풀고, Python 3.10+ / NumPy 2.2.6에서:

```bash
python qwen-extra-errors-20261029-v1/load_data.py qwen-extra-errors-20261029-v1 --verify
```

```python
from load_data import iter_features, iter_generations
for record, arrays in iter_features("qwen-extra-errors-20261029-v1"):
    print(record["id"], arrays["hidden"].shape, arrays["posthoc_mean"].shape)
    break
# All outcomes, including correct answers and format failures:
for record in iter_generations("qwen-extra-errors-20261029-v1"):
    print(record["status"])
    break
```

ID와 `response_sha256`로 연결한다. JSONL은 LF(`\n`)로만 나눈다. 질문 문자열 안의 U+2028 등
Unicode separator는 유효한 문자열이므로 `str.splitlines()`로 나누면 안 된다.

## 고정 조건과 선택

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- RTX 3090 한 장, FP32, SDPA, TF32 off, torch 2.9.1+cu126, transformers 4.57.6.
- 기존 system/user prompt와 숫자 parser, T=0.7, top_p=0.95, top_k=0, 최대 새 토큰 1,024.
- 처리 batch 32, seed namespace 2026102901. batch별 정렬된 ID로 seed를 도출한다.
  기존 serial 실행과 token 단위 동일 재현을 주장하지 않는다.
- GSM8K 공식 train 중 이전 sampling 500문제를 제외한 6,973문제와 기존 SVAMP 1,000문제.
  GSM8K 공식 test는 로드하지 않았다. GSM8K revision `740312add88f781978c0658806c59bc2815b9866`.
- 고정 GSM8K:SVAMP=4:1 **생성 응답 스케줄**. 순환마다 섞고 반복 sampling한다.
  선택된 오답이 정확히 4:1이라는 뜻은 아니다.
- 이전 sampling/SVAMP 및 full-train greedy의 알려진 질문·응답 pair 11,966개와 새 실행 내부의 동일 pair를 제외한다.
- 처음 4,000개의 서로 다른 형식 정상·비잘림 숫자 오답을 선택한다. 채점은 strict Decimal 일치다.
- 새 GSM 질문은 정규화 질문 hash 단위 60/20/20 exploratory fold, SVAMP는 기존 fold를 보존한다.
  같은 질문의 여러 응답을 독립 문제처럼 분할하면 안 된다.
- 기존 SVAMP와 GSM train은 과거 연구에서 이미 관찰했다. 새 미사용 최종검증셋으로 부르지 않는다.

## Feature 해석

Gold는 채점에만 쓴다. 모델의 **실제 생성 token IDs**를 causal replay한다.
`hidden`의 t=0/32/64/128은 이미 생성한 답변 토큰 수이며, 첫 축은 해당 응답에서 가능한 checkpoint만 포함한다.
두 번째 축은 embedding 0 + block 1..28, 마지막 차원은 1,536이다. final RMSNorm 전 raw state다.
완료 pooling은 block별 end(non-special)/whole(nonwhitespace)/final-number mean이다.
별도 `posthoc_question/end/mean/number`는 각각 h28의 t0 및 완료 pooling이다.
`DATA_SCHEMA.json`과 metadata의 `source/smollm2-detailed-v1/extract.py`가 세부 계산을 정의한다.
extractor 이름에 SmolLM2가 있어도 이 실행의 모델은 **Qwen 1.5B**이고 실제 28층·1536차원을 쓴다.

완료 feature는 생성 종료 후 정보다. 이를 조기 오류 감지 feature라고 주장하면 안 된다.
전체 vocabulary logits, attention, 모든 토큰의 dense hidden은 저장하지 않았다.
정답/오답 classifier를 만들려면 선택 오답만이 아니라 raw의 정답도 필요하지만,
이 실행은 선택 오답에만 상세 feature가 있으므로 그 비대칭을 고려해야 한다.
정답 수·원문·reference·오류 주석·ID·fold를 학습 입력에 섞지 않는다.

## 원본 보존과 재현 범위

원본 583 generation ZIP + 583 feature ZIP 및 receipt를 byte 그대로 보존했다.
SHA256/member hash, ID/hash 연결, 숫자 label, 중복, 질문별 fold, feature shape/link 검사를 통과했다.
독립 prefix replay 대조는 초기 4응답 × 4 checkpoint = 16회이며 전수 독립 replay 검증을 의미하지 않는다.
수집 시간은 약 3시간 55분이다. 중간 SSH 백업 장애 후 완성본을 회수해 로컬 전수 검증했다.

원본 import 코드 기준 commit: `a178c22b807ee4727b59fb29365d9ff8c3ec9c2a`.
`run_manifest.json`은 수집 시작 때 동결한 파일 hash이며, producer 3파일은 이 hash와 일치한다.
`audit_local.py`는 수집 후 JSONL Unicode parsing과 nested ZIP 읽기를 수정한 별도 검증기다.
**보존된 `protocol.py.read_records`에는 역사적인 `splitlines()` 제한이 있다.** 완성 데이터 읽기에는
`load_data.py`를 쓴다. 보존 producer로 새 수집을 재개하기 전에 이 제한을 명시적으로 고쳐야 한다.

이 폴더의 producer/helper는 당시 staging 경로를 가정한 보존 코드이며 일반적인 일괄 실행기가 아니다.
GPU 재실행 시 동결 `backup/input/`, 고정 모델 파일, 위 commit의 `experiments/sunghun` imports가 필요하다.
원래 staging은 `REPO/extra_errors_20261029/worker.py`와 `REPO/experiments/sunghun/...`였다.
현재 완성 경로에 새 producer를 실행하지 않는다. 모델 가중치·인증정보·이전 baseline 대용량 파일은 배포하지 않는다.

출처와 원본 MIT 고지는 [ATTRIBUTION.md](ATTRIBUTION.md), `LICENSE-GSM8K.txt`, `LICENSE-SVAMP.txt`에 있다.
