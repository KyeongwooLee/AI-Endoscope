# 검수 지침 (Reviewer guide) — grounded error-description pilot v1

이 검수는 **사람 검수자**가 독립적으로 수행해야 합니다. AI가 작성한 판정은 `reviewer_type: "ai_provisional"`로만 제출할 수 있으며, 주 지표에 절대 포함되지 않습니다.
`review_key_private/` 폴더(방법·순위·ID 대응표)는 검수가 끝날 때까지 열지 마세요.

## 0. 준비

1. `review_package/templates/reviewer_1/`(또는 배정된 `reviewer_2/`) 폴더를 통째로 복사해 `completed_reviews/<본인ID>/`로 만듭니다.
2. `reviewer.json`에 `reviewer_id`, 이니셜, 날짜를 적습니다. `reviewer_type`은 `"human"`, `saw_method_key_or_scores`는 `false`여야 합니다.
3. CSV는 판정 열만 채우고 텍스트 열은 수정하지 마세요(코드로만 매칭하지만, 원문 확인을 위해 보존). 원문은 `*.html` 파일이 더 읽기 쉽습니다. `'`로 시작하는 칸은 스프레드시트 수식 방지를 위한 표시이며 원문에는 없습니다.
4. 두 번째 검수자는 `review_assignment.json`의 사전 지정 부분집합만 합니다. 서로 결과를 보지 말고, 불일치는 별도 판정자가 `adjudication_template.csv`로 정리합니다.

## 1. 항목 검수 Stage 1 — 설명 없이 먼저 진단 (`items_stage1.csv`)

문제, 모델 풀이, (틀릴 수 있는) 참조 풀이만 보고 기록합니다. **Stage 2를 보기 전에 끝내세요.**

| 열 | 값 | 의미 |
| --- | --- | --- |
| `observable_error_present` | yes / no / uncertain | 풀이 텍스트에서 실제로 확인되는 수학적 실수가 있는가 |
| `observed_error_or_uncertainty` | 자유 기술 | 가장 이른 실질적 실수 또는 판단 불가 이유 (한국어/영어 무관) |
| `reference_problem` | yes / no / unsure | 참조 풀이나 문제 문구 자체에 오류·모호성이 있는가 |

- 다른 정당한 풀이 경로는 오류가 아닙니다. 벤치마크 숫자와 달라도 문구상 타당한 해석이면 `no` 또는 `uncertain` + `reference_problem=yes`.
- 하류 전파(앞 실수 때문에 뒤 값이 틀린 것)는 새 실수로 세지 않습니다.
- 계산 과정 없이 틀린 숫자만 있으면 `uncertain`이 정상입니다.

## 2. 항목 검수 Stage 2 — 설명 A/B 평가 (`items_stage2.csv`)

각 항목에 두 설명이 무작위 중립 라벨 A/B로 붙어 있습니다. 어느 방법인지 추측하지 마세요.
"(no description: ...)"는 주석자가 기권한 경우이며 해당 열은 `not_applicable`로 둡니다.

| 열 | 값 |
| --- | --- |
| `correctness` | correct / partially_correct / incorrect / not_applicable — Stage 1 진단과 원문에 비추어 실수를 맞게 짚었는가 |
| `specificity` | specific / generic / not_applicable — 관련 양·관계·조건을 구체적으로 지목하는가, 아니면 "계산 실수가 있다" 수준인가 |
| `unsupported_assertion` | yes / no / not_applicable — 원문에 없는 내용(관찰 불가한 원인 추정, 없는 식)을 주장하는가 |
| `evidence_fidelity` | faithful / unfaithful / not_applicable — 설명이 인용·서술한 풀이 내용이 원문과 일치하는가 |

## 3. 이웃 쌍 검수 (`pairs.csv`) — 주 지표

두 응답(각각 문제·풀이·참조)만 보고 판단합니다. 어떤 방법이 이 쌍을 찾았는지, 생성된 설명, 유사도 점수는 보이지 않습니다. 좌우 순서도 무작위입니다.

**질문: 두 응답이 같은 구체적·관찰 가능한 수학적 오류 패턴을 보이는가?**

- `same`: 두 풀이 모두 같은 수학적 관계/연산을 비슷한 방식으로 잘못 다루고, 이것이 원문에서 확인된다.
- `different`: 실제로 작동한 실수가 다르다. 주제·연산자·문체·넓은 범주("설정 오류")가 같아도 `different`.
- `uncertain`: 한쪽 또는 양쪽 진단이 불확실하거나, 참조가 의심스럽거나, 다중 오류 해석이 양립하지 않는다.

"둘 다 틀림", "둘 다 곱셈 사용", "둘 다 모델링 오류"는 `same`의 근거가 아닙니다.

### 개발 단계(30건)에서 정한 세분성 기준

개발 사례를 검토하며 반복된 패턴을 기준으로 다음 수준에서 `same`을 판정합니다(이 목록은 강제 분류표가 아니며, 목록에 없는 패턴도 같은 원리로 판단합니다).

1. **관계 방향 반전**: "A가 B보다 k 적다/길다"를 반대로 적용 (예: 더 싼 물건에 k를 더함). 서로 다른 이야기여도 방향 반전이면 `same` 후보.
2. **잘못된 기준량/대상에 비율 적용**: 분수·퍼센트를 문제가 지정한 양이 아닌 다른 양(전체 인원, 세 사람 합계 등)에 적용.
3. **수량 누락 또는 이중 계산**: 문제에 명시된 항목(입양비, 새 결제액, 본인 자신, 나머지 요일)을 빼먹거나 같은 인수를 두 번 곱함. 누락된 항목의 역할이 비슷할 때 `same`.
4. **올바른 식의 실행 오류**: 식은 맞게 썼으나 덧셈/곱셈/나눗셈 결과가 틀림. 연산자가 달라도 "적힌 식을 잘못 계산"이라는 점이 같으면 `same` 가능. 단, 한쪽이 관계 자체를 잘못 세운 경우는 `different`.
5. **식만 세우고 계산하지 않은 채 근거 없는 최종 숫자**: 풀이에 숫자가 들어간 올바른 식이 있는데 최종 값이 그 식과 모순. 식 자체가 없고 숫자만 있으면 진단 불가 → 보통 `uncertain`.
6. **질문 대상 혼동**: 묻는 양(차이, 남은 분, 시간당 금액 등) 대신 다른 중간값·주어진 수를 답함.

합성 예시(데이터 관측 아님): 4개씩 든 봉지 3개 + 낱개 2개에서 `(3+2)*4`는 낱개를 봉지처럼 취급한 오류입니다. 다른 이야기에서 같은 "묶음 vs 낱개" 혼동이면 `same` 후보입니다. `3*4+2`를 올바르게 쓰고 16으로 계산한 응답은 연산자가 같아도 `different`입니다.

보조 열 `same_topic`, `same_operation`, `same_style`(yes/no/unsure)은 표면적 유사성을 기록하는 용도이며, 판정에는 쓰지 않습니다. 판단 가능할 때만 채우세요.

## 4. 제출

`completed_reviews/` 아래에 검수자별 폴더(reviewer.json + 세 CSV)를 둔 뒤 실행합니다.

```bash
python -I experiments/minseon/error_description_pilot/run.py --stage evaluate-reviewed \
  --reviews /absolute/path/to/completed_reviews [--adjudication /absolute/path/to/adjudication.csv]
```

비어 있는 판정은 "미검수(missing)"로 처리되며 0점이나 uncertain으로 바뀌지 않습니다. 필요한 쌍이 하나라도 미검수이면 주 지표는 계산되지 않습니다.
