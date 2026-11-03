# 오류 유형 라벨링 기준 (성훈 codebook v0.1의 수학 오류 축 그대로, 2026-11-05)

각 항목은 Qwen2.5-1.5B-Instruct가 GSM8K 또는 SVAMP 문제에 낸 **틀린** 답변이다 (최종 숫자가 gold와 다름).
REFERENCE는 판정 보조자료이지 무조건 옳다고 가정하지 않는다.

## 출력 형식

한 항목당 한 줄: `id<TAB>type<TAB>conf<TAB>reason`

- **type**: 아래 7개 값 중 하나 (영문 코드 그대로).
- **conf**: `h` 분명함 / `m` 다른 유형도 그럴듯함. `m`이면 reason에 `alt=<다른 type>`을 적는다.
- **reason**: 영어 15단어 이내. 무엇이 틀렸는지.

## 수학적 주 오류 `math_error_type` (codebook 원문)

| 값 | 관찰 가능한 기준 | 경계 |
|---|---|---|
| modeling_relation | 수량의 역할·비교 방향·조건·목표·관계식을 잘못 구성 | 문장 해석과 식 세우기를 억지로 구분하지 않고 subtype으로 기록 |
| arithmetic_execution | 적힌 피연산자/연산의 수치 계산 또는 식 변형이 틀림 | 앞의 잘못된 관계식을 일관되게 계산한 것은 이 유형으로 중복 집계하지 않음 |
| unit_conversion | 명시된 단위를 바꾸지 않거나 서로 다른 단위를 같은 단위로 계산 | 잘못된 비율 공식을 선택한 것은 modeling_relation; 단위 자체의 누락/오표기는 이 유형 |
| reasoning_state | 관계·계산이 맞는데 경우 누락/잘못된 논리 전개/상태 갱신이 확인됨 | 근거가 명확할 때만 사용; 모델의 주의력/기억 실패를 추측하지 않음 |
| answer_reporting | 목표의 올바른 값을 풀이에서 명시적으로 구한 뒤 최종 답을 다르게 옮김 | 올바른 중간 수량을 답으로 냈지만 목표 수량을 계산하지 않았다면 modeling_relation/target_mismatch |
| undetermined | 틀린 값은 확인되지만 오류 종류를 특정할 풀이가 부족함 | 답만 제시하거나 유도 없는 중간 값이 틀린 경우, 산술/수량 혼동 등을 임의로 택하지 않음 |
| none_observed | 표시된 계산과 수학적 결론에서 오류가 관찰되지 않음 | strict-label이 오답이어도 형식만 어긴 경우 가능; 수학적 완전성을 증명한 뜻 아님 |

## 원칙 (codebook 원문)

- 주 오류는 텍스트상 가장 이른 근거로 정하고, 독립적인 추가 오류는 무시한다. 단순 전파 오류는 새 원인으로 세지 않는다.
- 최종 gold와 다르다는 사실만으로 오류 원인을 지어내지 않는다. 질문과 gold가 충돌하거나 gold가 틀렸으면, 모델 풀이에
  수학 오류가 따로 있는지만 본다 (없으면 none_observed).

## codebook의 경계 사례 (원문 요약)

- 답만 있어 계산/해석 오류를 구분할 수 없음 → undetermined.
- 올바른 거리/속도 관계를 적었지만 계산을 생략하고 잘못된 숫자를 출력 → undetermined; 단순 산술/답 복사를 임의로 택하지 않음.
- 잘못된 중간/전체 수량은 찾을 수 있으나 해당 값의 유도가 없으므로 원인 유형은 undetermined.
- 수학적 결론은 reference와 같지만 final-line 규약 위반 → none_observed.
- 질문은 chapters/day를 묻지만 gold는 days/chapter에 해당하고, 모델의 식도 틀림 → 모델의 수학 오류 유형을 따로 붙임.
- 모델이 탑승 인원 72를 구하고, 물은 순증가 9 대신 답함 → modeling_relation (target_mismatch).
