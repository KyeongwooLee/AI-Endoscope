# 오류 유형 수동 LLM 주석: Pilot v0.1

## 목적과 지위

현재 대화의 assistant가 실제 저장 문제·모델 출력·reference를 직접 읽고 작성한 **검수 전 초안 주석**이다. 별도 judge 모델을 VESSL에서 돌리거나 외부 judge API를 호출하지 않는다. 정답 reference는 판정 보조자료이지 무조건 옳다고 가정하지 않는다. 이 주석을 인간 확정 라벨 또는 객관적인 내부 실패 원인이라고 부르지 않는다.

Pilot은 각 데이터셋 fit fold의 parsed-wrong 8개와 format-invalid 2개, 총 20개다. 같은 질문은 한 번만 포함한다. 고정 SHA256 순서로 선택하며 feature·verifier 점수를 주석자에게 보여주지 않는다. 형식 실패를 의도적으로 포함한 샘플이므로 유형 비율을 전체 데이터의 비율로 추정하면 안 된다.

## 다축 라벨

### 수학적 주 오류 `math_error_type`

| 값 | 관찰 가능한 기준 | 경계 |
|---|---|---|
| modeling_relation | 수량의 역할·비교 방향·조건·목표·관계식을 잘못 구성 | 문장 해석과 식 세우기를 억지로 구분하지 않고 subtype으로 기록 |
| arithmetic_execution | 적힌 피연산자/연산의 수치 계산 또는 식 변형이 틀림 | 앞의 잘못된 관계식을 일관되게 계산한 것은 이 유형으로 중복 집계하지 않음 |
| unit_conversion | 명시된 단위를 바꾸지 않거나 서로 다른 단위를 같은 단위로 계산 | 잘못된 비율 공식을 선택한 것은 modeling_relation; 단위 자체의 누락/오표기는 이 유형 |
| reasoning_state | 관계·계산이 맞는데 경우 누락/잘못된 논리 전개/상태 갱신이 확인됨 | 근거가 명확할 때만 사용; 모델의 주의력/기억 실패를 추측하지 않음 |
| answer_reporting | 목표의 올바른 값을 풀이에서 명시적으로 구한 뒤 최종 답을 다르게 옮김 | 올바른 중간 수량을 답으로 냈지만 목표 수량을 계산하지 않았다면 modeling_relation/target_mismatch |
| undetermined | 틀린 값은 확인되지만 오류 종류를 특정할 풀이가 부족함 | 답만 제시하거나 유도 없는 중간 값이 틀린 경우, 산술/수량 혼동 등을 임의로 택하지 않음 |
| none_observed | 표시된 계산과 수학적 결론에서 오류가 관찰되지 않음 | strict-label이 오답이어도 형식만 어긴 경우 가능; 수학적 완전성을 증명한 뜻 아님 |

### 출력/종료 문제 `output_issues`

독립적인 multi-label 축이다: `inline_final_marker`, `missing_final_marker`, `repetitive_loop`, `truncated`, `none`. 형식과 수학 오류가 동시에 있을 수 있다. 파싱에 실패했다고 모든 사례를 하나의 수학 오류 유형으로 덮지 않는다.

### Reference 문제 `reference_issue`

`none`, `ambiguous_objective`, `question_gold_mismatch`, `potential_premise_ambiguity`. 최종 gold와 다르다는 사실만으로 오류 원인을 지어내지 않는다. 질문과 gold가 충돌하면 별도 검수 대상으로 보낸다. 잠재적 모호성 flag는 gold 오류 확정이 아니다. 기존 binary label은 이 pilot에서 자동 수정하지 않는다.

## 근거와 최초 오류

- `evidence_quote`는 실제 response의 부분 문자열이어야 한다. 복수로 나오면 가장 이른 출현을 명시적인 anchor로 사용한다.
- 근거의 역할을 `first_observable_error`, `incorrect_final_unlocalized`, `format_violation`, `reference_review_context` 중 하나로 구분한다.
- `first_observable_error_span`은 명확히 잘못된 문장/식이 있는 경우에만 기록한다. 답만 틀렸고 앞의 오류를 특정하지 못하면 이 필드는 null이다. 잘못된 중간 값의 위치는 알아도 유도가 없어 원인 유형은 undetermined일 수 있다. 이 span은 모델 내부에서 최초 실패가 일어난 위치를 뜻하지 않는다.
- 주 오류는 텍스트상 가장 이른 근거로 정하고, 독립적인 추가 오류는 secondary label/subtype으로 남긴다. 단순 전파 오류는 새 원인으로 세지 않는다.
- `certainty`의 high/medium/low는 주석자의 비보정 판단이며 확률이 아니다. 모든 항목은 `human_review_status=not_reviewed`다.

## 실제 Pilot의 경계 사례

- P03/P08/P12: 답만 있어 계산/해석 오류를 구분할 수 없음 → undetermined.
- P05: 올바른 거리/속도 관계를 적었지만 계산을 생략하고 잘못된 숫자를 출력 → undetermined; 단순 산술/답 복사를 임의로 택하지 않음.
- P01/P09: 잘못된 중간/전체 수량은 찾을 수 있으나 해당 값의 유도가 없으므로 원인 유형은 undetermined.
- P10/P19: 수학적 결론은 reference와 같지만 final-line 규약 위반 → none_observed + 출력 형식 라벨.
- P15: 질문은 chapters/day를 묻지만 gold 44는 days/chapter에 해당 → question_gold_mismatch. 모델의 식도 틀렸으므로 별도 수학 오류는 존재함. 기존 binary label은 그대로 보존.
- P13: 초기 5명인데 63명이 내렸다는 서술은 별도 검수 flag. while에 따른 승하차 순서가 명시되지 않아 물리적 불가능으로 확정하지 않는다. 모델은 탑승 인원 72를 구하고, 물은 순증가 9 대신 답했으므로 target_mismatch도 기록.

## 다음 분석 원칙

1. fit 자료로 codebook을 정한 후 고정하고 나머지를 주석한다. 먼저 팀원이 일부를 독립 검수해 유형 경계와 불일치를 확인한다. 같은 assistant의 재검토를 독립 annotator 간 agreement로 보고하지 않는다. 이 pilot은 알려진 strict 오답을 골랐으므로 judge 자체의 오답 탐지 정확도 평가도 아니다. 별도 정답 control 사례를 섞은 검수도 필요하다.
2. 수학 오류 유형 분석에서는 format-only 및 reference 문제/불확실 사례를 별도 보고한다. 불확실 사례를 억지로 한 클래스에 배정하지 않는다.
3. **오답들끼리** 유형을 구분하는지를 주 평가로 삼고, length/confidence/question-only baseline과 hidden을 비교한다. 정답 대 오답 구분을 오류 유형 분류 성능으로 오해하지 않는다.
4. 동일 질문의 8개 답변은 같은 fold에 둔다. 데이터셋별 결과도 분리해 유형-데이터셋 상관을 오류 신호로 오해하지 않도록 한다. Macro-F1, per-class precision/recall 및 충분한 표본에서의 class-vs-rest AUROC를 보고한다.
5. 20개 pilot으로 새 classifier를 학습하거나 hidden 패턴이 있다고 주장하지 않는다. 각 유형의 표본을 확보하고 codebook/검수 후 연구를 확장한다. 기존 종료 hidden은 오류 후의 상관 신호이며 오류의 인과 메커니즘을 증명하지 않는다.

LLM judge의 reasoning 한계와 bias에 대한 참고: https://arxiv.org/abs/2306.05685 . 이 논문의 preference agreement 수치를 수학 오류 유형 라벨의 정확도로 전용하지 않는다.
