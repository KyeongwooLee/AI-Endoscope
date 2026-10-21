# 전체 오류 유형 주석 v1

**상태: 현재 대화 assistant의 직접 텍스트 검토로 작성한, 인간 검수 전 초안.**

**Human review: NOT REVIEWED. 검증된 정답 라벨이나 내부 오류 원인의 ground truth로 사용하지 않는다.**
모든 행의 `human_review_status`는 `not_reviewed`다. 파일 연결·근거 구절의 무결성은 검사했지만,
사람이 오류 유형의 타당성을 확인하거나 독립 주석자 간 일치도를 측정한 데이터는 아니다.

## 다운로드와 문서 안내

- [전용 GitHub Release: error-taxonomy-full-v1-draft](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft): **인간 검수 전 prerelease**. 전체 ZIP과 개별 JSONL/문서, `SHA256SUMS.txt` 제공.
- [전체 묶음 다운로드](https://github.com/KyeongwooLee/AI-Endoscope/releases/download/error-taxonomy-full-v1-draft/error-taxonomy-full-v1.zip): 압축을 풀고 이 README → CODEBOOK → labels 순서로 읽는다.
- [labels.jsonl만 다운로드](https://github.com/KyeongwooLee/AI-Endoscope/releases/download/error-taxonomy-full-v1-draft/labels.jsonl): 실제 질문·응답·reference·주석 1,100행.
- 코드 저장소에는 `sunghun` 브랜치의 문서와 기준을 올리고, 데이터 본문은 Release에서 배포한다. `main`은 변경하지 않는다.

**기존 `gsm8k-train-features-v1` Release와 다른 데이터다.** 기존 것은 greedy 생성한 공식 train 7,473문제의 feature이고,
이번 것은 후속 sampling 실험의 GSM8K/SVAMP 오답에 붙인 주석이다. 기존 Release의 feature에 행 순서나 질문 ID만으로 연결하면 안 된다.
같은 질문이어도 생성 답변이 다르다. 이번 source sampling/posthoc 실험의 **응답 단위 `id`와 `response_sha256`**를 확인해야 한다.

## 대상

- 저장된 GSM8K 4,000답변 중 원본 오답 1,024개에서 형식 실패 128개를 제외한 **896개**.
- 저장된 SVAMP 1,000답변 중 원본 오답 238개에서 형식 실패 34개를 제외한 **204개**.
- 합계 **1,100개 전부 주석 완료**. 전체 과거 GSM8K 실행의 모든 오답이 아니라, 이 두 sampling/transfer 실험의 오답이다.
- 원본 `answer_correct=false`, gold, 응답, 모델 feature 파일은 수정하지 않았다. 새 주석은 별도 overlay다.
- 고유 질문 517개. 동일 질문·완전히 동일한 응답만 중복 검토 단위로 합쳐 1098단위를 읽고, 중복 2행에 같은 주석을 연결했다.
- 별도 GPU judge, 외부 judge API, 자동 오류분류 규칙, 서브에이전트는 사용하지 않았다. 준비/검증 코드는 원본 연결·중복 합치기·정확한 구절 검사·집계만 수행한다.

## 어떻게 labeling했는가

1. 원본 실험에서 `answer_correct=false`이고 `analysis_eligible=true`인 답변을 선택했다. 모두 최종 숫자 파싱 정상·비잘림이며, 형식 실패 162개는 제외했다. 정답 답변을 섞은 blind judge 평가가 아니다.
2. 기존 fit-only 20개 pilot에서 정한 7개 주 유형 기준을 `CODEBOOK.md`로 고정했다. 전체 검토 전에 보존한 파일의 SHA256은 `MANIFEST.json`에 있다. 전체 검토 전 독립 인간 검수는 거치지 않았다.
3. 저장된 질문·모델의 실제 응답·예측 숫자·gold/reference(및 SVAMP equation)를 assistant가 직접 읽었다. **원본 오답 여부와 reference를 알고 판단**했으며, hidden/confidence/verifier 점수는 보지 않았다.
4. 출력 텍스트에서 관찰되는 주 오류 1개를 고르고, 실제 응답에서 근거를 인용하고, 이유와 교정 계산을 작성했다. 풀이가 부족하면 `undetermined`, 타당한 해석에서 오류가 관찰되지 않으면 `none_observed`로 남겼다. gold/전제가 의심스러우면 별도 `reference_issue`를 붙였다.
5. 같은 질문의 완전히 동일한 응답 2행에만 동일 주석을 전달했다. 비슷한 응답·같은 숫자라는 이유로 다른 답변의 라벨을 복사하지 않았다. 나머지 1,098개 검토 단위는 직접 읽고 주석했다.
6. 손으로 작성한 압축 주석 원장 `drafts/*.json`을 코드가 펼쳐 JSONL로 만들었다. ID 연결, 누락/중복, 원문·인용 위치·해시, fold 일관성을 검사했다. **코드가 오류 유형을 추론하거나 자동 판정한 것은 아니다.**

단일 고정 judge prompt를 API로 반복 호출한 방식이 아니라 이 대화 안에서 기준에 따라 직접 읽고 작성한 방식이다.
`annotator=assistant_in_current_conversation`으로 기록했으며, 정확한 judge 모델 버전·독립 실행 seed·API 요청 로그는 기록하지 않았다.
따라서 동일 모델 호출을 통한 완전한 주석 재생을 주장하지 않는다. 원문 packet, 고정 기준, 주석 원장으로 판단 경위를 확인할 수 있다.

`certainty`는 정답 확률이나 독립적인 신뢰도 측정이 아니다. 저장 코드가 `undetermined`/`none_observed` 또는 reference 플래그가 있는 경우 `medium`,
그 외에는 `high`를 기본 부여했다. `high` 역시 인간 검수 완료 또는 보정된 정확도를 뜻하지 않는다.

## 유형별 집계

| 유형 | 코드 | GSM8K | SVAMP | 합계 |
|---|---|---:|---:|---:|
| 관계·조건·목표 구성 | `modeling_relation` | 594 | 138 | 732 |
| 산술 실행 | `arithmetic_execution` | 60 | 16 | 76 |
| 상태 갱신·논리 전개 | `reasoning_state` | 38 | 11 | 49 |
| 단위 변환 | `unit_conversion` | 28 | 0 | 28 |
| 최종 답 옮기기 | `answer_reporting` | 0 | 0 | 0 |
| 오류 유형 판단 근거 부족 | `undetermined` | 141 | 28 | 169 |
| 관찰된 수학 오류 없음 | `none_observed` | 35 | 11 | 46 |

숫자만 제시하거나 올바른 계획 뒤 계산 없이 틀린 숫자를 출력한 경우, 산술 오류라고 단정하지 않았다. `undetermined`는 라벨링 누락이 아니라 근거 부족 판정이다.
`answer_reporting=0`은 이 보수적 기준에서 확정하지 않았다는 뜻이다. 올바른 최종 목표 수치를 명시적으로 도출한 뒤 단순히 다르게 옮긴 경우만 해당하며, 이후 잘못된 새 계산이나 다른 목표를 선택한 경우와 구별했다.

## Reference 검수

**155답변 / 89질문**에 문제 전제·목표·gold 검수 플래그가 있다. 모델 오류와 동시에 있을 수 있으며 유형별 표와 중복 집계된다.
그중 `none_observed` 46개는 gold 오류, 단위 등가, 타당한 다른 해석 등으로 수학 오류를 확인하지 못한 경우다. 이 수를 인간이 확정한 정답 수 또는 확정 gold 오류 수로 읽으면 안 된다.
플래그는 확정 gold 오류뿐 아니라 모호한 수량 범위, 음수 재고 같은 비현실적 전제, 질문과 gold의 단위 불일치를 포함한다. 전제 검사는 완전하지 않으며, `reference_issue=none`도 reference의 정당성을 보증하지 않는다.

대표 사례:

- `train-00228`: 운전 120명, 대중교통 40명의 차이는 80인데 gold는 40.
- `train-04796`: 20대 + 5×20대 = 120인데 reference 풀이 마지막 gold만 140.
- `train-01258`: 모델의 22.5시간은 gold의 1,350분과 같은 시간. 질문에 답 단위가 지정되지 않음.
- `svamp-chal-676`, `svamp-chal-333`: 질문은 chapters/day인데 gold는 days/chapter.
- `svamp-chal-680`: 4−2+3=5인데 gold는 1.
- `svamp-chal-730`: 첫 chapter를 물었는데 gold는 둘째 chapter의 페이지 수.

## 파일과 읽는 순서

1. `README.md`: 대상, 결과 집계, 해석 주의점.
2. `CODEBOOK.md`: 전체 검토 전에 보존한 pilot v0.1 분류 기준과 경계 사례. 문서의 20개 pilot 선택 설명은 과거 맥락이며, 이번 대상 범위는 이 README의 1,100개를 따른다.
3. `labels.jsonl`: **전수 1,100개**, 한 줄에 한 답변. 실제 질문·모델 응답·reference·유형·근거·교정 설명 포함.
4. `reference_review.jsonl`, `REFERENCE_REVIEW.md`: reference/전제 검수 대상의 기계 판독 파일과 읽기용 문서.
5. `type_analysis_candidates.jsonl`: reference 플래그, 판단 불가, 오류 미관찰을 제외한 보수적 분석 후보. 아직 검수 전이므로 확정 학습셋이 아님.
6. `AUDIT.json`, `MANIFEST.json`: 개수, fold별 집계, SHA256, 검증 결과와 원본 packet 정보.
7. `drafts/*.json`: assistant가 직접 작성한 압축 주석 원장. `packet.json`, `review_units.json`은 원본 연결 및 검토 재현용.

`excluded_format_failures.json`은 제외한 162개 원본 ID/품질 메타데이터다. `SOURCE_MEMBER_HASHES.json`에는 원본 generation 파일별 SHA256이 있다. `INDEPENDENT_INTEGRITY_AUDIT.json`은 별도 검사 코드로 원본 응답·ID·span을 대조한 결과이며, 독립 주석자에 의한 의미 검수가 아니다.
`error-taxonomy-full-v1.zip`에 전체 결과를 묶었다. AI에 전달할 때 README + CODEBOOK + labels.jsonl을 먼저 주면 된다. 이 묶음에는 문제·응답·reference·주석이 포함되지만, 기존 모델 hidden/confidence feature 배열이나 모델 가중치는 포함되지 않는다.
압축 파일의 `tools/`는 작성·검증 코드의 사본이다. 원본 실험 백업 경로를 사용하는 작업용 코드이며, 묶음만으로 원본 대조 검사를 재실행할 수는 없다. JSONL을 읽는 데에는 GPU나 별도 모델이 필요 없다.

## 주요 필드

- `id`, `question_id`, `dataset`, `fold`: 기존 저장 feature와 연결할 키. 동일 질문의 여러 답변은 반드시 같은 fold에 둔다.
- `answer_correct`, `gold_number`, `predicted_number`, `analysis_eligible`: **원본 실험의 정오답 및 품질 판정**. 여기서 `analysis_eligible=true`는 형식 정상·비잘림을 뜻할 뿐, 새 오류 유형의 신뢰성을 뜻하지 않는다.
- `math_error_type`: 관찰 가능한 주 오류 1개. `subtypes`는 비교 방향, 목표 수량, 시간 범위 등 세부 메모이며 일부 후속 독립 오류도 포함한다. 완전한 다중 오류 원장이나 고정된 subtype 클래스 목록은 아니다.
- `evidence_quote`, `evidence_span`: 실제 response의 부분 문자열과 Python 문자열 기준 `[시작, 끝)` 문자 위치. UTF-8 바이트 위치나 토큰 위치가 아니다.
- `first_observable_error_span`: 주석자가 선택한 최초 관찰 오류 **후보 구절**의 위치. 독립 검증된 최초 오류 또는 모델 내부 최초 실패 위치가 아니다. 판단 불가/오류 미관찰에서는 null로 보수적으로 유지한다.
- `explanation`, `corrected_step`: 오류 판단 이유와 올바른 계산 또는 reference 해석. 영문으로 기록해 도구/AI에 그대로 전달할 수 있다.
- `reference_issue`, `reference_review_needed`: 질문·reference 검수 여부. 원본 binary label 자동 수정에는 사용하지 않는다.
- `recommended_type_analysis_candidate`: 형식 정상 외에, reference 플래그 없음 + 관찰 가능한 오류 유형인 임시 후보 여부.
- `certainty`: high/medium의 비보정 주관적 확실도. 수치 확률이 아니며, 모두 `human_review_status=not_reviewed`.
- `response_sha256`, `source_member_sha256`, `review_unit`: 원본 응답/원본 generation 파일 식별 및 정확한 중복 추적.

## Feature 연결과 후속 분석

현재 보수적 후보는 **786개**다. 관계 구성·산술·단위 변환·상태/논리 주석을 기존 완료 hidden/confidence/length와 `id`로 연결할 수 있다.
이 단계에서는 classifier 학습이나 hidden 패턴 검정은 수행하지 않았다. 주석 작성 중 hidden·confidence·verifier 점수는 보지 않았다.
모든 fit/validation/test 답변을 요청에 따라 주석했다. 분류 기준은 기존 fit-only pilot을 기준으로 유지했지만 독립 인간 검수 없이 확장한 v1 초안이다. 이후 test 사례를 보고 클래스/feature를 다시 정하면 새 탐색으로 기록하고 기존 test를 미사용 최종검증셋으로 주장하지 않는다.
분석 전 일부를 팀원이 독립 검수하고, 동일 질문 단위 분할, 데이터셋별 성능, 유형별 표본 수, length/confidence/question-only baseline을 확인해야 한다. 반복 답변 때문에 1,100행이 1,100개의 독립 질문인 것은 아니다.
오류 유형은 출력 텍스트의 관찰적 분류다. 모델 내부의 인과적인 실패 원인이나 올바른 추론 과정을 직접 증명하지 않는다. 판단 불가 169개를 임의의 오류 클래스에 편입하지 않는다.

## 팀원 검수와 버전 관리

- 우선 유형별·데이터셋별·reference 플래그별 사례를 나눠 사람이 검수한다. 가능하면 기존 주석을 보기 전에 독립 판정하고, 이후 불일치를 비교한다. 같은 assistant의 재검토는 독립 annotator 검증이 아니다.
- 786개 후보도 **검수 전**이다. 소수 클래스·클래스 불균형과 질문 난이도/길이 영향을 확인하고, 검수 없이 확정 학습셋이나 최종 benchmark로 부르지 않는다.
- 수정 시 원본 `id`, 기존 라벨, 수정 라벨, 검수자, 수정 이유를 별도 기록한다. 이 v1 초안을 조용히 덮어쓰지 않고 새 버전/Release로 배포한다.
- AI에게도 README + CODEBOOK + labels를 함께 전달하고, `not_reviewed`, reference 플래그, 판단 보류 사례를 유지하도록 요청한다. AI가 만든 라벨을 검증된 인간 라벨로 요약하지 않도록 한다.

## 출처

문제와 reference의 출처는 [GSM8K / OpenAI](https://github.com/openai/grade-school-math),
[SVAMP / Arkil Patel 외](https://github.com/arkilpatel/SVAMP)다. SVAMP 원본은
`78e727689e1c1bebfc4be39c446898e8e10b0518` revision을 사용했다.
응답 생성 모델은 [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)이며,
이번 source 실험은 sampling(T=0.7, top_p=0.95, top_k=0, 최대 생성 1,024토큰)이다. 모델 가중치는 배포하지 않는다.
GSM8K는 500질문×8응답, SVAMP는 1,000질문×1응답에서 위 기준으로 오답을 선택했다.
`ATTRIBUTION.md`, `LICENSE-GSM8K.txt`, `LICENSE-SVAMP.txt`를 함께 보존한다. 출처 고지는 전체 저장소나 모델의 라이선스를 새로 지정하는 것이 아니다.

## 로딩 예시

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path("labels.jsonl").read_text(encoding="utf-8").splitlines()]
by_id = {row["id"]: row for row in rows}
assert len(rows) == len(by_id) == 1100
for row in rows:
    start, end = row["evidence_span"]
    assert row["response"][start:end] == row["evidence_quote"]
candidates = [row for row in rows if row["recommended_type_analysis_candidate"]]
```

정답 reference, 주석 설명, 오류 근거, 교정 풀이, 원본 정오답 값은 **분류기의 입력 feature로 넣으면 안 된다**. 학습 목표/검수 전용이다. 저장된 모델 내부 feature만 별도로 선택해 입력한다.

`labels.jsonl` SHA256: `6137ffc9efa9482260039b08378f8b479710753013f46857fd16b5b30c23193d`
