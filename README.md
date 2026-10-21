# AI-Endoscope

## 처음 읽는 팀원을 위한 안내

이 `sunghun` 브랜치는 Qwen2.5-1.5B-Instruct가 GSM8K를 직접 풀 때의 정답/오답 신호를
조사한 실험 코드와 문서를 담고 있다. **처음에는 아래 1번 실험 개요를 읽고, 목적에 따라 이동하면 된다.**
전체 train 7,473문제의 데이터는 Git 파일이 아니라
[GitHub Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1)에서 받는다.

**추가 공개: [오류 유형 주석 1,100개 (인간 검수 전 초안)](experiments/sunghun/error-taxonomy-full-v1/README.md)**
후속 sampling 실험의 GSM8K 896개 + SVAMP 204개를 assistant가 직접 분류했다.
[전용 prerelease에서 다운로드](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/error-taxonomy-full-v1-draft)할 수 있다.
같은 릴리즈에 **GSM8K sampling 4,000응답·SVAMP 1,000응답 전체 원문/feature**도 함께 올렸다.
[다운로드 안내](experiments/sunghun/error-taxonomy-full-v1/DOWNLOAD_GUIDE.md)에서 필요한 파일을 고르면 된다.
인간 검수·독립 의미 검증 전이며, 기존 7,473문제 feature Release와 생성 답변이 다른 데이터다.

**SVAMP inference·양방향 전이 데이터:** [설명](experiments/sunghun/posthoc-transfer-data-v1/README.md) /
[Release 다운로드](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/posthoc-transfer-data-v1).
SVAMP 1,000응답, 별도 GSM8K sampling 4,000응답, 완료 hidden/confidence feature와 양방향 verifier 결과를 제공한다.
기존 전체 train 7,473응답을 줄인 것이 아니며, 두 생성 실험은 별개다.

### 문서별 내용

| 문서 | 어떤 정보가 있는가 |
|---|---|
| 1. [실험 개요: experiments/sunghun/README.md](experiments/sunghun/README.md) | 연구 목적, 고정 모델·생성 설정, 1~8차 실험 타임라인, 실제 확인한 결과, 전체 train 추출 완료 집계와 해석의 한계. 현재 상황을 파악하는 출발점 |
| 2. [DATA_README.md](experiments/sunghun/DATA_README.md) | 배포 데이터의 정확한 범위, 다운로드/AI 전달법, prompt, 파일 구조, ID join, t와 layer 정의, hidden·trajectory·logit-lens feature 계산식, 정답 label, 파생 scalar feature, 누수 방지, CPU 로딩·해시 검증 |
| 3. [REPRODUCE.md](experiments/sunghun/REPRODUCE.md) | 환경·의존성 설치, GPU용 설정, 실험별 실행 순서와 필요한 입력 데이터, 백업 도구, 과거 helper의 경로 제약, Release 다운로드와 패키징·로더 테스트 |
| 4. [최초 study README](experiments/sunghun/gsm8k-logit-study/README.md) | 첫 logit-lens study의 설정, VESSL 준비, 생성과 causal replay의 구분, pilot→exploration→test 절차, 원래 feature와 결과 폴더 구조, 원격 회수 및 테스트 |
| 5. [CLASSIFIER.md](experiments/sunghun/gsm8k-logit-study/CLASSIFIER.md) | 생성이 끝난 뒤 confidence/internal/답변 길이를 사용하는 사후 classifier, logistic·MLP 학습 설정, split·전처리·평가 방법과 초기 결과 |
| 6. [MIDLAYER_CLASSIFIER.md](experiments/sunghun/gsm8k-logit-study/MIDLAYER_CLASSIFIER.md) | layer14 이하와 현재 prefix에서 관측 가능한 scalar만 사용하는 조기 classifier. 사후 feature와의 차이, 미래 정보 제외 규칙과 학습·평가 설계 |
| 7. [HIDDEN_PROBES.md](experiments/sunghun/gsm8k-logit-study/HIDDEN_PROBES.md) | h14 / [h14,h14−h13] / [h7,h10,h14] 세 probe 입력, 추출·학습 protocol, layer 변화량 및 1/x 가설 분석, 저장·복구와 한계 |
| 8. [FEATURE_ABLATION.md](experiments/sunghun/gsm8k-logit-study/FEATURE_ABLATION.md) | 벡터 크기·집중도, 인접/depth 간 변화, 변화량 곡선, entropy 등의 안정성 feature 그룹. 같은 대상군에서 비교하는 ablation, fit/validation과 선택 편향 주의사항 |
| 9. [ENTROPY_CONFIRMATION.md](experiments/sunghun/gsm8k-logit-study/ENTROPY_CONFIRMATION.md) | 탐색에서 고른 entropy 후보를 고정한 뒤 새 train 1,000문제에서 확인하는 실험. 후보·학습·capture 품질·평가 기준과 원격 실행 |
| 10. [TRAIN_ALL_EXTRACTION.md](experiments/sunghun/gsm8k-logit-study/TRAIN_ALL_EXTRACTION.md) | 공식 train 7,473개 확대 추출의 설계. 기존 답변 재사용, compact feature 범위, test 제외, 품질 gate, 배치 저장·백업·resume, 이후 분석의 경계 |
| 11. [오류 유형 주석 README](experiments/sunghun/error-taxonomy-full-v1/README.md) / [CODEBOOK](experiments/sunghun/error-taxonomy-full-v1/CODEBOOK.md) | 후속 sampling 응답 1,100개의 오류 유형 초안. 선택 범위, assistant 직접 labeling 절차, 유형별 수, 근거·교정·reference 플래그, 인간 검수 전 상태, 로딩·누수 방지. 기존 feature Release와 별도 |
| 12. [SVAMP inference·전이 데이터 README](experiments/sunghun/posthoc-transfer-data-v1/README.md) | SVAMP 1,000응답과 비교 GSM8K 4,000응답, 생성 protocol, 완료 h28/confidence/length 계산, 양방향 전이 결과, 7,473개와의 차이, 오류 유형 주석 연결과 로딩 |

### 추천 읽기 순서

- **현재까지 무엇을 했고 결과가 어땠는지:** 1번 실험 개요 → 관심 있는 실험의 4~10번 문서.
- **데이터를 분석하거나 AI에게 전달할 때:** 1번 실험 개요 → 2번 DATA_README → 아래 JSON 보조 자료 → Release의 `AI_STARTER.zip`. raw hidden 분석에는 전체 데이터 ZIP도 필요하다.
- **코드를 실행하거나 후속 실험을 할 때:** 1번 실험 개요 → 2번 DATA_README → 3번 REPRODUCE → 실행할 실험의 4~10번 문서.
- **연구 진행 순서대로 자세히 볼 때:** 4번 최초 study → 5번 사후 classifier → 6번 조기 scalar classifier → 7번 hidden probe·layer 변화 → 8번 ablation → 9번 새 문제 confirmation → 10번 전체 train 추출.
- **새 오류 유형 데이터를 사용할 때:** 11번 주석 README → CODEBOOK → 전용 Release의 `labels.jsonl`. 검수 전 초안이며, 기존 7,473문제 배열과 행 순서로 연결하지 않는다.

### 데이터 분석용 보조 자료

- [DATA_SCHEMA.json](experiments/sunghun/DATA_SCHEMA.json): 기계 판독용 필드·배열 축 순서·계산식·허용/금지 predictor 사전.
- [dataset_summary.json](experiments/sunghun/dataset_summary.json): 실제 데이터 개수, 정답/오답·checkpoint별 집계, origin별 provenance, 36개 scalar feature의 정확한 열 순서.
- [데이터 Release](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1): 전체 데이터 약 701MB, AI용 묶음 약 5.3MB, standalone README/schema/summary와 SHA-256 검증 파일.

**문서의 범위를 구분해야 한다.** `gsm8k-logit-study/`의 문서는 당시 실험 설계와 실행 안내를 보존한 것이다.
현재 배포된 데이터의 실제 필드·계산식은 `DATA_README.md`와 `DATA_SCHEMA.json`을 기준으로 보고,
확인된 실험 결과와 현재 완료 상태는 1번 실험 개요에서 확인한다.
초기 study의 모든 feature나 학습된 probe가 이번 Release에 들어 있는 것은 아니며,
전체 train 7,473개로 새 probe를 재학습한 결과는 아직 없다.

## Experiments

- [성훈: Qwen2.5-1.5B / GSM8K 오답 위험 예측 실험](experiments/sunghun/README.md)
  - 모델이 직접 생성한 답변의 logit-lens 특징 분석, 답변 생성 후 정오답 분류,
    생성 중간 시점의 특징·은닉 상태를 이용한 예측, 층별 변화량 분석, 특징별 비교와 엔트로피 후보 검증.
  - RTX 3090 실행 설정, 오프라인 테스트, 실제 실험 실행·백업 보조 도구와 재현 방법 포함.
  - 이 브랜치에는 코드와 문서를 올렸으며, **실험 데이터는 아래 GitHub Release에서 다운로드할 수 있다.**
  - [실험 데이터 다운로드 (Release)](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/gsm8k-train-features-v1):
    공식 train 7,473문제의 전체 데이터 약 701MB와 AI 전달용 묶음 약 5.3MB. 공식 test와 학습된 모델은 포함하지 않는다.
  - [데이터·특징 상세 설명](experiments/sunghun/DATA_README.md): 파일 구조, 특징 계산식, 정답 판정 기준, 분석 시 주의사항과 로딩 방법.
