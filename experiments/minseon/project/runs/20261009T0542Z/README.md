# ErrorProbe run 20261009T0542Z

한국어 결과: `REPORT_KO.md`, 회의용 요약: `MEETING_KO.md` (평가 완료 후 생성).
실행 지시문: `../../markdown files/ERRORPROBE_TASK.md`.

## 재실행과 복구

```bash
cd /home/coder/yaicon-errorprobe/runs/20261009T0542Z
bash setup.sh
bash run.sh prepare > prepare.log 2>&1
bash run.sh pilot > pilot.log 2>&1
# pilot_validation.json의 passed 및 pilot_tokens_01..05.tsv를 확인한다.
bash run.sh main > main.log 2>&1
bash run.sh evaluate > evaluate.log 2>&1
```

기존 run의 manifest와 생성 record는 보존한다. pilot/main은 JSON+NPZ가 모두 있는 샘플을 건너뛰고 이어서 실행한다. 별도 실험에는 새 run directory에 코드/스크립트만 복사한다. evaluate 재실행은 해당 run의 파생 평가 파일을 재작성한다. 선택 모델은 train에만 fit하며 validation 선택 후 train+validation 재학습 없이 test를 평가한다. 모든 실제 패키지 버전은 packages.json, 설치 과정은 install.log에 남긴다.

/tmp 가상환경 및 HF 캐시는 재시작 후 없어질 수 있다. setup.sh에서 /usr/bin/python은 **새 venv 생성에만** 사용하고 모든 설치/실행은 지정된 venv Python을 사용한다. 더 정확한 패키지 재구성에는 환경 준비 후 저장되는 requirements-lock.txt를 사용한다. 모델/데이터 revision은 config.json으로 고정한다. 기존 manifest가 있으면 prepare는 다시 추출하지 않는다.

초기 홈 디스크 20KB로 실험 불가하여 `/home/coder/.cache/pip`와 `.cache/code-server`를 `/tmp/yaicon-errorprobe.xCaEjb/relocated-cache/home-pip` 및 `code-server`로 보존 이동하고 원 경로에 symlink를 남겼다. 내용 삭제는 없었다. 캐시는 임시 보관이므로 컨테이너 재생성 후에는 재생성이 필요하다. 원 위치 복구는 충분한 공간 확보 후 symlink만 제거하고 보관된 디렉터리를 원 위치로 이동한다. 자동 복구 스크립트에서 사용자 환경을 임의 변경하지 않는다.

## 고정 규칙 (생성 전 작성)

- GSM8K 공식 train의 질문 공백을 정규화해 동일 질문을 묶고 정렬된 그룹에서 numpy seed 42로 550개 그룹을 무복원 추출. 그룹 첫 source row를 대표로 사용하며 모든 중복 source index를 manifest에 보존. 첫 50 pilot, 다음 300 train, 다음 100 validation, 마지막 100 test. pilot은 모든 fitting/평가에서 제외.
- 모델 prompt에 gold를 넣지 않음. 영어 system/user prompt 및 tokenizer chat template을 사용. greedy, FP16, batch 1, max_new_tokens=512. 실제 원본 prompt/generated IDs 보존.
- 최종 답은 대소문자를 무시한 `Final answer: <decimal>` 독립 행 **정확히 1개**이고 이후 비공백 텍스트가 없어야 함. 숫자 앞 `$`와 마지막 문장 마침표 허용. 부호/천단위 쉼표/소수 허용, fraction/exponent/boxed/단위/markdown 장식은 미지원. Decimal로 쉼표 제거, 부호/소수 정규화 후 정확 비교. 추출 실패=오답. 생성 결과를 보고 규칙을 수정하지 않음.
- 숫자 정규식은 experiment.py의 NUM. +/- 직전 비공백 문자가 숫자 또는 `)]`이면 이항 연산자로 분리하고 나머지 +/-는 unary sign. 연산자는 `+ - * / = × ÷ −`. unicode minus는 연산자로 취급하며 숫자 부호 정규화에 넣지 않음. 자연어 하이픈도 연산자로 세는 단순 휴리스틱 한계가 있음.
- step=비어 있지 않은 각 줄. `Reasoning:` 또는 `Final answer:`로 시작하는 줄 전체는 제외. 문장/산술 의미 단위가 아닌 사전 고정 줄 기반 규칙.
- 텍스트 span은 [start,end) 문자 범위. 원본 생성 token 중 special IDs를 제외한 ID 배열과 텍스트 재encoding IDs가 **완전히 같은지** 확인한 뒤 offsets만 이용. 다르면 원본 IDs의 단조 증가 prefix decoding 사용, replacement-character 또는 비단조 경계는 alignment 실패. forward에는 항상 원본 IDs 사용.
- span과 한 글자 이상 겹치는 token을 포함. 숫자와 주변 공백이 한 token이면 그 token 전체 NLL을 사용. token이 여러 span과 겹칠 수 있음. 각 숫자 span의 mean을 구해 max span-mean 계산. number_mean_nll은 모든 숫자 token union의 mean (token 가중). operator/step도 overlap 규칙 동일. final−prior는 final 시작 이전 종료한 모든 숫자 span token union NLL 평균을 뺌.
- confidence에서는 prompt/padding/EOS 및 모든 special ID 제외. generated token i의 predictor는 absolute index len(prompt)+i−1. attention mask는 원본 sequence 전체 1, position은 모델 기본 causal position. padding은 없음.
- P0=prompt 마지막 token. P1=최종 숫자 첫 overlap token 바로 이전, P2=마지막 overlap token 포함. P1 span 위치는 사후 정해진 위치이며 reasoning 이후일 수 있음. 경계 token에 공백 등이 붙으면 그 전체 token 경계로 정의.
- Qwen 실제 28 blocks 확인 후 model.layers[6,13,20,27] output hook 사용. 전부 final norm 이전 raw output. hidden vector 저장은 float16, probe 입력과 delta 계산은 float32 이상. model.eval(), inference_mode, requires_grad=False.
- delta mask 우선순위는 final > number > operator > 일반 text; mutually exclusive. text에는 나머지 비특수 문자 token 포함. 3개 선택 block 구간별 final/text 각각 mean relative L2 및 mean(1−cosine): 12개. number/operator는 3구간 mean의 평균 각각 두 지표: 4개. number/operator의 구간별 mean relative L2의 max: 2개. 총 18개. relative L2 분모 epsilon=1e−8, cosine 분모에도 epsilon=1e−8.
- 없는 span feature는 null→NaN으로 저장. sklearn train-only median imputation, missing indicator(모든 scalar 열에 고정적으로 추가; train에서 missing이 없던 열도 포함), keep_empty_features=True, StandardScaler, logistic regression(liblinear,max_iter=3000). PCA 없음. 단일 class train/validation이면 분석 중단하고 한계를 보고.
- 공통 집합=final parsing+token alignment+final token span 확보. optional operator/previous number가 없어도 제외하지 않고 impute. B0/B1/B2/H/H+D와 hidden-only P0/P1/P2는 동일 공통 집합 사용. 추가 all B0/B1/P0로 전체 집합 평가. P0/P1 hidden-only에는 어떤 미래 통제 변수도 없음.
- C=[.01,.1,1,10], hidden 후보 block=[7,14,21,28]. validation AUROC 최대 선택, 동점이면 낮은 layer/작은 C 우선. 모든 선택을 selected_models.json에 기록한 뒤 test 평가. train으로만 scaler/imputer/probe fit.
- AP=sklearn average_precision_score. recall@20%=stable descending score 정렬에서 ceil(.2*N)개 선택, 동점은 고정된 ID 순. paired bootstrap 2000회, seed43; 단일 class draw 제외. frozen selected models에 대한 test-row percentile CI이며 학습·모델 선택 변동성은 반영하지 않음.
- Label shuffle: B2/H 각각 train 및 validation labels를 독립 permutation한 20회 재실험, 매번 원래 전체 layer/C 선택을 재실행. test label은 원본이며 sanity evaluation 외에는 사용하지 않음.
- confidence 구간별 탐색은 train 전체 response NLL 삼분위 경계로 test를 나누고 B2/H 성능을 계산. 소규모 사후 탐색으로 취급.

## 산출물

config/manifest/model_config/generation_config/packages/environment, records/*.json (IDs/text/label/token NLL/offsets/spans/features), hidden/*.npz (선택 vector), pilot_tokens TSV, pilot_validation, checkpoint, 단계별 로그, comparison.csv/png, selected_models/validation_grid/test_predictions, bootstrap, label_shuffle, cohort_counts 및 한국어 보고서. 전체 vocabulary logits나 전체 dataset activation은 저장하지 않는다.

## Pilot 수치 안정성 복구

초기 FP16 eager attention 시도에서 첫 사례의 hidden/NLL NaN으로 저장 실패했다. 실패 로그와 initial eager 코드/hidden을 보존했다. SDPA math backend(float32 intermediate accumulation, model FP16)를 사용하도록 변경하고 모든 NLL/hidden finite 검사를 추가했다. 데이터/parser/probe 설계 변경은 없다. raw_generations에도 추출 전 원본 생성 결과를 보존한다.

## Pilot-only prompt 디버깅

이 run은 `../20261009T0535Z`의 초기 pilot에서 연속적인 최종 행 형식 불이행을 관찰한 후 시작했다. 첫 시도 결과는 원 run에 보존했다. 정답 비교를 이용한 parser 완화는 하지 않고, plain text 및 concise reasoning을 명확하게 요청하는 prompt를 **본 실험 생성 이전**에 고정했다. 동일한 새 prompt로 pilot 50개를 처음부터 실행하고 본 실험 500개에도 사용한다. 원본 split과 parser 규칙은 동일하다. P1 prefix 검증은 첫 5개 샘플이 아니라 첫 5개 span 확보 가능 샘플에서 수행한다.

선택적 static-cache torch.compile 속도 최적화는 별도 pilot token 일치 검사 전에 compiler subprocess crash 및 단일 프로세스 exit 139(segmentation fault)로 실패했다. 본 실험에는 컴파일을 사용하지 않는다. compile_benchmark*.log/json은 이 구현 진단의 기록이며 연구 결과가 아니다.

본 실험 도중 홈 여유 공간은 1GB(decimal) 이상이었으나 더 보수적인 1GiB와 향후 산출물 여유를 확보하기 위해 VS Code의 다운로드 캐시 `~/.vscode-server/data/CachedExtensionVSIXs`(355MiB)를 `/tmp/yaicon-errorprobe.xCaEjb/relocated-cache/vscode-extension-vsix`에 보존 이동하고 symlink를 남겼다. 설치된 extension이나 사용자 설정은 변경하지 않았다. 최종 audit는 1GiB 이상을 요구한다.

평가 전에 결측 indicator를 모든 scalar 열에 고정 추가하는 ColumnTransformer+MissingIndicator(features=all)로 구현을 보완했다. hidden vector는 finite 검증을 통과한 경우만 사용하므로 중복 indicator를 만들지 않는다. imputer/scaler/분류기는 모두 train에만 fit한다. 생성/parser/feature 정의와 layer/C grid는 동일하다.

선택된 11개 probe의 fitted imputer/indicator/scaler/logistic pipeline은 `probes/*.joblib`에 저장한다. 모델명/선택 layer/C는 selected_models.json 참조. 입력 열 순서는 B0/B1/B2 정의 순서이며 H는 B2 후 P2 vector, H+D는 B2 후 D 18개 후 P2 vector, hidden-only는 해당 P0/P1/P2 vector만 사용한다. `format_success`는 최종 숫자 답 행의 형식 성공 여부이며 Reasoning 헤더 존재를 추가 조건으로 요구하지 않는다.

## 최종 산출물과 완료 상태

[회의 요약](MEETING_KO.md), [상세 보고서](REPORT_KO.md), [비교 그래프](comparison.png), [최종 검증](audit.json), [완료 기록](completion.json). 동일한 고정 run을 재개/재평가하려면 `bash finish.sh`를 사용한다. 환경이 사라졌으면 먼저 `bash setup.sh`를 실행한다. 이미 저장된 생성 사례는 다시 생성하지 않는다.

`finalize_report.py`의 해석 문장은 이 run의 확정 수치에만 적용하며 주요 지표가 다르면 중단한다. 새 seed·모델·환경 실험에 해석 문장을 복사하지 말고 새 결과로 작성한다. 새 run에는 experiment.py/setup.sh/run.sh를 기반으로 생성 전 설계를 별도로 고정한다.
