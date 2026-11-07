# SAE error clustering (server execution only)

Qwen2.5-1.5B-Instruct의 저장된 오답을 SAE 특징으로 군집화하는 독립 패키지입니다.
**코드·테스트를 작성했지만 실행하지 않았습니다. GPU 호환성·복원 품질·군집 의미는 미검증입니다.**

## 범위와 출처

- [qwen-gsm8k-self-errors-v1](https://github.com/KyeongwooLee/AI-Endoscope/releases/tag/qwen-gsm8k-self-errors-v1): GSM8K train 3,489문제의 숫자 오답 7,691응답.
- Qwen revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- [공개 SAE](https://huggingface.co/HuggingAnalist/sae-qwen2.5-1.5B-res), revision `2a9297bf085d40d36ae0aad8a3a76a28d0ed6d59`, `layer_17/final_sae`.
- `model.layers[17]` 출력: **0부터 세는 block index 17**, 18번째 블록의 residual. HF `output_hidden_states[17]`와 다릅니다.
- 확인된 config/weight header: standard ReLU SAE, 1536→24576, input에서 b_dec 차감, normalization/reshape 없음. SAE 학습에 사용한 Qwen의 정확한 revision은 metadata에 없어 일치 여부는 미검증입니다.
- 릴리스 hidden.npy의 집계값에는 필요한 토큰별 block 17 출력이 없어 사용하지 않습니다. 저장 prompt/output 토큰으로 재생하며 새 답변은 생성하지 않습니다.
- `tokens.py`는 기존 배포의 `gsm8k/methods/answers.py`에서 복사했습니다. ByteLevel fallback과 validation을 유지하며 별도 원본을 수정하지 않았습니다.
- `rule_type`/reference/gold는 extraction·clustering에 넣지 않습니다. 규칙 라벨은 report에서만 읽으며 사람 검수 정답으로 취급하지 않습니다.

## 서버 준비

이 문서의 명령은 사용자가 서버에서 실행할 명령입니다. 로컬에서 자동 실행되는 작업은 없습니다.
Python 3.11, BF16 지원 CUDA GPU 24GB, RAM 16GB 이상, 가용 디스크 20GB 이상을 권장합니다.
메모리 요구량은 아직 측정하지 않았습니다. BF16 미지원 시 정밀도를 조용히 바꾸지 않고 중단합니다.

```bash
python -m venv .venv-sae
source .venv-sae/bin/activate
# 서버 CUDA와 맞는 torch 2.6.0 wheel을 먼저 설치합니다.
python -m pip install -r sae_error_clustering/requirements.txt
python -m pip check
```

저장소 루트에서 실행하며 config의 상대 경로도 현재 작업 디렉터리 기준입니다.
config의 data_dir/output_dir를 서버 절대 경로로 바꿀 수 있습니다. 실행 도중 config를 수정하지 마세요.

```bash
python -m sae_error_clustering prepare --config sae_error_clustering/config.json
python -m sae_error_clustering extract --config sae_error_clustering/config.json --limit 32
# shards/*.json의 복원 오차, 활성 특징 수, 제외 사유와 메모리 사용을 먼저 확인합니다.
python -m sae_error_clustering extract --config sae_error_clustering/config.json
python -m sae_error_clustering cluster --config sae_error_clustering/config.json
python -m sae_error_clustering report --config sae_error_clustering/config.json
```

prepare는 약 32MB의 텍스트/메타데이터/확신도 파일만 내려받고 고정 SHA256을 검사합니다.
extract에서 모델·SAE를 처음 다운로드합니다. 그 외 단계는 GPU를 사용하지 않습니다.
`--limit`는 extract에만 허용하며 shard_size(기본 32)의 양의 배수입니다.
완료된 shard는 해시·ID·shape·signature를 검사한 뒤 건너뜁니다. signature는 코드·Python·패키지·GPU·설정·입력 해시를 포함합니다.
변경된 환경으로 이어쓰지 말고 새 output_dir를 사용하세요. OOM 시 기존 shard를 보존합니다.
비정상 종료로 `.lock`이 남으면 기존 프로세스가 종료됐는지 확인한 후 **해당 output_dir의 `.lock`만** 제거하세요.

## 분석 정의

동일 pid는 seed 42의 GroupShuffleSplit으로 discovery 80% / confirmation 20%에 배치합니다.
discovery의 동일 pid+response 중복은 fitting에서 한 번만 사용합니다. 서로 다른 pid의 동일 response 수는 manifest에 보고합니다.
표시하는 전체 건수에는 원본 응답을 모두 유지하므로 반복 생성이 많은 문제에 대한 빈도 편향이 있을 수 있습니다.

각 응답은 prompt prefill 후 KV cache를 유지하며 128개 output token씩 재생합니다.
평균 집계 대상은 reasoning 영역의 비공백·비특수·비경계 token입니다. 마지막 #### 줄은 제외합니다.
개별 token에 SAE를 적용한 후 평균합니다. SAE(mean(hidden))로 대체하지 않습니다.
decoder는 복원 오차 계산에 사용하고 Qwen의 생성 경로를 대체하지 않습니다.
빈 풀이·잘림·안전하게 정렬되지 않는 응답은 제외 사유를 남깁니다. 수치/shape 오류는 전체 단계를 중단합니다.

SAE mean은 discovery에서 dead feature 제거 → log1p → PCA(중심화, 최대 64차원, whitening 없음) → L2 정규화합니다.
원시 hidden mean은 log1p 없이 같은 절차로 비교합니다. PCA는 discovery에서만 fit합니다.
K=2..10, seed=42/43/44, n_init=10을 비교합니다. silhouette 표본은 최대 2,000개이며 각 군집을 최소 하나 포함합니다.
seed별 silhouette 평균이 가장 높은 K가 주 후보이며 동점은 작은 K를 선택합니다. 최종 모델은 seed 42입니다.
모든 K와 seed ARI, discovery 문제 80%를 5회 재표집한 ARI를 남깁니다.
confirmation에서는 재학습/재선택 없이 변환·배정·silhouette만 계산합니다.
기하학적 점수에 대한 검증된 의미 판정 임계값을 가정하지 않습니다. 낮은 점수/안정성은 구조 미확인으로 해석하세요.

## 출력과 검토

- manifest.json / splits.json / runtime.json: 원본 해시·분할·환경과 코드 fingerprint.
- sae_provenance.json: SAE 실제 config, weight SHA256, base revision 미검증 기록.
- shards/: 32응답마다 sae_mean, hidden_mean, 특징별 최대 활성 output token 위치, 정렬 offset과 품질 지표.
- metrics.json: SAE 및 hidden의 모든 K 후보, 안정성, confirmation 점수, 군집별 표본·문제 수.
- models.joblib: discovery에 fit한 PCA/feature mask/KMeans. 신뢰할 수 있는 본 실행의 로컬 파일만 로드하세요.
- report/REPORT.md / pca.png: 중심·경계·무작위 discovery 사례와 confirmation 사례. 산점도는 첫 두 좌표의 표시용입니다.
- report/assignments.csv: 제외 응답도 포함한 전체 ID별 상태·군집.
- report/features.csv: discovery의 응답별 활성 여부 차이 상위 20개 특징, 평균 차이와 실제 최대 활성 token 주변 구간.
- report/review.csv: 사람이 카테고리 후보·혼합·미판정을 기록할 빈 양식. 기존 자동 라벨은 표시하지 않습니다.
- report/rule_cross_table.csv / cluster_summary.csv: 군집 선택 이후의 규칙 라벨·길이·확신도·생성 모드 비교.
- report/extraction_quality.csv: 복원 오차·입력 에너지 대비 제곱 오차·활성 특징 수·제외 사유.

review.csv에 category_candidate/notes/review_status를 수정했다면 report 재생성이 중단되어 덮어쓰기를 방지합니다.
검토 파일을 별도로 보존한 후 생성 양식만 다시 만드세요. 군집 이름은 cluster_N이며 외부 LLM API로 이름을 붙이지 않습니다.
규칙 라벨과 맞거나 silhouette가 높아도 의미 있는 오류 taxonomy·인과성·교정 효과가 검증된 것은 아닙니다.
오답만 분석하므로 정상/오류 탐지 성능도 추정할 수 없습니다.

## 실행하지 않은 테스트

아래 테스트는 합성 데이터와 임의 초기화한 작은 CPU Qwen을 사용하며 pretrained 모델이나 데이터를 받지 않습니다.

```bash
python -m unittest discover -s sae_error_clustering/tests -v
```

ID/해시/재개 오류, privileged label 격리, 문제 단위 분할, confirmation PCA 누출,
토큰 경계 제외, SAE bias·집계 계산, 전체/청크 causal forward 일치 등을 검사하도록 작성했습니다.
테스트 성공을 확인한 상태가 아닙니다. 서버에서 먼저 CPU 테스트, 이어서 32응답 GPU 점검을 수행하세요.
