# GSM8K 반복 생성·오류 유형별 SAE 분석

`AI-Endoscope` 저장소 루트에서 `python -m exp_sae --config exp_sae/config.local.json <단계>`로 실행한다.
기존 `sae_error_clustering`의 토큰 경계·엄격한 수치 채점·문제 단위 분할·원자적 저장·캐시 replay를 가져오며 기존 실험 파일은 수정하지 않는다.

기본 실험은 GSM8K **train 7,473문제 × 20회 = 149,460응답**이다. 모델과 tokenizer는 각각
`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 데이터는
`openai/gsm8k@740312add88f781978c0658806c59bc2815b9866`의 `main/train`으로 고정했다.
temperature=0.7, top-p=0.95, 최대 1,024토큰이며 `(seed, pid, repeat)`의 SHA256에서 seed를 얻는다.
GPU/라이브러리가 바뀌면 seed가 같아도 bitwise 동일 생성은 보장하지 않는다. 실행 환경·코드 해시를 기록하고 변경된 환경으로 같은 실행을 이어가지 않는다.

## 로컬 검증

```powershell
python -m unittest discover -s exp_sae/tests -v
python -m unittest discover -s sae_error_clustering/tests -v
python -m exp_sae --help
```

테스트는 합성 문자열, mock API, 임의 초기화 tiny Qwen만 사용한다. 실제 GSM8K·모델 가중치·공개 SAE를 내려받거나 유료 API를 호출하지 않는다.
이 구현 작업에서 실제 3090 파일럿, 전체 생성, 실제 라벨 비용·품질 검증, commit, push는 수행하지 않았다.

## 서버 준비: 데이터 디스크에 환경과 캐시 배치

Linux/RTX 3090 두 장을 전제로 한다. `/data`는 **120GiB 데이터 디스크의 실제 mount point**로 바꾼다.
`run_dir`와 `cache_dir`는 같은 데이터 디스크의 별도 디렉터리여야 한다. Linux에서는 root filesystem 사용을 거부한다.
모델 다운로드 전 5GiB 추가 공간을 점검하고, 출력·checkpoint·임시 hidden을 쓰기 전에 **15GiB 여유**를 점검한다.
디스크 공간은 이 프로세스가 점유한 동안 다른 프로세스가 사용하는 용량까지 예약하지 않으므로, 해당 디스크를 다른 대용량 작업과 공유하지 않는다.

```bash
# AI-Endoscope 저장소 루트에서
mkdir -p /data/exp_sae/cache/tmp
export TMPDIR=/data/exp_sae/cache/tmp
export PIP_CACHE_DIR=/data/exp_sae/cache/pip
export HF_HOME=/data/exp_sae/cache/hf
export HF_HUB_CACHE=/data/exp_sae/cache/hf/hub
export HF_DATASETS_CACHE=/data/exp_sae/cache/datasets
export TORCH_HOME=/data/exp_sae/cache/torch
export XDG_CACHE_HOME=/data/exp_sae/cache
python3 -m venv /data/exp_sae/venv
source /data/exp_sae/venv/bin/activate
python -m pip install -r exp_sae/requirements.txt
python -m pip check
cp exp_sae/config.json exp_sae/config.local.json
# config.local.json의 run_dir/cache_dir를 실제 데이터 디스크 경로로 수정
python -m unittest discover -s exp_sae/tests -v
python -m exp_sae --config exp_sae/config.local.json prepare
```

CUDA를 지원하는 PyTorch 환경인지 확인한다. CPU로 조용히 전환하지 않는다. 각 worker는 보이는 GPU 한 장의 `cuda:0`만 사용한다.
API 키는 서버의 `OPENAI_API_KEY` 환경변수로만 전달한다. 설정·키·출력·캐시는 Git에 추가하지 않는다.
서버 설정 파일은 준비 후 변경하지 않는다. sampling, 예산·단가, 분석 옵션, 경로를 바꾸려면 새 run directory를 사용한다.

## 1. GPU 파일럿: 2문제 × 20회

```bash
CUDA_VISIBLE_DEVICES=0 python -m exp_sae --config exp_sae/config.local.json generate --worker 0 --limit 1
CUDA_VISIBLE_DEVICES=1 python -m exp_sae --config exp_sae/config.local.json generate --worker 1 --limit 1
# pid=0,1의 40응답. 라벨 없이도 추출 품질 측정 가능
CUDA_VISIBLE_DEVICES=0 python -m exp_sae --config exp_sae/config.local.json extract --limit 2
python -m exp_sae --config exp_sae/config.local.json report
```

`responses/00000.meta.json`, `00001.meta.json`의 seconds/peak_cuda_bytes/bundle_bytes,
`report/extraction_quality.json`의 reconstruction/mean_l0/peak_cuda_bytes/seconds/disk_free_bytes를 확인한다.
`--sae <registry ID>`를 주면 한 checkpoint만 먼저 검사할 수 있다. 기본값은 적용 대상 44개를 순차 처리한다.
이 파일럿은 호환성·시간·공간·재구성이 동작하는지 보는 절차이며 의미 있는 오류 유형의 증거가 아니다.
실행 시간과 120GiB 내 전체 집계 특징 저장 가능 여부를 파일럿 실측으로 판단한 후 확대한다.

## 2. 전체 생성

```bash
CUDA_VISIBLE_DEVICES=0 python -m exp_sae --config exp_sae/config.local.json generate --worker 0 > /data/exp_sae/generate-0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m exp_sae --config exp_sae/config.local.json generate --worker 1 > /data/exp_sae/generate-1.log 2>&1 &
wait
python -m exp_sae --config exp_sae/config.local.json report
```

worker는 `pid % workers`로 문제를 나눈다. 문제별 20응답의 JSONL와 checksum metadata가 모두 저장된 경우에만 완료로 인정한다.
중복 원문·token IDs를 모두 보존한다. JSONL에는 원문, gold, prompt/output IDs, sampling, seed, 모델/tokenizer revision,
finish, parse_status, answer_correct, truncated, canonical_id가 들어간다. `#### <number>`가 마지막 비어 있지 않은 줄에 없는 응답은 형식 실패다.
형식 실패·잘림을 수치 오답과 혼합하지 않는다. `analysis_eligible`과 extraction 제외 이유를 확인한다.

## 3. taxonomy 탐색 → 사람 검토 → 고정

문제 단위로 seed 42의 탐색 20% / 확인 80% 분할을 한 번 고정한다. 같은 문제의 모든 반복은 같은 영역에 있다.
탐색 영역에서 무작위 20문제·400응답을 선택한다. 먼저 문제별 20응답에서 후보 정의·경계·관찰 예시를 만들고, 별도 Batch에서 통합한다.
한 거대한 400응답 요청 대신 문제별 요청을 사용해 short-context 단가 상한을 유지한다.

아래 **submit만 실제 유료 작업을 제출**한다. build는 파일 작성과 예산 예약만 한다.
`KEY`는 build 출력의 64자리 key로 교체한다. collect가 진행 중이라고 출력하면 나중에 같은 명령을 다시 실행한다.

```bash
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy build
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy submit --key KEY
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy collect --key KEY
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy synthesize
# 새 SYNTHESIS_KEY를 사용
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy submit --key SYNTHESIS_KEY
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy collect --key SYNTHESIS_KEY
```

`taxonomy/codebook.draft.json`을 사람이 검토한다. 각 유형의 definition/include/exclude, 실제 response_id/quote,
다중 라벨·최초 오류·미판정 규칙을 수정하고 `reviewed_by`, `review_notes`, `version`을 채운다.

```bash
python -m exp_sae --config exp_sae/config.local.json discover-taxonomy freeze --draft /data/exp_sae/run-v1/taxonomy/codebook.draft.json
```

비어 있는 검토자·경계·예시와 원문에 없는 예시는 거부한다. codebook version·SHA256과 탐색 응답 hash를 고정하고 후속 라벨에서 검증한다.
확인 영역의 결과를 본 뒤 codebook을 수정하려면 새 탐색/확인 실험을 설계해야 한다.

## 4. $9 장부 안에서 라벨링

기본 모델은 `gpt-5.6-luna`다. submit 직전에 `models.retrieve`로 계정 가용성을 확인하며 다른 모델로 대체하지 않는다.
Responses + Batch API, strict JSON schema, reasoning effort none을 사용한다. 실제 Batch 지원/권한은 서버 제출에서 최종 확인한다.
가격 근거는 [Luna 모델 설명](https://developers.openai.com/api/docs/models/gpt-5.6-luna)과
[Batch 문서](https://developers.openai.com/api/docs/guides/batch), [가격표](https://developers.openai.com/api/docs/pricing)다.
기본 Batch 단가는 입력 $0.10/M, 출력 $0.60/M(2026-11-15 확인)이다. **서버 실행 전 가격을 다시 확인하고 준비 전에 설정한다.**

전체 $9는 taxonomy 문제별 탐색·통합·탐색 라벨·파일럿·확인 라벨·제출 중 요청을 모두 포함한다.
요청마다 입력 UTF-8 byte 수+4096 framing 여유와 최대 출력 토큰 수로 상한을 예약한다.
272,000 입력 토큰의 보수적 상한을 넘는 요청, 예산을 넘는 계획, 중복 custom_id를 거부한다.
API usage에서 읽은 토큰 수를 설정 단가로 환산한다. cached-input 할인·청구서 조정은 자동 정산하지 않으므로 **환산값은 실제 청구액의 보수적 추정**이다.
알 수 없는 실패 비용은 예약액을 유지한다. 공급자 사용량이 예약액을 초과하면 장부를 잠그고 신규 제출을 차단한다.

먼저 탐색 20문제를 고정 codebook으로 라벨링한다. 이 표본만 분석 모델 학습에 쓴다.

```bash
python -m exp_sae --config exp_sae/config.local.json label build --scope discovery
python -m exp_sae --config exp_sae/config.local.json label submit --key DISCOVERY_LABEL_KEY
python -m exp_sae --config exp_sae/config.local.json label collect --key DISCOVERY_LABEL_KEY
python -m exp_sae --config exp_sae/config.local.json label build --scope pilot
python -m exp_sae --config exp_sae/config.local.json label submit --key PILOT_KEY
python -m exp_sae --config exp_sae/config.local.json label collect --key PILOT_KEY
```

파일럿은 확인 영역의 무작위 2문제·40응답이다. `labels/labels.jsonl`과 원문·근거 구절을 직접 확인하고 API usage와 실제 청구액을 확인한다.
같은 문제의 완전히 같은 **원문**만 라벨 요청을 재사용한다. 원래 20개 행은 남긴다. 문제를 넘어서 라벨을 공유하지 않는다.
정답 응답도 과정 오류를 검사한다. labels에는 다중 유형·최초 오류 step/character·type별 정확한 quote/span·미판정 사유를 기록한다.
잘못된 근거 offset, 미완료 응답, API 실패는 `unclassified`로 남기고 원래 API 결과를 보존한다.

`label review-pilot`를 한 번 실행하면 필요한 `pilot_sha256`을 오류 메시지로 알려준다. 다음 JSON을 별도 review 파일로 작성한다.

```json
{
  "approved": true,
  "reviewed_by": "실제 검토자 이름",
  "pilot_sha256": "출력된 hash",
  "evidence_quality_notes": "실제로 확인한 근거 정확도와 미판정 비율, 한계",
  "actual_cost_notes": "API usage, 청구액 및 단가 대조 결과"
}
```

```bash
python -m exp_sae --config exp_sae/config.local.json label review-pilot --review /data/exp_sae/pilot-review.json
# --problems는 pilot 이후 무작위 확인 문제 목록의 누적 prefix 길이. 생략하면 남은 예산 내 최대 prefix.
python -m exp_sae --config exp_sae/config.local.json label build --scope confirmation
python -m exp_sae --config exp_sae/config.local.json label submit --key CONFIRMATION_KEY
python -m exp_sae --config exp_sae/config.local.json label collect --key CONFIRMATION_KEY
```

확인은 무작위 문제 순서의 prefix를 선택하고 각 문제의 20응답 전체를 포함한다. 비용이 싼 문제를 골라 건너뛰지 않는다.
예산에 다음 문제 전체가 안 들어가면 그 지점에서 멈춘다. 미라벨 나머지는 원래 생성 JSONL에 보존된다.
파일럿도 확인 평가에 포함한다. 파일럿을 보고 codebook·분류기 등을 조정했다면 독립 확인 평가로 해석하면 안 된다.

## 5. SAE 추출

```bash
CUDA_VISIBLE_DEVICES=0 python -m exp_sae --config exp_sae/config.local.json extract
python -m exp_sae --config exp_sae/config.local.json analyze
python -m exp_sae --config exp_sae/config.local.json report
```

[sae_registry.json](sae_registry.json)은 5개 배포자의 **48개** 항목을 조사한 snapshot이다.
모델/revision·레이어/hook·구조·입력 scale/bias 처리·config SHA256·가중치 LFS SHA256/size·safetensors shape와 출처를 기록한다.
`metadata_verified` **44개**, 레이어 **4·10·12·14·17·18·19·21·24**가 기본 적용 대상이다.
동일 층의 Geaming BT(F)/BT(P)/FAST, 8×/16×, standard/JumpReLU는 모두 별도 결과다.

| 배포자 | 적용/제외 |
|---|---|
| HuggingAnalist | layer 17,18,19,21 standard 4개 |
| Geaming | layer 4,14,24의 36개 변형 |
| huypn16 | native Eleuther config의 layer 10,12,18 3개 적용. layer 8,14,16은 `apply_b_dec_to_input=false`와 Eleuther의 decoder-bias subtraction이 충돌해 제외 |
| 09Catho | layer 18 TopK/input_scale 1개. backbone revision 일치. 20k-token smoke SAE의 과소학습 한계를 유지 |
| elrashid | layer 14: pinned README/sae.py가 HTTP 401, 차원별 표준화 및 tensor layout 확인 불가로 제외 |

공개 설정·헤더 검증을 actual weights load/reconstruction 검증과 구분한다. 대부분 학습 backbone revision이 공개되지 않아 exact match는 미검증이다.
서버에서 SHA256을 확인한 checkpoint만 로드하며 remote Python은 실행하지 않는다. `.pt`는 `torch.load(weights_only=True)`만 사용한다.
공개 source의 현재 구현 근거는 registry에 링크하며, 과거 학습 코드의 정확한 revision이 없다는 한계는 남는다.

저장된 prompt를 prefill한 뒤 **원래 output token IDs**를 frozen 모델에 chunk replay한다.
각 생성 토큰을 **처리한 직후 block 출력**을 hook한다. 토큰을 생성하기 전 상태 또는 마지막 RMSNorm 출력과 다르다.
토큰을 재인코딩해 모델 입력으로 바꾸지 않는다. original IDs, decoded text와 character offsets를 검증한다.
SAE를 토큰별 적용한 뒤 엄격히 분리된 reasoning 영역의 mean/max/활성 빈도, peak token, raw hidden mean을 얻는다.
최종 답 숫자·형식 토큰·special·경계 겹침은 reasoning에서 제외한다. 잘림·형식 실패는 별도 추출 제외 사유다.

checkpoint 하나와 문제별 작은 묶음을 순차 처리한다. 실제 base-model replay도 checkpoint마다 수행하므로 44개 전체의 시간은 파일럿으로 추정해야 한다.
hidden은 해당 묶음에서만 임시 보존한다. 집계 NPZ/품질 JSONL의 **ID·shape·SHA256**, hidden과 집계의 hash 연결,
원래 hidden shape·finite 값을 다시 확인한 후에만 임시 hidden을 지운다.
모든 응답에는 품질 통계, 라벨 표본에는 mean/max/frequency/raw/peaks를 보존한다.
같은 문자열의 라벨을 재사용해도 원래 token IDs가 다를 수 있어 feature를 무조건 재사용하지 않는다.
checkpoint 처리가 끝나면 checksum을 확인한 해당 캐시 파일만 삭제한다. base model 캐시는 유지한다.
라벨 snapshot이 달라지면 별도 추출 하위 경로를 사용한다. 초기 무라벨 파일럿을 덮어쓰지 않는다.

## 6. 평가와 산출물

- 동일 문제에 두 유형이 모두 있는 경우 각 유형에만 속하는 응답끼리 평균 차이를 계산한다. 둘 다 속하는 응답은 해당 쌍에서 제외한다.
  탐색 문제에서 feature를 선택하고 확인 문제에서 문제별 차이·bootstrap 구간을 보고한다. 짝지어진 문제가 부족하면 평가 불가다.
- 탐색에서만 variance filtering, StandardScaler, PCA와 L2 logistic(C=1)을 fit한다.
  SAE mean/max/frequency를 연결한 입력과 raw hidden PCA를 비교하고 길이/정오답/길이+정오답 기준선을 별도로 둔다.
  한 checkpoint 내 다섯 기준선은 같은 라벨·토큰 유효 표본을 쓴다.
- 유형별 positive/negative 문제 수가 최소치보다 적으면 AUPRC/F1을 만들지 않는다. F1 threshold는 0.5 고정이다.
  확인 응답의 문제 단위 bootstrap으로 95% 구간과 유효 반복 수를 보고한다.
- 각 유형의 top feature는 탐색에서 선택하고, 확인 영역의 최대 활성 구절과 같은 SAE에서 유형 간 공통 feature를 제공한다.
  여러 유형·SAE 비교는 다중 비교 보정 없는 탐색적 결과다.

| 경로(run_dir 아래) | 내용 |
|---|---|
| `manifest.json`, `problems.jsonl`, `splits.json` | 설정·데이터 revision/hash·실행 환경·문제 분할 |
| `generation-worker-*.json`, `responses/*.jsonl`, `*.meta.json` | 모든 반복 응답·토큰·재생 설정·생성 파일럿 실측 |
| `taxonomy/` | 탐색 원문 400응답, draft, 검토 후 고정 codebook |
| `api/ledger.json`, `*.input.jsonl`, `*.output.jsonl`, `*.remote.json` | 통합 예산 장부·원래 요청·원래 결과·provider 상태 |
| `labels/labels.jsonl`, `labels.jsonl.meta.json` | 검증/미판정·다중 유형·first error·근거·원문 hash |
| `sae_registry.json` | 적용/제외 후보와 공개 근거 snapshot |
| `extraction/<SAE>/<label-hash>/` | 문제별 집계 NPZ·품질 JSONL·완료 marker, 임시 hidden |
| `analysis/<SAE>.json` | 유형별 지표·문제 bootstrap·동일 문제 차이·공통 feature·구절 |
| `report/REPORT.md`, `quality.json`, `extraction_quality.json`, `metrics.json`, `*.png` | 한국어 보고서·명시적 분모·그림·출처 hash |

보고서는 일부 생성/파일럿 상태에서도 작성된다. 전체 추출/라벨이 필요한 analyze는 미완료 상태를 거부한다.
분리도는 **생성 완료 후 AI 라벨의 관찰 가능한 오류 유형과 activation 사이의 연관성**이다.
사람의 독립 다중 라벨 신뢰도, 외부 gold 검증, 오류 원인의 인과적 기여·생성 전 예측·교정 효과는 별도 검증해야 한다.

## 중단·재개와 실패 복구

- 정상 완료된 문제 묶음은 ID/설정/hash를 확인하고 건너뛴다. `--limit` 파일럿 뒤 같은 설정으로 제한 없이 실행하면 이어간다.
- OOM·공간 부족은 예외로 중단한다. 완료 결과와 아직 검증되지 않은 hidden을 보존한다. chunk/config를 바꾸려면 새 run을 쓴다.
- 손상 파일·commit metadata가 없는 orphan은 자동 삭제하지 않는다. 오류가 지목한 문제 파일과 sidecar를 **별도 백업 경로로 이동**하고 원인을 확인한 뒤 그 문제를 재생성/재추출한다.
  같은 seed만으로 원문이 바뀔 수 있으므로 이미 라벨/특징이 있는 생성 원문은 함부로 재생성하지 않는다.
- 잠금 파일 `.lock`이 남았다면 기록된 PID의 프로세스가 종료됐는지 먼저 확인한 후 **해당 lock 파일만** 수동 제거한다.
- Batch 제출 전에 `submitting`을 기록한다. timeout 이후 같은 submit은 metadata key를 가진 기존 Batch를 전체 페이지에서 조회해 **정확히 하나일 때만** 복구한다.
  0개/복수라면 재제출하지 않고 예약액을 유지한다. provider dashboard에서 실제 접수 여부를 확인해야 한다. SDK 자동 재시도는 꺼져 있다.
- `expired/cancelled/failed`의 일부 성공은 원래 결과로 보존하고 나머지를 미판정으로 남긴다. 실패 요청을 자동 유료 재시도하지 않는다.
  명시적 복구 후에도 중복 제출 위험이 남는 상태에서 ledger를 임의로 지우지 않는다.
- `label materialize`는 수집한 모든 label Batch를 다시 검증해 행으로 만든다. build를 반복해도 기존 문제를 다시 제출하지 않는다.

## 검토 후 수동 Git 전달

현재 로컬 브랜치는 최신 `origin/kyeongwoo`의 `9b21c925116bb5b8a6b2f389ca9c0379f09e78d2`에서 시작한 `kyeongwoo/exp-SAE`다.
Git ref 이름 충돌을 피하려고 기존 로컬 `kyeongwoo`는 `kyeongwoo-before-exp-SAE`로 이름만 변경해 보존했다. 원격 브랜치는 변경하지 않았다.
변경은 `exp_sae/` 안에만 있다. 구현·문서·registry만 추가하고 실행 데이터/hidden/가중치/캐시/키는 제외한다.

```powershell
# AI-Endoscope 저장소 루트에서, 사용자가 검토한 뒤 직접 실행
git diff --stat
git diff
git status --short
# 아직 untracked이므로 처음에는 일반 git diff에 안 보인다. exp_sae 파일 자체도 검토한다.
git add exp_sae
git diff --cached
git commit -m "Add repeated GSM8K SAE experiment pipeline"
git push -u origin kyeongwoo/exp-SAE
```
