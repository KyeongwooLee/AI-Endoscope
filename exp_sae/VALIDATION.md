# 구현 검증 기록

검증일: 2026-11-15. 구현 브랜치: `kyeongwoo/exp-SAE`.
기준 commit: `origin/kyeongwoo@9b21c925116bb5b8a6b2f389ca9c0379f09e78d2`.
기존 실험 파일은 수정하지 않았다. 이 문서는 실험 결과가 아니라 구현 검증 기록이다.

## 수행한 검증

- 신규 오프라인 CPU 테스트 **28개 통과**, 기존 `sae_error_clustering` 회귀 테스트 **10개 통과**.
- 마지막 chat-template 반환형/보고서 비율 변경 후 생성·보고서 통합 테스트 2개를 다시 실행해 통과했다.
- ID 149,460개, 안정적인 seed, 문제 단위 분할, 수치 채점/형식 실패/잘림 구분을 확인했다.
- mock 생성기에서 두 worker가 각각 1문제 × 20응답을 저장하고 재개 시 재생성하지 않음을 확인했다.
- installed Transformers 5.1.0의 `apply_chat_template` 기본 `return_dict=True`를 확인하고 생성에 `return_dict=False`를 명시했다.
- 임의 초기화 tiny Qwen의 전체 forward와 원래 token IDs의 cached chunk replay에서 block hidden 일치를 확인했다.
- standard/TopK/JumpReLU, input scale, native/BrainPatch tensor 방향·shape, mean/max/frequency/peak 집계를 확인했다.
- mock OOM 이후 임시 hidden 보존, 재개, 집계 완료 후 ID/shape/hash 검증을 거친 삭제를 확인했다.
- 손상된 생성 묶음, 변경된 codebook/Batch 입력, 잘못된 evidence offset을 감지했다.
- taxonomy/파일럿/확인 요청의 공통 $9 예약, 모호한 제출 timeout의 기존 작업 복구, 중복 제출 차단을 확인했다.
- 계정 모델 미가용·업로드 실패·일부 실패/누락 응답에서 비용 예약과 미판정 보존을 확인했다.
- 확인 문제의 20응답 전체를 유지한 예산 절단, 실제 usage 정산과 사람의 파일럿 검토 gate를 확인했다.
- 탐색 전처리/PCA가 확인 영역을 보지 않음, 문제 단위 bootstrap, 겹치는 다중 유형을 제외한 동일 문제 차이를 확인했다.
- 합성 label → checksum으로 검증된 특징 → 분석 → 한국어 Markdown/JSON/PNG 보고서 연결을 확인했다.
- 모든 Python 파일 compile, CLI help, Git ignore를 확인했다. 로컬 설정·키·캐시·hidden·가중치·응답은 제외하고 registry JSON은 포함한다.

CPU 검증 환경은 Python 3.12.7, torch 2.6.0+cu124, Transformers 5.1.0, NumPy 1.26.4,
scikit-learn 1.5.1이다. OpenAI SDK는 로컬에 설치하지 않고 mock 경계에서 검증했다.
Windows에서 fsync는 쓰기 가능한 descriptor로 호출해야 함을 확인해 durable JSON/NPZ 저장에 반영했다.

## 공개 SAE 조사

Hugging Face의 고정 revision 설정, safetensors 헤더, LFS SHA256/크기를 읽었다. 전체 가중치는 로컬에서 로드하지 않았다.
registry는 후보 48개를 보존한다. 설정을 확인한 파일럿 대상 44개는 9개 레이어에 걸쳐 있다.
3개 huypn16 변형은 bias 처리 충돌, elrashid 후보는 pinned source 접근 401로 제외했다.
09Catho의 backbone revision 일치와 다른 후보의 학습 revision 미기록을 구분한다.
공개 metadata 확인을 실제 SAE 호환성·의미 타당성 검증으로 부르지 않는다.

## 수행하지 않은 검증

실제 3090 × 2의 메모리/속도, 전체 149,460응답 생성, 전체 공개 checkpoint 로드,
실제 API 계정의 Luna/Batch 가용성·청구액·라벨 품질은 서버 파일럿에서 확인해야 한다.
120GiB에서 모든 라벨 표본의 44개 SAE 특징이 들어가는지는 중복 재사용률과 표본 수에 따라 달라진다.
파일럿으로 추정하고, 실행 중에는 15GiB 여유 검사에서 기존 결과를 보존하며 중단한다.
사람의 독립 라벨 신뢰도, held-out 외부 검증, 인과 개입·오류 교정 실험은 포함하지 않았다.
유료 호출·전체 GPU 실행·자동 staging·commit·push를 수행하지 않았다.
