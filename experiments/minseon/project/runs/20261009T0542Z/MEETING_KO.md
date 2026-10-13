# 다음 회의용 요약

**질문:** 전체/숫자 confidence 대비 hidden state와 block 변화량이 최종 오답 탐지에 추가 예측 가치를 주는가?

**실행:** Qwen2.5-1.5B-Instruct 동결, GSM8K pilot 50 + 본 실험 500개, train/validation/test=300/100/100. 실제 V100 GPU에서 greedy FP16 실행.

**실제 결과:** 전체 정답률(형식 추출 실패 포함) 58.8%. 공통 test N=79, 오류율=24.1%. B2 AUROC=0.810, H=0.668. H−B2=-0.141, paired bootstrap 95% CI [-0.302, +0.031].

B0=외형 통제, B1=전체 NLL 추가, B2=숫자 등 NLL 추가, H=P2 vector 추가, H+D=block 변화량 추가. P0=생성 직전, P1=최종 숫자 첫 token 직전, P2=최종 숫자 마지막 token 포함.

**공통 test 비교:**

|모델|AUROC|AP|오답 Recall@20%|
|---|---:|---:|---:|
|B0|0.552|0.401|0.316|
|B1|0.720|0.469|0.368|
|B2|0.810|0.577|0.421|
|H|0.668|0.470|0.421|
|H+D|0.685|0.461|0.368|
|P0|0.447|0.292|0.211|
|P1|0.832|0.611|0.526|
|P2|0.650|0.436|0.316|

H+D−H AUROC 차이: +0.017, 95% CI [-0.015, +0.052]. P0/P1/P2는 hidden-only이며 각 layer/C는 validation으로 선택했다.

**해석:** P2 hidden과 block 변화량의 추가 이득은 확인되지 않았다. P1은 후속 재현 검증 후보이지만, B2보다 낫다고 확정하거나 “생각하기 전 탐지”라고 부르지 않는다. Main 500개 중 형식 parsing 실패 127개는 전체 정확도에서 실패로 세고 공통 feature 비교에서는 제외했다.

Label-shuffle 20회 평균 AUROC는 B2 0.464, H 0.512였다. Bootstrap은 2,000회 모두 유효했다.

**한계:** 공통 test는 79개·오답 19개의 소규모 탐색이다. B2/H/H+D는 답변 완료 후 탐지다. P1도 reasoning 이후일 수 있고 위치는 사후 정의다. CI와 label-shuffle 결과를 함께 보고 일반화 또는 reasoning 오류 localization을 주장하지 않는다.

**다음:** 표본/seed 확대 재현성 확인 후 FinQA로 확장. 상세 비교와 실제 사례는 REPORT_KO.md, 그래프는 comparison.png.
