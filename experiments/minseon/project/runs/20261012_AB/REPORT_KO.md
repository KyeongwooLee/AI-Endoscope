# GSM8K A/B 실험 결과

완료: pilot 40, train 200, validation 80, 공식 test 100. C는 보류했습니다. 기존 원본 응답 320개를 재사용하고 공식 test 원본 100개만 새로 생성했습니다. 이전 결과는 보존했습니다.

- A: 외부 D3 AUROC 0.707, hidden 추가 D4 0.719. 차이 +0.012, paired bootstrap 95% CI [-0.017371961805555603, 0.04709964045287621].
- B: 외부 selector 정확도 64.0%, 내부 selector 61.0%, validation에서 선택한 고정 정책 Rebuild 63.0%.
- 내부 selector는 원래 오답 35개 중 1개를 고쳤고, 원래 정답 65개 중 5개를 망가뜨렸습니다.
- B 내부−외부 정확도 차이 -3.0%p, 95% CI [-8.0, 2.0].

LLM은 frozen Qwen2.5-1.5B-Instruct이며 작은 logistic classifier만 학습했습니다. A의 test 결과를 B 설정 선택에 사용하지 않았습니다. 네 action의 성공 label을 독립적으로 보존했고 Keep = 1 − A error를 전 사례에서 검증했습니다.

Pilot에서 token 경계 병합, 잘린 응답의 중간 숫자 채점, 결론 뒤 부수 숫자 추출 문제를 발견해 수정했습니다. 수정 후 pilot 원래 정답 29/40, oracle 32/40입니다. Pilot 수치는 효과 검증 결과가 아닙니다.

현재 GPU는 GTX 1080 Ti 11GB입니다. 새 추론 FP32와 이전 V100 FP16 원본의 차이, 작은 표본, 규칙 기반 숫자 추출, 제한적인 calculator 추출 범위를 한계로 기록했습니다. 기존 팀의 1,319 test 전체 실험과 직접 같은 설정은 아닙니다. 음성 결과와 넓은 불확실성도 그대로 보고합니다.

상세 보고서는 REPORT.md, 재실행 명령은 ../../experiments/ab/README.md, 수치와 예측은 test_results.json 및 test_predictions.json을 확인하세요.

정성 검토에서 공식 test 823번의 reference 오류(문제상 18점, reference 14점)를 확인했습니다. 주 결과는 공식 reference 채점을 유지했고 해당 1개를 제외한 민감도 분석을 별도로 기록했습니다. 이 분석에서도 hidden-state 이득이라는 결론은 나오지 않습니다.
