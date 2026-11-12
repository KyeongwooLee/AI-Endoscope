import csv,json,pathlib,sys,collections,hashlib
base=pathlib.Path('/home/coder/yaicon-errorprobe/AI-Endoscope/experiments/minseon/error_description_pilot');p=base/'runs/pilot_v1';o=p/'ai_review_v1';t=p/'review_package/templates/reviewer_1';dest=o/'reviews/ai_codex_v1';dest.mkdir(parents=True,exist_ok=True)
def read(f):return list(csv.DictReader(open(f,encoding='utf-8-sig')))
def write(f,rs):
 with open(f,'w',encoding='utf-8-sig',newline='') as fp:
  w=csv.DictWriter(fp,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def dump(f,x):f.write_text(json.dumps(x,ensure_ascii=False,indent=2))
d=json.load(open(o/'independent_diagnoses.json'));ix={r['item_code']:i for i,r in enumerate(d)}
meta=dict(reviewer_id='ai_codex_v1',reviewer_type='ai_provisional',initials='AI',review_dates=['2026-11-11'],independent_of_other_reviewer=False,saw_method_key_or_scores=True,notes='Codex 인공지능 단독 탐색 검수. 정확한 모델 snapshot은 이 세션에서 확인 불가. 100개 원문을 먼저 진단하고 200개 설명 슬롯과 319개 쌍을 검토함. 이전 대화의 방법별 예시를 알고 있으며, 쌍 판정 저장 전 군집 및 방법 대응을 열었으므로 눈가림 평가가 아님. 사람 검수나 민선 본인 판정으로 사용하지 말 것. 보조 표면 유사성 3열은 사용자 요청에 따라 공란. 기존 독립 사람 평가 지표는 변경하지 않음.')
dump(dest/'reviewer.json',meta)
s1=read(t/'items_stage1.csv')
for r in s1:
 z=d[ix[r['item_code']]]
 for k in ['observable_error_present','observed_error_or_uncertainty','reference_problem']:r[k]=z[k]
 r['reviewer_notes']='AI 탐색 판정. 내부 원인은 추정하지 않으며 원문에서 관찰한 관계를 기술함.'
write(dest/'items_stage1.csv',s1)
exceptions={8:'실제 오류를 짚지만 P-20d+20d로 미래 보유량을 상쇄한 앞 단계도 문제여서 최초 오류 진단으로는 불완전하다.',17:'잘못된 첫 #### 표시는 있으나 뒤의 최종 서술은 52로 옳다. 풀이 전체를 수학적으로 틀렸다고 보면 답 추출 문제와 혼동한다.',19:'기하 관계가 잘못된 것은 맞지만 끝이 단순히 2피트 떨어졌다고 생각했다는 원인까지 원문에서 확인되지는 않는다.',40:'재고 수량을 가격으로 나눈 오류는 맞다. 그러나 앞서 60/15로 새 가격을 만든 근거 없는 관계가 먼저 있다.',53:'잘못된 첫 #### 14와 달리 마지막 자연어 답은 21로 옳다. 최종 답이 틀렸다는 표현은 전체 응답을 정확히 반영하지 못한다.',62:'추가 발가락을 빼는 오류는 맞다. 앞서 보통 앵무새의 발톱을 6이 아닌 3으로 계산한 실수도 있다.',75:'왕복 비용 누락은 명확하다. 다만 문제는 봉지 가격을 제시했을 뿐 낱개 판매를 명시적으로 금지하지 않았으므로 판매 단위를 단정할 때 주의가 필요하다.',76:'18개월+6년=8.5년은 틀리다. 그러나 18개월을 2.5년으로 변환했는지, 1.5+6의 덧셈을 틀렸는지는 보이지 않는다.',94:'합과 절반의 계산 오류는 실제다. 그러나 절반을 쓰는 문제에서는 남은 돈과 쓴 돈이 같으므로 남은 돈을 구했다는 것 자체는 수학적 오류가 아니다.'}
s2=read(t/'items_stage2.csv')
for r in s2:
 i=ix[r['item_code']];label=r['description_label']
 if r['description_text'].startswith('(no description:'):
  for k in ['correctness','specificity','unsupported_assertion','evidence_fidelity']:r[k]='not_applicable'
  r['reviewer_notes']='주석자 기권. 설명의 정확성을 채점하지 않음.';continue
 r.update(correctness='correct',specificity='specific',unsupported_assertion='no',evidence_fidelity='faithful',reviewer_notes='원문 진단과 일치: '+d[i]['observed_error_or_uncertainty'])
 if i in exceptions:r.update(correctness='partially_correct',reviewer_notes=exceptions[i])
 if i in (19,76):r['unsupported_assertion']='yes'
 if i==19:r['evidence_fidelity']='unfaithful'
 if i==75 and label=='A':r['unsupported_assertion']='yes'
 if i==95 and label=='A':r.update(correctness='partially_correct',reviewer_notes='누락과 나눗셈 오류는 맞지만 전체 G를 각 하위 부대의 인원처럼 사용하는 앞선 오류도 있다.')
 if i==91 and label=='A':r.update(correctness='partially_correct',unsupported_assertion='yes',reviewer_notes='반감 횟수가 부족한 것은 맞지만 어떤 시간 구간을 건너뛰었는지는 원문에서 확정하기 어렵다.')
write(dest/'items_stage2.csv',s2)
# Explicit case-by-case SAME decisions after reading original diagnoses and all pairs.
same={28:'기간 전체량을 다른 길이의 기간량으로 잘못 취급한다.',85:'여러 기간의 합계를 한 기간의 양으로 취급해 다시 배율을 적용한다.',95:'수치가 들어간 올바른 식을 제시하고 그 값과 다른 근거 없는 최종 수를 쓴다.',111:'요구한 몫과 나머지 몫을 뒤바꾼다.',113:'적힌 산술식을 잘못 계산한다.',152:'누적 기간의 양과 하루 또는 한 주의 양을 혼동한다.',209:'올바른 수치 식과 모순되는 최종 값을 계산 근거 없이 쓴다.',222:'한 번 또는 한 달의 기여량만 사용하고 지정된 반복 횟수의 곱을 누락한다.',251:'하위 집단별 계산에 전체 집단의 수를 넣는다.',261:'올바른 수치 식과 모순되는 최종 값을 계산 근거 없이 쓴다.',265:'올바른 수치 식과 모순되는 최종 값을 계산 근거 없이 쓴다.',279:'여러 날 또는 주의 총량과 하루 또는 한 주의 양을 혼동한다.',306:'기간에 따라 반복해야 할 변화 연산을 필요한 횟수보다 적게 적용한다.'}
pairs=read(t/'pairs.csv');pi=json.load(open(o/'pair_index.json'))
for r,z in zip(pairs,pi):
 assert r['pair_code']==z['pair_code']
 a,b=d[z['left']],d[z['right']]
 uncertain=any(v['observable_error_present']!='yes' for v in [a,b])
 judgment='uncertain' if uncertain else 'same' if z['index'] in same else 'different'
 reason='한쪽의 관찰 가능한 수학적 오류 진단이 불확실하거나 답 표기 문제여서 동일 오류를 확정하지 않는다.' if uncertain else same.get(z['index'],'구체적인 오류의 대상 또는 관계가 다르다. 넓은 오류 범주나 같은 연산만으로 동일 유형으로 세지 않는다.')
 r.update(judgment=judgment,same_topic='',same_operation='',same_style='',reviewer_notes=reason+' [풀이 1] '+a['observed_error_or_uncertainty']+' [풀이 2] '+b['observed_error_or_uncertainty'])
write(dest/'pairs.csv',pairs)
# Freeze review artifact hashes before matching to retrieval keys.
dump(o/'review_hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir()})
sys.path.insert(0,str(base));from edp.evaluate import verified_yield,paired_bootstrap
jl=lambda f:[json.loads(l) for l in open(f) if l.strip()]
key=jl(p/'review_key_private/pair_key.jsonl');retr=jl(p/'retrieval/top5.jsonl')
resolved={r['pair_code']:{'final':r['judgment']} for r in pairs};codeof={(r['query_id'],r['gallery_id']):r['pair_code'] for r in key}
yields=verified_yield(retr,codeof,resolved,5,['R','E','G','Q','Random'])
contr={}
for a,b in [('G','E'),('G','R'),('E','R')]:contr[a+'-'+b]=paired_bootstrap([r['conservative'] for r in yields[a]['per_query']],[r['conservative'] for r in yields[b]['per_query']],2000,20261108,.95)
ik=jl(p/'review_key_private/item_label_key.jsonl');condition={(r['item_code'],r['description_label']):r['condition'] for r in ik}
stats={}
for c in ['E','G']:
 rows=[r for r in s2 if condition[(r['item_code'],r['description_label'])]==c];stats[c]={k:dict(collections.Counter(r[k] for r in rows)) for k in ['correctness','specificity','unsupported_assertion','evidence_fidelity']}
result={'status':'AI_ONLY_EXPLORATORY_NOT_BLINDED','scope':'100 evaluation responses, 200 description slots including 12 abstentions, all 319 unique retrieval pairs. Development 30 not re-reviewed. No full 7691 corpus review.','pair_counts':dict(collections.Counter(r['judgment'] for r in pairs)),'stage1_counts':dict(collections.Counter(r['observable_error_present'] for r in s1)),'reference_problem_counts':dict(collections.Counter(r['reference_problem'] for r in s1)),'description_ratings':stats,'retrieval':yields,'paired_query_bootstrap':contr,'limitations':['One AI judge; prior examples and cluster/method exposure; no independent human ground truth.','Same-error granularity is subjective; clusters are not used as truth labels.','Query bootstrap does not capture judge error or common-gallery dependence.','Different eligible galleries for description methods. Uncertain upper bound is a scenario, not a confidence interval.']}
dump(o/'ai_evaluation.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['retrieval']},ensure_ascii=False,indent=2))
for m,r in yields.items():print(m,{k:v for k,v in r.items() if k!='per_query'},'same slots',sum(z['same'] for z in r['per_query']))
