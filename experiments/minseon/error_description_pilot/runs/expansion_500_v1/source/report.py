from pathlib import Path
import json,csv,html,hashlib,sys,collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
out=Path(__file__).resolve().parents[1];base=out.parents[1];jl=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l];esc=html.escape
sample=jl(out/'sample.jsonl');byid={r['id']:r for r in sample};ann={c:{r['source_id']:r for r in jl(out/f'annotation/{c}.jsonl')} for c in 'EG'};summ=json.load(open(out/'annotation_summary.json'));cl=json.load(open(out/'cluster_details.json'));met=json.load(open(out/'cluster_metrics.json'));ev=json.load(open(out/'ai_evaluation.json'));common=json.load(open(out/'common_cohort.json'));names={'R':'문제와 풀이 전체','E':'일반 오류 설명','G':'근거 기반 오류 설명','Q':'문제만','Random':'무작위 선택'}
# Export source-aligned annotation tables; never embed metadata or quote fields.
for c in 'EG':
 rows=[]
 for s in sample:
  a=ann[c][s['id']];v=a['output'] or {};rows.append({'source_id':s['id'],'role':s['role'],'question':s['question'],'model_solution':s['response'],'reference_solution':s['reference'],'valid':a['valid'],'status':v.get('status'),'embedding_text':v.get('explanation' if c=='E' else 'error_description','') if a['valid'] and v.get('status')=='supported_error' else '', 'evidence_quote':v.get('evidence_quote',''),'validation_errors':str(a['errors'])})
 with (out/f'annotations_{c}.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for r in rows:w.writerow({k:"'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v for k,v in r.items()})
if not (out/'cluster_pca.png').exists():
 # PCA is only a display, not the clustering input.
 fig,axes=plt.subplots(2,2,figsize=(12,9))
 for ax,c in zip(axes.flat,'REGQ'):
  z=np.load(out/f'embeddings/{c}.npz');x=z['vectors'];ids=z['ids'].tolist();labels={s:g['cluster'] for g in cl[c] for s in g['members']};pca=PCA(n_components=2,random_state=42);xy=pca.fit_transform(x);ax.scatter(xy[:,0],xy[:,1],c=[labels[s] for s in ids],s=14,cmap='tab10',alpha=.65);ax.set_title(f'{c}: n={len(ids)}, PCA variance={pca.explained_variance_ratio_.sum():.1%}');ax.set_xlabel('Principal component 1');ax.set_ylabel('Principal component 2')
 fig.suptitle('Exploratory clusters: K-means k=10 on original 1024-D vectors\n2-D PCA is visualization only; colors are not verified error types');fig.tight_layout();fig.savefig(out/'cluster_pca.png',dpi=160);plt.close(fig)
body=['<h1>새 500문제: 오류 설명·임베딩·군집 분석</h1><p class="notice">2026-11-12 · AI 탐색 실험 · 독립 사람 검증 없음 · 기존 130개 실험과 주석 모델이 다름</p>', '<p><b>핵심 결과:</b> 새 500문제에서도 오류 유형별로 선명한 군집이 만들어졌다고 보기는 어렵습니다. 실제 묶음에는 비슷한 소재의 다른 오류가 함께 들어 있습니다. 근거 기반 방식의 인용·형식 실패와 잘못된 설명도 확인됐습니다. 아래 검색 수치는 AI 잠정 판정이며 독립 사람 평가가 아닙니다.</p><p>민수님의 기존 Qwen2.5-1.5B-Instruct 오답 응답에서 새 문제 500개를 골랐습니다. 기존130문제와 이전 검수 기록의 문제를 제외하고, 완료된 응답만 대상으로 문제당 한 응답을 선택했습니다. 후보 문제2,040개에서 고정 난수20261112로500개를 뽑았습니다. 기준100개와 후보400개는 서로 다른 문제입니다. 기존 SAE discovery/confirmation 구분을 합친 탐색 표본이며 새 SAE confirmation 결과가 아닙니다.</p>', '<h2>1. 어떤 모델에 무엇을 요청했나</h2><table><tr><th>역할</th><th>모델 및 실행</th></tr><tr><td>원래 수학 풀이 생성</td><td>Qwen/Qwen2.5-1.5B-Instruct, 민수님 저장 데이터 그대로. 새로 문제를 풀게 하지 않음.</td></tr><tr><td>오류 설명 생성</td><td>Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554. 로컬 V100, float16, greedy decoding, 최대1000출력토큰. E와G 및 각 사례는 독립된 입력 문맥. 이전claude-opus-5-5 기록의 실행 에이전트와 다른 모델.</td></tr><tr><td>임베딩</td><td>BAAI/bge-m3, revision5617a9f61b028005a4858fdac845db406aefb181. 기존과 같은1,024차원 밀집벡터, CLS pooling, float32, L2 정규화.</td></tr><tr><td>검색 결과 AI 판정</td><td>Qwen3-4B-Instruct-2507. 주석 모델과 동일하므로 독립 모델 검증이 아님. 처음에는 최대200출력토큰, JSON이 완성되지 않은 경우에만 같은 프롬프트로512·1024토큰 한도를 사용해 형식 복구. 별도 문맥에 원본 두 문제·풀이·참조만 제공. 방법명·점수·생성 설명·군집번호는 제공하지 않음.</td></tr></table>', '<p>기존 E/G 프롬프트를 사용했습니다. 입력은 질문·원본 풀이·참조 풀이이며 기존 오류 라벨을 넣지 않았습니다. 첫 출력이 형식·단어수·정확 인용 검증에 실패하면 검증 오류를 알려주고 한 번만 다시 요청했습니다. 재시도 후 실패한 출력은 임베딩에서 제외했습니다. 근거 기반 검증 실패198개 중77개는 인용문이 유일하게 정확히 존재하고 위치 숫자만 수정하면 검증을 통과할 수 있었습니다. 이는 사후 진단이며 주 분석의261개에는 추가하지 않았습니다. 인용문 일치는 설명이 수학적으로 옳다는 보증이 아닙니다.</p>', '<h2>2. 오류 설명 생성 결과</h2><table><tr><th>조건</th><th>형식 검증 통과</th><th>최종 상태별 개수</th></tr>']
for c in 'EG':body.append(f'<tr><td>{names[c]}</td><td>{summ[c]["valid"]}/500</td><td>{esc(json.dumps(summ[c]["statuses"],ensure_ascii=False))}</td></tr>')
body+=['</table><p>supported_error만 설명 임베딩에 사용합니다. no_error_observed는 모델이 오류를 확인하지 못한 경우, insufficient_evidence는 관찰 가능한 근거 부족, ambiguous는 해석 불확실, invalid는 출력 검증 실패입니다. 기권을 정답 판정으로 바꾸지 않았습니다. 오류 설명 E는 explanation, G는 error_description 문장만 임베딩했습니다. ID·인용문·위치·판정 상태를 벡터에 넣지 않았습니다.</p>',f'<p>두 설명 모두 사용 가능한 공통 집합은 {common["n"]}개: 기준 {common["query"]}개, 후보 {common["gallery"]}개입니다. 전체 표본 결과와 이 공통 집합 결과를 별도로 계산했습니다.</p>', '<h2>3. 군집 결과</h2><p>K-means 묶음 수4·6·8·10·15·20, 시작값42·43·44, 각10회 초기화를 실행했습니다. 평균 연결 계층 군집도 같은 묶음 수로 실행했습니다. 표시용10개 묶음/시작값42는 결과를 보기 전에 선택했습니다. 군집은 원래1,024차원에서 만들었습니다.</p>']
for cohort in ['native','common']:
 body.append(f'<h3>{"각 방법의 전체 사용 가능 사례" if cohort=="native" else "모든 방법에서 동일한 공통 사례"}</h3><table><tr><th>방법</th><th>개수</th><th>실루엣 분리도</th><th>시작값 변경 시 조정 랜드 지수 범위</th></tr>')
 for c in 'REGQ':
  z=next(z for z in met if z['cohort']==cohort and z['method']==c and z['algorithm']=='kmeans' and z['k']==10 and z['seed']==42);st=next(z for z in met if z['cohort']==cohort and z['method']==c and z['algorithm']=='stability' and z['k']==10)['ari'];body.append(f'<tr><td>{names[c]}</td><td>{z["n"]}</td><td>{z["silhouette"]:.4f}</td><td>{min(st):.3f}–{max(st):.3f}</td></tr>')
 body.append('</table>')
body+=['<p>실루엣은 같은 묶음 안 거리와 다른 묶음까지 거리를 비교합니다. 0근처면 경계가 약하고1에 가까울수록 잘 분리됩니다. 조정 랜드 지수는 시작값을 바꿔도 같은 구성원이 묶이는지 보여줍니다. 둘 다 “오류 유형 정답률”은 아닙니다. 높은 값이 나와도 실제로 오류가 같은지는 원문 확인이 필요합니다.</p><img src="cluster_pca.png" alt="네 표현별 PCA 시각화"><p>각 표현마다 PCA를 별도로 계산했으며, 같은 색이 방법 간 같은 오류를 뜻하지 않습니다. 그림에서 떨어져 보이는 정도는2차원 투영 결과입니다. 원래 공간의 분리나 같은 오류를 확정하는 증거로 쓰면 안 됩니다.</p>', '<h2>4. 같은 오류 검색: AI 잠정 평가</h2><p>각 기준 문제에서 후보5개를 찾고, 중복 쌍은 한 번 판정하여 모든 방법에 동일하게 적용했습니다. 아래 same 판정은 같은 모델 계열의 AI 판정이며 사람 정답이 아닙니다.</p>']
for cohort in ['native','common']:
 body.append(f'<h3>{"전체 기준100개, 기권도 분모에 포함" if cohort=="native" else "공통 기준·공통 후보만 비교"}</h3><table><tr><th>방법</th><th>실제 반환</th><th>same</th><th>uncertain</th><th>같은 오류 비율</th></tr>')
 for m,s in ev[cohort]['methods'].items():body.append(f'<tr><td>{names[m]}</td><td>{s["returned"]}</td><td>{s["same"]}</td><td>{s["uncertain"]}</td><td>{s["same_at_5"]:.1%}</td></tr>' if s['same_at_5'] is not None else f'<tr><td>{names[m]}</td><td colspan="4">미검수 때문에 계산하지 않음</td></tr>')
 body.append('</table>')
 for name,v in ev[cohort]['paired_query_bootstrap'].items():
  if 'difference' in v:body.append(f'<p>{name}: 차이 {v["difference"]*100:.1f}%포인트, 기준 문제 재표집95%구간 [{v["ci_low"]*100:.1f}, {v["ci_high"]*100:.1f}]%포인트.</p>')
body+=['<p>2,000회 기준 문제 단위 재표집은 AI 판정 오류나 공통 후보 의존성을 완전히 반영하지 않습니다. 주석 모델을 바꿨으므로 이전130개 실험보다 개선 또는 악화한 수치를 표본 확대의 효과라고 단정할 수 없습니다. 동일 모델이 설명 생성과 평가를 수행한 편향도 있습니다.</p>', '<h2>5. 설명 품질에서 발견한 실제 문제</h2><p>초기 E출력8개를 원문과 대조한 점검에서, 7갤런을56파인트로 바꾼 맞는 변환을 오류라고 덧붙인 설명과, 제한속도50mph라는 맞는 풀이를 틀렸다고 설명한 사례가 있었습니다. 이8개는 짧은 풀이부터 처리한 첫 출력이므로 전체 정확도를 추정하는 무작위 표본이 아닙니다. early_annotation_spotcheck.json에 판정 이유를 보존했습니다. 주석 정확성과 군집 분리도를 구분해서 읽어야 합니다.</p>', '<p><b>실제 타일 사례(train-3995-2):</b> 문제는 타일 한 장의 넓이를6제곱인치라고 명시합니다. 원래 Qwen 풀이는 전체넓이48×72=3456을6으로 나눠576장을 구해 문구에 맞습니다. 참조는6을 한 변 길이로 취급해96장을 제시합니다. 근거 기반 주석도 이 참조를 따라 원래 풀이를 오류라고 설명했고, 인용 형식 검증을 통과하여 실제 임베딩·군집에 포함됐습니다. 따라서 인용 검증 통과와 수학적으로 타당한 오류 설명은 다릅니다.</p><h2>6. 묶음별 실제 사례 탐색</h2><p>표현을 선택하고 각 사례를 펼치면 원본과 두 설명을 확인할 수 있습니다. 묶음 번호는 오류 유형 이름이 아닙니다. 임시 해석은 묶음마다 중심에 가까운 설명6개와 일부 원문 대조를 바탕으로 작성했으며, 모든 구성원의 수학적 오류를 검증한 결과는 아닙니다. 아래 모든 묶음은10개 설정의 결과입니다.</p><select id="method" onchange="filterGroups()">'+''.join(f'<option value="{m}" {"selected" if m=="G" else ""}>{names[m]}</option>' for m in 'REGQ')+'</select>']
interpret=json.load(open(out/'cluster_interpretation.json')) if (out/'cluster_interpretation.json').exists() else {}
for c,groups in cl.items():
 for g in groups:
  body.append(f'<section class="group" data-method="{c}"><h3>{names[c]} · 묶음 {g["cluster"]+1} · {g["n"]}개</h3>')
  if c=='G' and str(g['cluster']) in interpret:body.append('<p>'+esc(interpret[str(g['cluster'])])+'</p>')
  center={z['source_id'] for z in g['center_examples']};ordered=[z['source_id'] for z in g['center_examples']]+[s for s in g['members'] if s not in center]
  for sid in ordered:
   s=byid[sid];a=ann['G'][sid]['output'] or {};short=a.get('error_description') or s['question'];body.append(f'<details><summary>{sid}{" · 중심 근처 사례" if sid in center else ""} — {esc(short[:180])}</summary>')
   for title,col in [('문제','question'),('Qwen 원본 풀이','response'),('참조 풀이','reference')]:body.append(f'<h4>{title}</h4><pre>{esc(s[col])}</pre>')
   for cc in 'EG':body.append(f'<h4>{names[cc]} · valid={ann[cc][sid]["valid"]}</h4><pre>{esc(json.dumps(ann[cc][sid]["output"],ensure_ascii=False,indent=2))}</pre>')
   body.append('</details>')
  body.append('</section>')
if (out/'annotation_audit_G.json').exists():
 audits={cc:json.load(open(out/f'annotation_audit_{cc}.json')) for cc in 'EG'}
 labels={'correct':'설명이 타당','partially_correct':'부분적으로 타당','incorrect':'설명이 잘못됨','overconfident_ambiguous':'모호한 문제를 오류로 단정','appropriate_abstention':'적절한 오류 없음 판정','invalid':'검증 실패로 사용 불가'}
 body.append('<h2>추가 점검: 무작위20개 사례</h2><p>현재 Codex가 원문을 먼저 진단한 뒤 두 설명을 비교한 비눈가림 AI 점검입니다. 사람 검증이 아니며 작은 표본입니다. invalid는 내용이 옳다는 뜻도 틀렸다는 뜻도 아닙니다. 본문 평가 전에 사용 불가로 구분한 것이며 잘못된 내용이 함께 있을 수 있습니다.</p><table><tr><th>판정</th><th>일반 설명</th><th>근거 기반 설명</th></tr>')
 for key,label in labels.items():body.append(f'<tr><td>{label}</td><td>{sum(r["rating"]==key for r in audits["E"])}</td><td>{sum(r["rating"]==key for r in audits["G"])}</td></tr>')
 body.append('</table>')
 for i in range(20):
  body.append('<details><summary>'+audits['E'][i]['source_id']+'</summary>')
  for cc in 'EG':body.append('<p><b>'+names[cc]+'</b>: '+labels[audits[cc][i]['rating']]+' — '+esc(audits[cc][i]['reason_ko'])+'</p>')
  body.append('</details>')
body.append('<h2>자동 검색 평가를 해석할 때의 중요한 한계</h2><p class="notice">고유2,109쌍에서 같은 오류 판정은 극소수이며 불확실 판정은0개였습니다. 이 극단적인 분포와 아래 판정 이유 오류 때문에 검색 점수를 신뢰할 만한 실제 정확도로 사용할 수 없습니다. 낮은 수치만으로 임베딩 검색이 실패했다고 결론내릴 수 없습니다.</p><p>별도로 원문을 읽은10쌍의 AI 점검과 최종 라벨은8쌍에서 일치했지만, 대부분 다른 오류인 표본이므로 이 일치율은 평가 모델 정확도 검증이 아닙니다. 실제로 pair-0986에서는 질문이 작년 나이를 묻는데 평가 모델은 현재 나이를 묻는다고 설명했습니다. pair-2036에서는32×12를360으로 낸 계산 오류를 제대로 짚지 못하고384=384라는 모순된 설명을 제시했습니다. 라벨이 우연히 맞아도 이유는 틀릴 수 있습니다. pair_audit_comparison.json에 두 AI의 판정과 이유를 함께 보존했습니다.</p>')
body.append('<h2>7. 파일</h2><ul><li>sample.jsonl:500개 원본과 기준/후보 역할</li><li>annotations_E.csv / annotations_G.csv:원본, 설명, 검증 상태</li><li>annotation/raw_attempts.jsonl:모델 원출력과 모든 검증 오류</li><li>embeddings/:실제로 임베딩한 텍스트와npz벡터</li><li>cluster_assignments.csv:사례별 묶음 번호(파일0~9, 화면1~10)</li><li>cluster_metrics.json:모든 설정의 수치</li><li>ai_pair_reviews.csv:검색 쌍 판정과 이유</li><li>ai_evaluation.json:방법별·기준문제별 평가</li><li>source/:재현 스크립트</li></ul>')
(out/'analysis_report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>새500개 오류 임베딩 분석</title><style>body{max-width:1150px;margin:40px auto;padding:0 20px;font:17px/1.7 system-ui;color:#172334}h2{margin-top:48px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd4df;padding:10px;text-align:left}th{background:#eef3f8}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:14px}details{border-bottom:1px solid #ddd;padding:12px}summary{cursor:pointer}.notice{background:#fff2d4;padding:15px}select{font:inherit;padding:8px}img{width:100%}</style><body>'+''.join(body)+'<script>function filterGroups(){let m=document.getElementById("method").value;document.querySelectorAll(".group").forEach(e=>e.hidden=e.dataset.method!==m)}filterGroups()</script></body></html>')
old=base/'runs/pilot_v1';hashes=json.load(open(out/'pilot_preservation_hashes.json'));changed=[f for f,h in hashes.items() if not (old/f).exists() or hashlib.sha256((old/f).read_bytes()).hexdigest()!=h];assert not changed
assert len({r['question'] for r in sample})==500
for c in 'EG':assert len(ann[c])==500
status=json.load(open(out/'analysis_status.json'));status.update(judge_complete=False,report_complete=False);(out/'analysis_status.json').write_text(json.dumps(status,indent=2));(out/'validation.json').write_text(json.dumps({'original_files_preserved':len(hashes),'changed_original_files':changed,'unique_questions':500,'annotation_slots':1000,'complete_judgment_records':len(jl(out/'ai_pair_judgments.jsonl')),'human_review':False},indent=2))
import subprocess
subprocess.run([sys.executable,'-u',str(out/'source/validate_results.py')],check=True)
status.update(judge_complete=True,report_complete=True)
(out/'analysis_status.json').write_text(json.dumps(status,indent=2))
print('REPORT COMPLETE',flush=True)
