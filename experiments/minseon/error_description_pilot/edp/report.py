"""Stage status and the Korean results report, generated only from actual run outputs."""
import collections
import itertools
import json
from pathlib import Path

from .io import read_json, read_jsonl, write_text


def _load(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return read_jsonl(p) if p.suffix == '.jsonl' else read_json(p)


def compute_status(ctx):
    p = ctx.path
    audit = _load(p('audit.json'))
    tests = _load(p('tests_result.json'))
    rep = {ph: _load(p('annotation', f'validation_report_{ph}.json')) for ph in ('dev', 'eval')}
    emb = _load(p('embeddings', 'summary.json'))
    ret = _load(p('retrieval', 'summary.json'))
    pkg = _load(p('review_package', 'package_summary.json'))
    sem = _load(p('evaluation', 'semantic_metrics.json'))
    frozen = _load(p('protocol_frozen.json'))
    ann_ok = all(r and all(s.get('complete') for s in r.values()) for r in rep.values())
    sem_ok = bool(sem and sem.get('semantic_evaluation_complete'))
    stages = {
        'implementation_complete': bool(tests and tests.get('failures') == 0 and tests.get('errors') == 0 and tests.get('ran', 0) > 0),
        'data_audit_complete': bool(audit and not audit['mismatches']),
        'protocol_frozen': bool(frozen),
        'annotations_complete': ann_ok,
        'embedding_complete': bool(emb) and not any(isinstance(v, str) for v in emb.values()),
        'retrieval_complete': bool(ret) and not any(isinstance(v, str) for v in ret.values()),
        'review_package_ready': bool(pkg),
        'human_review_pending': not sem_ok,
        'semantic_evaluation_complete': sem_ok,
    }
    annot = ctx.cfg['annotation']['annotator']
    return {
        'stages': stages,
        'primary_contrast_protocol_compliant': bool(annot['condition_context_isolated']),
        'condition_context_isolated': annot['condition_context_isolated'],
        'semantic_metrics': (sem or {}).get('verified_same_error_at_5') if sem_ok else None,
        'semantic_metrics_status': 'computed' if sem_ok else 'human_review_pending',
        'contrasts': (sem or {}).get('contrasts') if sem_ok else None,
        'tests': tests,
        'resume_command': ('python -I experiments/minseon/error_description_pilot/run.py --stage evaluate-reviewed '
                           '--reviews /absolute/path/to/completed_reviews [--adjudication /absolute/path/to/adjudication.csv]'),
    }


def _status_counts(ctx, phase, cond):
    rows = _load(ctx.path('annotation', 'validated', phase, f'{cond}.jsonl'), [])
    return collections.Counter(r['status'] for r in rows), rows


def _md_escape(text):
    return (text or '').replace('|', '\\|').replace('\n', ' ')


def _opening_stats(rows):
    texts = [r['embedding_text'] for r in rows if r.get('embedding_text')]
    first2 = collections.Counter(' '.join(t.split()[:2]) for t in texts)
    correctly = sum(' correctly ' in f' {t} ' for t in texts)
    return len(texts), first2.most_common(3), correctly


def _overlap(ctx):
    rows = _load(ctx.path('retrieval', 'top5.jsonl'), [])
    top = {(r['method'], r['query_id']): {s['id'] for s in r['top_k']} for r in rows}
    methods = ctx.cfg['retrieval']['methods']
    qs = sorted({q for _, q in top})
    out = []
    for a, b in itertools.combinations(methods, 2):
        js = [len(top[(a, q)] & top[(b, q)]) / len(top[(a, q)] | top[(b, q)]) for q in qs
              if top.get((a, q)) and top.get((b, q))]
        if js:
            out.append((a, b, sum(js) / len(js), len(js)))
    return out


def write_results_ko(ctx, status):
    p, cfg = ctx.path, ctx.cfg
    s = status['stages']
    audit = _load(p('audit.json'))
    ss = _load(p('sample_summary.json'))
    ov = _load(p('overlap_audit.json'))
    emb = _load(p('embeddings', 'summary.json'), {})
    res = _load(p('embeddings', 'resources_measured.json'), {})
    prov = _load(p('embeddings', 'provenance.json'), {})
    ret = _load(p('retrieval', 'summary.json'), {})
    pkg = _load(p('review_package', 'package_summary.json'), {})
    frozen = _load(p('protocol_frozen.json'), {})
    clus = _load(p('clustering', 'dev_kmeans_diagnostic.json'), {})
    env = _load(p('environment.json'), {})
    sem = _load(p('evaluation', 'semantic_metrics.json'))
    annot = cfg['annotation']['annotator']
    yes = lambda b: '완료' if b else '미완료'
    L = []
    w = L.append
    w('# 근거 기반 오류 설명 파일럿 결과 (pilot_v1)')
    w('')
    w('> 이 문서는 `run.py --stage report`가 실제 실행 산출물에서 자동 생성합니다. 사람 검수 결과가 들어오기 전까지 의미 평가 지표는 **null(human_review_pending)** 입니다.')
    w('')
    w('## 0. 한 줄 요약')
    w('')
    if s['semantic_evaluation_complete']:
        w('사람 검수가 완료되어 의미 지표가 계산되었습니다. 아래 5절 표를 보십시오.')
    else:
        w('자동 단계(감사·분할·표본·E/G 주석·검증·임베딩·검색·블라인드 검수 패키지·SAE 브리지)는 모두 실행되었고, '
          '**의미 평가(같은 오류 이웃 비율)는 독립 사람 검수가 없어 계산하지 않았습니다.** '
          f'다음 단계는 검수 패키지의 항목 100개와 쌍 {pkg.get("pairs_unique", "?")}개를 사람이 판정하는 것입니다.')
    w('')
    w('## 1. 단계 상태')
    w('')
    w('| 단계 | 상태 |')
    w('| --- | --- |')
    for k, v in s.items():
        w(f'| `{k}` | {"예" if v else "아니오"} |')
    w(f'| `primary_contrast_protocol_compliant` | {"예" if status["primary_contrast_protocol_compliant"] else "아니오 — E와 G를 같은 에이전트 세션이 작성(condition_context_isolated=false)"} |')
    w('')
    w('## 2. 무엇을 시험했나')
    w('')
    w('같은 주석자·같은 입력(문제, 원본 풀이, 틀릴 수 있는 참조 풀이)·같은 임베딩 모델에서, **근거 인용형 짧은 설명(G)**이 **일반 짧은 오류 설명(E)**보다 '
      '"같은 구체적 수학 오류"를 가진 이웃을 더 많이 찾는가(주 대비 G−E, 상위 5개 중 사람 검증 same 비율). 보조 대비는 G−R(문제+원본 풀이 텍스트). '
      'R은 참조 풀이를 쓰지 않으므로 G−R은 순수 임베딩 효과가 아니라 "참조 보조 주석 파이프라인"의 효용입니다. Q(문제만)와 Random은 통제 조건입니다.')
    w('')
    w('설명 자체의 타당성(관찰 가능한 실수를 맞게 짚었는가)과 표현의 타당성(가까움이 같은 오류를 뜻하는가)은 별도로 평가하도록 설계했습니다. '
      '이 파일럿은 오답 탐지 AUROC, 내부 인과 기제, 수정(repair) 효과를 측정하지 않습니다.')
    w('')
    w('## 3. 데이터·모델·환경')
    w('')
    c = audit['counts'] if audit else {}
    w(f'- 원본: `qwen-gsm8k-self-errors-v1` (`errors.jsonl` SHA256 `{cfg["sources"]["primary"]["assets"]["errors.jsonl"][:16]}…`), '
      f'Qwen2.5-1.5B-Instruct @ `{cfg["sources"]["primary"]["source_model_revision"][:12]}` 기존 오답. 새 답변 생성 없음.')
    w(f'- 감사: {c.get("responses")}응답 / {c.get("questions")}문제, sampled {c.get("sampled")} / greedy {c.get("greedy")}, '
      f'stop {c.get("finish_stop")} / length {c.get("finish_length")}, 동일 (pid,response) 추가 사본 {c.get("extra_identical_pid_response_copies")}, '
      f'문제 간 동일 응답 {c.get("identical_response_text_across_pids")}, 정규화 문제 중복 pid {c.get("normalized_question_groups_spanning_multiple_pids")}. '
      f'인수인계 문서 값과의 불일치: {len(audit["mismatches"]) if audit else "?"}건.')
    w(f'- 규칙 라벨 분포 {c.get("rule_counts")} — 주석자·검수자에게 전달하지 않음. `finish=length` {c.get("finish_length")}건은 모두 `####` 표식이 있었으나 잘림으로 제외.')
    w(f'- 주석 모델: `{annot["model_identifier"]}` ({annot["provider"]}, {annot["interface"]}), 경로 `{annot["route"]}`. '
      f'스냅샷 버전·디코딩 설정은 노출되지 않음(`exact_model_version_known=false`). API 호출·유료 엔드포인트 없음.')
    w(f'- 임베딩: `{cfg["embedding"]["model"]}` @ `{cfg["embedding"]["revision"]}` dense only, '
      f'pooling={json.dumps((prov.get("pooling") or {}).get("pooling_mode_cls_token"))} (CLS), L2 정규화, cosine, {cfg["embedding"]["precision"]}, '
      f'sentence-transformers {prov.get("versions", {}).get("sentence-transformers")}, torch {prov.get("versions", {}).get("torch")}. PCA·군집·재순위 없음.')
    w(f'- 환경: Python {env.get("python")}, 사용 가능 CPU {env.get("cpu_affinity")}개(호스트 {env.get("cpu_count")}), 메모리 {(env.get("memory") or {}).get("MemTotal")}, GPU {(env.get("gpu") or {}).get("name")}, 디스크 여유 {env.get("disk_free_gb")} GB. '
      f'임베딩 실측: {res.get("wall_seconds_including_model_load")}초(모델 로드 포함), 프로세스 RSS 피크 {res.get("peak_process_rss_gb")} GB, '
      f'GPU 피크 {res.get("peak_gpu_allocated_gb")} GB, 최대 입력 {res.get("max_token_count")}토큰(한도 8192, 잘림 없음).')
    w('')
    w('## 4. 공유 분할과 표본')
    w('')
    if ss:
        sp = ss['split']
        el = ss['eligibility']
        w(f'- Kyeongwoo의 `prepare.assign_splits`와 같은 `GroupShuffleSplit(test_size=0.2, random_state=42)`를 **필터링 전 7,691행 전체**에 적용: '
          f'discovery {sp["discovery_responses"]}응답/{sp["discovery_questions"]}문제, confirmation {sp["confirmation_responses"]}응답/{sp["confirmation_questions"]}문제 (문서 값과 일치). '
          f'파티션 간 정규화 문제 충돌 {el["cross_partition_question_conflicts"]}건.')
        lab = ov['error_taxonomy_full_v1_draft']
        w(f'- 기존 주석과의 중복: Sunghun `labels.jsonl` {lab["overlap_questions"]}문제/{lab["overlap_primary_rows"]}행, 정확 일치 쌍 {lab["exact_normalized_question_and_unchanged_response_pairs"]}건 (문서 값과 일치).')
        mr = ov['minsu_ai_review']
        w(f'- 확장 제외(문서 6절 지시): Minsu 브랜치 d95116f의 AI 검수 출력 `ab_labels.tsv` {mr["ab_labels.tsv"]["questions"]}문제, '
          f'`loc_labels.tsv` {mr["loc_labels.tsv"]["questions"]}문제(이 릴리스의 동일 ID), 로컬 Minseon 수동 판정/예시 사례 문제(질문 텍스트로 검증). '
          f'제외 문제 그룹 합계 {ov["excluded_question_groups_total"]}.')
        w(f'- 적격 문제: labels.jsonl+length 제외만 적용 시 discovery {el["questions_after_length_and_labels_jsonl_exclusion"]["discovery"]} / '
          f'confirmation {el["questions_after_length_and_labels_jsonl_exclusion"]["confirmation"]} (문서의 2,612/663 재현), '
          f'전체 제외 적용 후 {el["questions_after_all_exclusions"]["discovery"]} / {el["questions_after_all_exclusions"]["confirmation"]}.')
        enrich = ', '.join('{}={}'.format(e['stratum'], e['selected']) for e in ss['enrichment'])
        w(f'- 표본(seed {ss["seed"]}, {ss["rng"]}): 개발 {ss["roles"]["development"]}(균등 {cfg["sampling"]["dev_uniform"]} + 희귀 규칙/형식 보강 '
          f'{enrich}), 평가 쿼리 {ss["roles"]["eval_query"]} + 갤러리 {ss["roles"]["eval_gallery"]}, 모두 서로 다른 문제. '
          '보강 비율은 말뭉치 유병률이 아닙니다.')
        w(f'- 확인하지 못한 것: {"; ".join(ov["not_checked"])}')
    w('')
    w('## 5. 결과')
    w('')
    w('### 5.1 주석과 스키마 검증 (자동, 의미 타당성 아님)')
    w('')
    w('| 단계 | 조건 | 패킷 | 수락(1차/2차) | supported_error | insufficient_evidence | ambiguous | no_error_observed | 경고 |')
    w('| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    for ph in ('dev', 'eval'):
        rep = _load(p('annotation', f'validation_report_{ph}.json'), {})
        for cond in ('E', 'G'):
            r = rep.get(cond, {})
            st = r.get('statuses', {})
            w(f'| {ph} | {cond} | {r.get("packets")} | {r.get("accepted", 0) - r.get("accepted_on_attempt2", 0)}/{r.get("accepted_on_attempt2", 0)} | '
              f'{st.get("supported_error", 0)} | {st.get("insufficient_evidence", 0)} | {st.get("ambiguous", 0)} | {st.get("no_error_observed", 0)} | {r.get("with_warnings")} |')
    w('')
    gv = _load(p('annotation', 'validated', 'eval', 'G.jsonl'), [])
    sup = [r for r in gv if r['status'] == 'supported_error']
    w(f'- G 평가: supported_error {len(sup)}건 모두 `response[start:end] == evidence_quote` 통과(유효 span 비율 {len(sup)}/{len(sup)}). '
      f'반복되는 인용에서 주석자가 occurrence를 명시한 경우 {sum((r.get("evidence_span_method") or "").startswith("annotator_occurrence") for r in gv)}건, '
      f'유일 위치 인용 {sum(r.get("evidence_span_method") == "unique_occurrence" for r in gv)}건(기권 항목의 선택적 인용 포함). '
      f'참조/문항 검토 필요 플래그: E {sum(bool(r.get("reference_review_needed")) for r in _load(p("annotation", "validated", "eval", "E.jsonl"), []))}건, '
      f'G {sum(bool(r.get("reference_review_needed")) for r in gv)}건.')
    w('- 경고는 대부분 설명 속 피연산자가 우연히 정답 숫자와 같은 경우입니다(예: 문제의 "$28"과 정답 28). 스키마 통과는 수학적 타당성을 뜻하지 않습니다.')
    for cond, label in (('E', 'E'), ('G', 'G')):
        rows = _load(p('annotation', 'validated', 'eval', f'{cond}.jsonl'), [])
        n, top2, corr = _opening_stats(rows)
        w(f'- 문체 점검({label}, 평가 {n}건): 첫 두 단어 상위 {top2}; "correctly" 포함 {corr}건. '
          + ('G 프롬프트는 "일반적 서두"를 빼라고 했으나 거의 모든 G 설명이 같은 주어로 시작합니다. 이 공통 문체가 이웃 순위에 섞였을 수 있으며, 사람 검수 전에는 영향 크기를 알 수 없습니다(평가 데이터를 보고 프롬프트를 고치지 않았습니다).' if cond == 'G' else ''))
    ev_rows = _load(p('annotation', 'validated', 'eval', 'E.jsonl'), [])
    e_sol = sum((r.get('embedding_text') or '').startswith('The solution') for r in ev_rows)
    g_resp = sum((r.get('embedding_text') or '').startswith('The response') for r in gv)
    w(f'- **블라인딩 한계(설명 검수 Stage 2):** E 설명 {e_sol}건이 "The solution", G 설명 {g_resp}건이 "The response"로 시작해 '
      '무작위 A/B 라벨에도 조건을 추정할 수 있습니다. 설명 품질 검수는 완전 블라인드가 아님을 결과 해석에 반영해야 합니다. '
      '주 지표인 쌍 검수는 원본 문제·풀이·참조만 보여주므로 이 단서의 영향을 받지 않습니다.')
    w('')
    w('### 5.2 임베딩·검색 커버리지')
    w('')
    w('| 표현 | 평가 임베딩 | 미임베딩(기권) | 순위 매긴 쿼리 | 평균 사용 갤러리 |')
    w('| --- | ---: | ---: | ---: | ---: |')
    for m in cfg['retrieval']['methods']:
        e = emb.get(f'eval/{m}', {}) if m != 'Random' else {}
        r = ret.get(m, {})
        w(f'| {m} | {e.get("embedded", "—")} | {e.get("not_embedded", "—")} | {r.get("queries_ranked")}/{r.get("queries")} | {r.get("mean_usable_gallery")} |')
    w('')
    w('상위 5개 집합의 방법 간 평균 Jaccard(서술 통계, 의미 판정 아님):')
    w('')
    w('| A | B | Jaccard | 쿼리 수 |')
    w('| --- | --- | ---: | ---: |')
    for a, b, j, n in _overlap(ctx):
        w(f'| {a} | {b} | {j:.3f} | {n} |')
    w('')
    w('R과 Q의 이웃이 크게 겹치면 원문 임베딩이 주로 문제 내용(주제)에 끌린다는 신호일 수 있습니다. 다만 겹침은 "같은 오류"의 증거가 아닙니다.')
    w('')
    w('### 5.3 의미 평가 (주 지표)')
    w('')
    if sem and s['semantic_evaluation_complete']:
        w('| 방법 | verified_same_error@5 (보수) | 낙관 상한 | 쿼리 커버리지 |')
        w('| --- | ---: | ---: | ---: |')
        for m, v in sem['verified_same_error_at_5'].items():
            w(f'| {m} | {v["verified_same_error_at_k"]:.3f} | {v["optimistic_bound"]:.3f} | {v["query_coverage"]:.2f} |')
        for name, cdict in sem['contrasts'].items():
            if 'difference' in cdict:
                w(f'- {name}: {cdict["difference"]:+.3f}, 95% paired bootstrap [{cdict["ci_low"]:+.3f}, {cdict["ci_high"]:+.3f}] (고정 갤러리·검수 조건부)')
        if not status['primary_contrast_protocol_compliant']:
            w('- 주의: E/G가 격리된 문맥에서 생성되지 않았으므로 G−E는 프로토콜 준수 주 결과가 아니라 탐색적 결과입니다.')
    else:
        w('**null — `human_review_pending`.** 사람 검수 판정이 없으므로 verified_same_error@5, G−E, G−R, 부트스트랩 구간을 계산하지 않았습니다. '
          'AI 판정이나 silhouette로 대체하지 않았습니다. 미검수 쌍은 0점도 uncertain도 아닌 결측입니다.')
        w('')
        w(f'검수 대상: 항목 {pkg.get("items")}개(설명 {pkg.get("item_descriptions")}개), 쌍 {pkg.get("pairs_unique")}개 '
          f'(최대 {pkg.get("max_possible_pairs")} 슬롯 중 기권 제외 {pkg.get("pairs_before_dedup")}개를 중복 제거). '
          f'2차 검수자 사전 지정 부분집합: 항목 {pkg.get("second_review_items")}개, 쌍 {pkg.get("second_review_pairs")}개.')
    w('')
    w('### 5.4 개발 데이터 K-means 진단 (예시용, taxonomy 검증 아님)')
    w('')
    w('| 표현 | n | K=2..5 군집 크기 (seed 42) | silhouette (서술) | seed 간 ARI 범위 |')
    w('| --- | ---: | --- | --- | --- |')
    for rep in cfg['clustering']['representations']:
        d = clus.get(rep)
        if not isinstance(d, dict):
            continue
        runs = [r for r in d['runs'] if 'sizes' in r and r['seed'] == 42]
        aris = [x for r in d['runs'] if 'seed_ari' in r for x in r['seed_ari']]
        sizes = '; '.join(str(r['sizes']) for r in runs)
        sils = ', '.join('{:.3f}'.format(r['silhouette_descriptive']) for r in runs)
        w(f'| {rep} | {d["n"]} | {sizes} | {sils} | {min(aris):.2f}–{max(aris):.2f} |')
    w('')
    w('모든 K·seed를 보고했습니다(K=2–5, seed 42–44, n_init=10). 30건에서 silhouette가 0에 가깝고 seed 간 ARI가 불안정하면 뚜렷한 군집 구조가 확인되지 않은 것으로 해석합니다.')
    w('')
    w('## 6. 실제 출력 예시 (검수 전)')
    w('')
    w('아래는 실제 산출물이며 **사람 검수 전**입니다. "성공/불일치/근거 없는 설명" 판정은 검수 후에만 가능합니다.')
    w('')
    for phase, title in (('dev', '개발(discovery)'), ('eval', '평가(confirmation)')):
        ev = {r['source_id']: r for r in _load(p('annotation', 'validated', phase, 'E.jsonl'), [])}
        gvv = {r['source_id']: r for r in _load(p('annotation', 'validated', phase, 'G.jsonl'), [])}
        w(f'#### {title}')
        w('')
        sup_ids = sorted((k for k, r in gvv.items() if r['status'] == 'supported_error'), key=lambda x: int(x.split('-')[1]))
        for sid in sup_ids[:2]:
            w(f'- `{sid}` supported_error')
            w(f'  - E: {_md_escape(ev[sid].get("explanation"))}')
            w(f'  - G: {_md_escape(gvv[sid].get("error_description"))}')
            w(f'  - G 인용(`[{gvv[sid]["evidence_start"]},{gvv[sid]["evidence_end"]})`): `{_md_escape(gvv[sid]["evidence_quote"])}`')
        others = [k for k, r in sorted(gvv.items()) if r['status'] != 'supported_error']
        for sid in others:
            r = gvv[sid]
            w(f'- `{sid}` {r["status"]} (E: {ev[sid]["status"]}); reference_review_needed={r.get("reference_review_needed")}; '
              f'{_md_escape(r.get("reference_review_reason") or r.get("uncertainty_note") or r.get("observed_error"))}')
        w('')
    top = {(r['method'], r['query_index']): r for r in _load(p('retrieval', 'top5.jsonl'), [])}
    gtxt = {r['source_id']: r for r in gv}
    if top:
        q = top[('G', 0)]['query_id']
        w(f'#### 검색 예시: 쿼리 `{q}` (평가, 미검수)')
        w('')
        w(f'쿼리 G 설명: {_md_escape(gtxt[q].get("error_description"))}')
        w('')
        w('| 방법 | 1위 이웃 | cosine | 이웃의 G 설명(참고용) |')
        w('| --- | --- | ---: | --- |')
        for m in ('G', 'E', 'R', 'Q'):
            t = top[(m, 0)]['top_k']
            if t:
                nb = t[0]
                w(f'| {m} | `{nb["id"]}` | {nb["score"]:.3f} | {_md_escape((gtxt.get(nb["id"]) or {}).get("error_description") or "(기권)")} |')
        w('')
    w('## 7. 이 결과가 지지하는 것 / 모르는 것')
    w('')
    w('- 지지: 파이프라인이 공유 SAE 분할을 정확히 재현하고, 원본 해시로 ID를 맞추며, 잘림·중복·기존 주석 문제를 기록된 규칙으로 제외하고, '
      '두 조건의 설명을 같은 입력으로 만들고, 정확한 문자 span 검증을 통과한 근거를 남기고, 블라인드 검수 패키지를 만든다는 것.')
    w('- 지지: 평가 100건 중 주석자가 실수를 관찰 가능하다고 판단한 비율과 기권·참조 문제 비율(5.1). 이는 AI 판단이며 사람 검증 전입니다.')
    w('- 모름: G가 E·R보다 같은 오류 이웃을 더 많이 찾는지(주 질문). 설명이 실제로 맞는지. 이웃 근접이 주제·문체가 아닌 오류 패턴을 반영하는지.')
    w('- 제한: E/G 비격리 작성, 같은 모델이 두 조건 작성, 20개 쿼리·80개 갤러리의 작은 표본(작은 갤러리에서는 관련 이웃이 아예 없을 수 있음), '
      'GSM8K train의 이미 연구된 데이터, 참조 보조 오프라인 분석, G 설명의 공통 서두, '
      'E/G 서두 차이로 인한 Stage 2 설명 검수의 부분적 비블라인드.')
    w('')
    w('## 8. 다음 결정 하나')
    w('')
    if s['semantic_evaluation_complete']:
        w('의미 지표를 보고 문서 14절의 분기(설명 부정확 → 진단 개선 / 설명 정확·이웃 불량 → 표현 연구 / 둘 다 양호 → 동일 예산 repair 실험)를 선택합니다.')
    else:
        w('**사람 검수를 완료할지 결정하고 배정합니다.** 검수자 1명 이상이 `review_package/`를 지침대로 채운 뒤 다음을 실행합니다:')
        w('')
        w('```bash')
        w(status['resume_command'])
        w('```')
        w('')
        w('검수 없이 taxonomy나 G의 우위를 주장하지 않습니다. 원한다면 같은 패킷으로 E/G를 격리 문맥(별도 세션)에서 다시 생성해 프로토콜 준수 G−E를 만들 수 있습니다.')
    w('')
    w('## 9. 산출물 경로')
    w('')
    base = 'experiments/minseon/error_description_pilot'
    for rel, desc in [
        ('REPRODUCE.md', '재현 명령'), ('SOURCE_AUDIT.md', '원본·브랜치 감사'), ('REVIEWER_GUIDE.md', '검수 지침'),
        ('runs/pilot_v1/status.json', '단계 상태(기계 판독)'), ('runs/pilot_v1/protocol_frozen.json', '동결 프로토콜'),
        ('runs/pilot_v1/sample_manifest.jsonl', '표본 130건'), ('runs/pilot_v1/split_manifest.jsonl', '7,691행 분할'),
        ('runs/pilot_v1/exclusions.jsonl', '제외 원장'), ('runs/pilot_v1/overlap_audit.json', '중복 감사'),
        ('runs/pilot_v1/annotation/', '패킷·원본 출력·시도 로그·검증 오버레이'), ('runs/pilot_v1/embeddings/', '벡터·입력 텍스트·해시'),
        ('runs/pilot_v1/retrieval/', '전체 순위·top5'), ('runs/pilot_v1/review_package/', '블라인드 검수 파일'),
        ('runs/pilot_v1/review_key_private/', '방법 키(검수자 비공개)'), ('runs/pilot_v1/sae_bridge.jsonl', 'SAE 연결 130행'),
        ('runs/pilot_v1/clustering/dev_kmeans_diagnostic.json', '개발 군집 진단')]:
        link = rel[len('runs/pilot_v1/'):] if rel.startswith('runs/pilot_v1/') else '../../' + rel
        w(f'- [{base}/{rel}]({link}) — {desc}')
    w('')
    w(f'프로토콜 sha256: `{frozen.get("protocol_sha256", "—")}` (동결 시각 {frozen.get("frozen_at", "—")}). 실행 디렉터리는 git에 포함되지 않습니다(`{base}/.gitignore`).')
    write_text(ctx.path('RESULTS_KO.md'), '\n'.join(L) + '\n')
