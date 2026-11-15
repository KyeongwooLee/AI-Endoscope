from collections import Counter
from pathlib import Path

from . import batch, extract, label, sae
from .common import digest, object_hash, prepared, read_json, responses, save


def run(cfg):
    problems, manifest = prepared(cfg)
    totals, generated_pids, response_groups = Counter(), set(), {}
    for row in responses(cfg, require_complete=False):
        totals['generated_responses'] += 1
        generated_pids.add(row['pid'])
        for key in ('answer_correct', 'truncated'):
            totals[key] += int(row[key])
        totals['format_failure'] += row['parse_status'] != 'ok'
        totals['within_problem_duplicates'] += row['canonical_id'] != row['id']
        response_groups.setdefault(object_hash(row['response']), set()).add(row['pid'])
    registry = sae.registry(cfg)
    out = Path(cfg['run_dir']) / 'report'
    out.mkdir(parents=True, exist_ok=True)
    annotations = label.load(cfg) if (Path(cfg['run_dir']) / 'labels/labels.jsonl').exists() else []
    counts = Counter(r['label']['status'] for r in annotations)
    quality = {'planned_problems': len(problems), 'planned_responses': manifest['planned_responses'],
        **{key: totals[key] for key in ('generated_responses', 'answer_correct', 'truncated', 'format_failure', 'within_problem_duplicates')},
        'generated_problems': len(generated_pids),
        'cross_problem_identical_response_groups': sum(len(pids) > 1 for pids in response_groups.values()),
        'labeled_responses': len(annotations), 'label_status': dict(counts),
        'label_partitions': dict(Counter(r['partition'] for r in annotations)),
        'type_positive_counts': dict(Counter(t for r in annotations for t in r['label']['types']))}
    quality['rates'] = {key: totals[key] / totals['generated_responses'] if totals['generated_responses'] else None
        for key in ('within_problem_duplicates', 'truncated', 'format_failure', 'answer_correct')}
    quality['rates']['unclassified'] = counts.get('unclassified',0) / len(annotations) if annotations else None
    rate_names = {'within_problem_duplicates':'중복', 'truncated':'잘림', 'format_failure':'형식 실패',
                  'answer_correct':'수치 정답', 'unclassified':'라벨 미판정'}
    rate_text = '; '.join(f'{rate_names[k]} {v:.2%}' for k,v in quality['rates'].items() if v is not None)
    _, book = batch.ledger(cfg)
    quality['api_accounting'] = {'charged_plus_reserved_usd': batch.charged(book) / 1e6,
        'observed_usage_usd': sum(b.get('observed_microusd', 0) for b in book['batches'].values()) / 1e6,
        'reserved_not_reconciled_batches': sum('actual_microusd' not in b for b in book['batches'].values()),
        'budget_usd': cfg['labeling']['budget_usd']}
    save(out / 'quality.json', quality)
    lines = ['# GSM8K 반복 생성·오류 유형별 SAE 분석', '',
        '이 결과는 생성 완료 후 hidden의 AI 라벨 기반 분리도다. 오류의 인과적 원인, 생성 전 예측 또는 교정 효과를 증명하지 않는다.', '',
        f'- 모델 revision: `{cfg["model_revision"]}` / 토크나이저: `{cfg["tokenizer_revision"]}`',
        f'- 데이터 revision: `{manifest["dataset_revision"]}`',
        f'- 생성: {totals["generated_responses"]:,}/{manifest["planned_responses"]:,}응답; {quality["generated_problems"]:,}/{len(problems):,}문제',
        f'- 정답 {quality["answer_correct"]:,}, 잘림 {quality["truncated"]:,}, 형식 실패 {quality["format_failure"]:,}, 문제 내 중복 {quality["within_problem_duplicates"]:,} / 생성 {totals["generated_responses"]:,}응답',
        f'- 라벨 {len(annotations):,}, 미판정 {counts.get("unclassified", 0):,} / 라벨 {len(annotations):,}응답',
        f'- 비율: {rate_text or "아직 산출할 응답 없음"}',
        f'- API usage 단가 환산 ${quality["api_accounting"]["observed_usage_usd"]:.6f}; 미정산 예약 포함 ${batch.charged(book)/1e6:.6f} / $9 (cached-input 할인·청구서 미대조)',
        '', '## SAE 적용 범위', '',
        '공개 설정·가중치 헤더 확인은 실제 activation 호환성 검증과 구분한다. 학습 backbone revision 미기록 후보는 그 제한을 유지한다.', '',
        '| 후보 | 레이어 | 상태 | 제외 이유 / 제한 |', '|---|---:|---|---|']
    for entry in registry['entries']:
        lines.append(f'| {entry["id"]} | {entry["layer"]} | {entry["status"]} | {entry.get("reason") or entry.get("revision_match", "")} |')
    extraction_index = Path(cfg['run_dir']) / 'extraction/index.json'
    if extraction_index.exists():
        from .analyze import summarize_quality
        measured = {}
        lines += ['', '## 추출 실측', '',
            '| SAE | 처리 응답 / 유효 응답 | 상대 제곱 재구성 오차 | 평균 L0 | GPU 최대 MiB | 이번 실행 초 | 남은 GiB | 전체 추출 완료 |',
            '|---|---|---:|---:|---:|---:|---:|---|']
        for entry in registry['entries']:
            if entry['id'] not in read_json(extraction_index):
                continue
            _, _, details, state = extract.load_entry(cfg, entry, require_complete=False, include_arrays=False)
            summary = summarize_quality(details)
            measured[entry['id']] = {'quality': summary, 'runtime': state}
            lines.append(f'| {entry["id"]} | {summary["responses"]} / {summary["eligible_responses"]} | {summary["relative_squared_error"]} | {summary["mean_l0"]} | {state["peak_cuda_bytes"]/2**20:.1f} | {state["seconds_this_invocation"]:.1f} | {state["disk_free_bytes"]/2**30:.2f} | {state["complete"]} |')
        save(out / 'extraction_quality.json', measured)
    index = Path(cfg['run_dir']) / 'analysis/index.json'
    metrics = []
    if index.exists():
        files = read_json(index)['files']
        lines += ['', '## 확인 영역 평가', '',
            '문제 단위 탐색 20% / 확인 80%; 전처리·PCA·분류기는 탐색 영역만 fit한다. F1 임계값 0.5 고정, 구간은 문제 단위 bootstrap이다.', '',
            '| SAE | 유형 | 기준선 | 양성 응답/확인 응답 | 양성 문제 | AUPRC [95% CI] | F1 [95% CI] |', '|---|---|---|---:|---:|---|---|']
        for name, expected in files.items():
            path = Path(cfg['run_dir']) / 'analysis' / (name + '.json')
            if digest(path) != expected:
                raise ValueError('Analysis result checksum mismatch')
            result = read_json(path)
            metrics.append(result)
            for record in result['types']:
                if record['status'] != 'ok':
                    lines.append(f'| {name} | {record["type"]} | 평가 불가: 희소 유형 | {record["confirmation"]["positive_responses"]}/{record["confirmation"]["responses"]} | {record["confirmation"]["positive_problems"]} | — | — |')
                for baseline, score in record.get('baselines', {}).items():
                    lines.append(f'| {name} | {record["type"]} | {baseline} | {record["confirmation"]["positive_responses"]}/{record["confirmation"]["responses"]} | {record["confirmation"]["positive_problems"]} | {score["auprc"]:.4f} {score["auprc_ci95"]} | {score["f1"]:.4f} {score["f1_ci95"]} |')
        save(out / 'metrics.json', metrics)
        plot(metrics, out)
    else:
        lines += ['', '## 평가 상태', '', '분석 결과가 아직 없다. 파일럿 또는 미완료 실행이며 성능 수치를 보고하지 않는다.']
    lines += ['', '## 해석 제한', '',
        '- 정오답은 수치 최종 답 채점이며 과정의 타당성과 별개다. 정답 응답도 AI가 과정 오류를 검사한다.',
        '- 독립적인 사람 라벨 신뢰도 검증은 별도 과제다. taxonomy와 파일럿 검토는 전체 라벨의 정확도를 보장하지 않는다.',
        '- 중복 응답은 보존하고 라벨 요청만 재사용한다. 반복 응답의 종속성은 문제 단위 분할·bootstrap으로 다룬다.',
        '- 잘림·형식 실패·미판정은 별도 분모로 남기며 강제 유형을 붙이지 않는다. 토큰 경계 실패는 추출 제외 사유로 남긴다.',
        '- 유형별 공통 feature·최대 활성 구절·동일 문제 내 유형 쌍의 차이와 0분산 등 기준선 제외 사유는 각 analysis JSON에 포함한다. 다중 비교 보정 없는 탐색적 기술 통계다.',
        '- 최대 활성 구절은 feature 의미나 오류 원인의 인과적 증거가 아니다. 같은 층의 여러 SAE는 별도 결과다.', '']
    temp = out / 'REPORT.md.tmp'
    temp.write_text('\n'.join(lines), encoding='utf-8')
    temp.replace(out / 'REPORT.md')
    save(out / 'provenance.json', {'manifest_sha256': digest(Path(cfg['run_dir']) / 'manifest.json'),
        'registry_sha256': object_hash(registry), 'labels_sha256': object_hash(annotations),
        'report_sha256': digest(out / 'REPORT.md')})
    print(out / 'REPORT.md')


def plot(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = [(r['entry']['id'], t['type'], b, s['auprc']) for r in results
        for t in r['types'] for b,s in t.get('baselines', {}).items()]
    if not data:
        return
    # One small figure per SAE keeps dozens of variants readable.
    for name in sorted({d[0] for d in data}):
        values = [d for d in data if d[0] == name]
        fig, ax = plt.subplots(figsize=(9, max(3, len(values) * .22)))
        ax.barh([f'{t} / {b}' for _,t,b,_ in values], [s for *_,s in values])
        ax.set(xlim=(0,1), xlabel='Confirmation AUPRC', title=name)
        fig.tight_layout()
        fig.savefig(out / (name + '.png'), dpi=140)
        plt.close(fig)
