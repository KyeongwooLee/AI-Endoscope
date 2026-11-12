"""Grounded error-description pilot: resumable CLI.

Stages (see REPRODUCE.md):
  audit, sample, export-packets --phase P, import-annotations --phase P, freeze,
  embed, retrieve, review-package, cluster-dev, bridge, report, all-auto,
  evaluate-reviewed --reviews DIR [--adjudication FILE]
"""
import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
PACKAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_DIR))

from edp import annotation, audit, cluster_bridge, evaluate, report, review, sampling  # noqa: E402
from edp.io import (ProtocolError, append_jsonl, guard_or_write_json, now_utc, object_sha256,  # noqa: E402
                    read_json, read_jsonl, sha256_file, write_json, write_jsonl, write_text)

PHASE_ROLES = {'dev': ('development',), 'eval': ('eval_query', 'eval_gallery')}


class Ctx:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.cfg = json.loads(self.config_path.read_text(encoding='utf-8'))
        base = self.config_path.parent
        self.run_dir = (base / self.cfg['run_dir']).resolve()
        self.data_dir = (base / self.cfg['data_dir']).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._rows = None

    def log(self, stage, message, **extra):
        append_jsonl(self.run_dir / 'worklog.jsonl', {'at': now_utc(), 'stage': stage, 'message': message, **extra})
        print(f'[{stage}] {message}')

    @property
    def rows(self):
        if self._rows is None:
            self._rows, self._records = audit.load_primary(self.cfg, self.data_dir)
        return self._rows

    @property
    def rows_by_id(self):
        return {r['id']: r for r in self.rows}

    def path(self, *parts):
        return self.run_dir.joinpath(*parts)

    def manifest(self):
        return read_jsonl(self.path('sample_manifest.jsonl'))

    def frozen(self):
        p = self.path('protocol_frozen.json')
        return read_json(p) if p.exists() else None

    def overlays(self, phase):
        out = {}
        for cond in self.cfg['annotation']['conditions']:
            p = self.path('annotation', 'validated', phase, f'{cond}.jsonl')
            out[cond] = {r['source_id']: r for r in read_jsonl(p)} if p.exists() else {}
        return out


def run_fingerprint(ctx):
    c = ctx.cfg
    return {'sources': {k: v.get('assets') for k, v in c['sources'].items() if isinstance(v, dict)},
            'split': c['split'], 'sampling': c['sampling'], 'protocol_version': c['protocol_version']}


def check_fingerprint(ctx):
    guard_or_write_json(ctx.path('fingerprint.json'), run_fingerprint(ctx), 'Run fingerprint')


def write_jsonl_guarded(path, rows, what):
    path = Path(path)
    text = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows)
    if path.exists():
        if path.read_text(encoding='utf-8') != text:
            raise ProtocolError(f'{what} exists with different content: {path}')
        return
    write_text(path, text)


def environment_record():
    import importlib.metadata as md
    versions = {}
    for p in ('numpy', 'scikit-learn', 'scipy', 'torch', 'transformers', 'sentence-transformers', 'huggingface-hub'):
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            versions[p] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {'name': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                   'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)}
    except ImportError:
        pass
    total, used, free = shutil.disk_usage(PACKAGE_DIR)
    mem = None
    try:
        mem = {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()[:3]}
    except OSError:
        pass
    return {'recorded_at': now_utc(), 'python': platform.python_version(), 'executable': sys.executable,
            'platform': platform.platform(), 'cpu_count': os.cpu_count(),
            'cpu_affinity': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'memory': mem, 'gpu': gpu,
            'disk_free_gb': round(free / 2 ** 30, 1), 'packages': versions}


# ---------------------------------------------------------------- stages
def stage_audit(ctx, args=None):
    assets = audit.fetch_and_verify(ctx.cfg, ctx.data_dir, allow_download=True)
    rows = ctx.rows
    counts = audit.audit_counts(rows, ctx._records)
    expected = ctx.cfg['expected_audit']
    mismatches = {k: {'expected': v, 'observed': counts.get(k)} for k, v in expected.items()
                  if k in counts and counts[k] != v}
    overlap, ledger = audit.prior_overlap(ctx.cfg, ctx.data_dir, rows, ctx._records)
    lab = overlap['error_taxonomy_full_v1_draft']
    for key, obs in (('prior_label_overlap_questions', lab['overlap_questions']),
                     ('prior_label_overlap_rows', lab['overlap_primary_rows']),
                     ('prior_label_exact_pairs', lab['exact_normalized_question_and_unchanged_response_pairs'])):
        if obs != expected[key]:
            mismatches[key] = {'expected': expected[key], 'observed': obs}
    write_json(ctx.path('asset_manifest.json'), {'assets': assets, 'verified_at': now_utc(),
                                                  'branches': audit.branch_heads()})
    write_json(ctx.path('audit.json'), {'counts': counts, 'expected': expected, 'mismatches': mismatches})
    write_json(ctx.path('overlap_audit.json'), overlap)
    write_json(ctx.path('prior_question_exclusions.json'), ledger)
    write_json(ctx.path('environment.json'), environment_record())
    if mismatches:
        ctx.log('audit', f'MISMATCH vs handoff audit: {sorted(mismatches)}; resolve before sampling')
        raise ProtocolError(f'Audit mismatches: {mismatches}')
    ctx.log('audit', f'{counts["responses"]} rows / {counts["questions"]} questions verified; '
                     f'{len(ledger)} prior-annotated question groups excluded')


def stage_sample(ctx, args=None):
    if not ctx.path('audit.json').exists() or read_json(ctx.path('audit.json'))['mismatches']:
        raise ProtocolError('Run a clean audit first')
    check_fingerprint(ctx)
    rows, records = ctx.rows, ctx._records
    parts = sampling.assign_sae_partitions([r['pid'] for r in rows], ctx.cfg['split']['random_state'],
                                           ctx.cfg['split']['test_size'])
    for rec, part in zip(records, parts):
        rec['sae_partition'] = part
    summary = sampling.split_summary(records)
    if summary != ctx.cfg['split']['expected']:
        raise ProtocolError(f'SAE split does not reproduce expected counts: {summary}')
    ledger = read_json(ctx.path('prior_question_exclusions.json'))
    units, exclusions, counts = sampling.build_eligibility(rows, records, ledger)
    elig = {u['source_id']: u for u in units}
    split_rows = []
    for rec in records:
        u = elig.get(rec['canonical_response_id'])
        split_rows.append({**rec, 'unit_eligible': bool(u and u['eligible']) and rec['canonical_response_id'] == rec['source_id'],
                           'unit_exclusion_reasons': u['exclusion_reasons'] if u else []})
    manifest, info = sampling.sample_pilot(units, ctx.cfg)
    write_jsonl_guarded(ctx.path('split_manifest.jsonl'), split_rows, 'split_manifest')
    write_jsonl_guarded(ctx.path('exclusions.jsonl'), exclusions, 'exclusions')
    write_jsonl_guarded(ctx.path('sample_manifest.jsonl'), manifest, 'sample_manifest')
    guard_or_write_json(ctx.path('sample_summary.json'), {'split': summary, 'eligibility': counts, **info,
                                                          'roles': {r: sum(m['pilot_role'] == r for m in manifest)
                                                                    for r in ('development', 'eval_query', 'eval_gallery')}},
                        'sample_summary')
    ctx.log('sample', f'split {summary}; eligibility {counts["questions_after_all_exclusions"]}; '
                      f'{len(manifest)} pilot rows')


def stage_export(ctx, args):
    phase = args.phase
    base = phase.split('_')[0]
    if base not in PHASE_ROLES:
        raise ProtocolError('phase must be dev, dev_rN or eval')
    prompts = annotation.prompt_info(ctx.cfg, PACKAGE_DIR)
    if base == 'eval':
        frozen = ctx.frozen()
        if frozen is None:
            raise ProtocolError('Freeze the protocol (stage freeze) before exporting evaluation packets')
        for cond, p in prompts.items():
            if p['sha256'] != frozen['protocol']['prompts'][cond]['sha256']:
                raise ProtocolError(f'Prompt {cond} differs from the frozen protocol')
    rows = [m for m in ctx.manifest() if m['pilot_role'] in PHASE_ROLES[base]]
    target = ctx.path('annotation', 'packets', phase)
    if target.exists():
        for cond, p in prompts.items():
            existing = read_jsonl(target / f'{cond}.jsonl')
            if any(x['prompt_sha256'] != p['sha256'] for x in existing):
                raise ProtocolError(f'{phase} packets exist with another prompt; use a new phase label (e.g. dev_r2)')
    counts = annotation.export_packets(ctx.run_dir, phase, rows, ctx.rows_by_id, prompts, ctx.cfg)
    ctx.log('export-packets', f'{phase}: {counts}')


def stage_import(ctx, args):
    frozen = ctx.frozen()
    phase = args.phase
    summary = annotation.import_annotations(ctx.run_dir, phase, ctx.cfg,
                                            frozen=frozen['protocol'] if (frozen and phase.startswith('eval')) else None)
    write_json(ctx.path('annotation', f'validation_report_{phase}.json'), summary)
    ctx.log('import-annotations', f'{phase}: ' + json.dumps({c: {k: v for k, v in s.items() if k != 'statuses'}
                                                              for c, s in summary.items()}))
    return summary


def protocol_payload(ctx):
    prompts = annotation.prompt_info(ctx.cfg, PACKAGE_DIR)
    c = ctx.cfg
    return {
        'prompts': {k: {'file': v['file'], 'sha256': v['sha256']} for k, v in prompts.items()},
        'annotation': {k: c['annotation'][k] for k in ('schema_version', 'language', 'word_range', 'max_attempts', 'statuses', 'annotator')},
        'embedding': c['embedding'], 'retrieval': c['retrieval'], 'review': c['review'],
        'bootstrap': c['bootstrap'], 'clustering': c['clustering'],
        'sample_manifest_sha256': sha256_file(ctx.path('sample_manifest.jsonl')),
        'split_manifest_sha256': sha256_file(ctx.path('split_manifest.jsonl')),
        'metrics': {
            'primary': 'mean over the 20 frozen queries of (#independently human-reviewed SAME neighbors among top-5)/5; '
                       'abstained/missing slots = 0; adjudicated uncertain = 0 (conservative), 1 in optimistic bound; '
                       'unreviewed = missing -> no score',
            'primary_contrast': 'G - E', 'secondary_contrast': 'G - R',
            'uncertainty': '2000 paired bootstrap draws over query questions; 95% percentile interval',
            'retrieval': 'cosine on L2-normalized BGE-M3 dense vectors, no PCA/clustering, ties by source_id; '
                         'abstaining E/G gallery entries excluded; abstaining query -> empty list'},
        'embedding_inputs': {'R': 'Question: {question}\\nSolution: {response}', 'Q': '{question}',
                             'E': 'explanation (supported_error only)', 'G': 'error_description (supported_error only)'},
        'reviewer_guide_sha256': sha256_file(PACKAGE_DIR / 'REVIEWER_GUIDE.md'),
        'prompt_history_sha256': sha256_file(PACKAGE_DIR / 'prompts' / 'PROMPT_HISTORY.md'),
    }


def stage_freeze(ctx, args=None):
    dev_phase = getattr(args, 'dev_phase', None) or 'dev'
    if not annotations_complete(ctx, dev_phase):
        raise ProtocolError(f'Development annotations ({dev_phase}) for E and G must be imported before freezing')
    payload = protocol_payload(ctx)
    payload['development_phase_used'] = dev_phase
    existing = ctx.frozen()
    if existing:
        if existing['protocol'] != payload:
            diff = [k for k in payload if existing['protocol'].get(k) != payload[k]]
            raise ProtocolError(f'Current settings differ from protocol_frozen.json in {diff}; use a new run_dir')
        ctx.log('freeze', 'protocol already frozen and unchanged')
        return
    write_json(ctx.path('protocol_frozen.json'), {'frozen_at': now_utc(), 'protocol': payload,
                                                  'protocol_sha256': object_sha256(payload)})
    ctx.log('freeze', f'protocol frozen sha256={object_sha256(payload)}')


def _phase_units(ctx, base):
    return [m for m in ctx.manifest() if m['pilot_role'] in PHASE_ROLES[base]]


def annotations_complete(ctx, phase):
    rep = ctx.path('annotation', f'validation_report_{phase}.json')
    if not rep.exists():
        return False
    return all(s.get('complete') for s in read_json(rep).values())


def stage_embed(ctx, args=None):
    from edp import embedding
    frozen = ctx.frozen()
    if frozen is None:
        raise ProtocolError('Freeze the protocol before embedding (revision/settings must be fixed)')
    if frozen['protocol']['embedding'] != ctx.cfg['embedding']:
        raise ProtocolError('Embedding settings differ from the frozen protocol')
    embedder = embedding.Embedder(ctx.cfg)
    results = {}
    dev_phase = getattr(args, 'dev_phase', None) or 'dev'
    for phase, base in ((dev_phase, 'dev'), ('eval', 'eval')):
        units = _phase_units(ctx, base)
        overlays = ctx.overlays(phase)
        for cond in ('R', 'Q', 'E', 'G'):
            if cond in ('E', 'G') and not annotations_complete(ctx, phase):
                results[f'{phase}/{cond}'] = 'pending_annotations'
                continue
            items = embedding.build_inputs(cond, units, ctx.rows_by_id, overlays.get(cond))
            results[f'{phase}/{cond}'] = embedding.embed_condition(ctx.run_dir, phase, cond, items, embedder, ctx.cfg)
    embedding.write_provenance(ctx.run_dir, embedder.provenance)
    write_json(ctx.path('embeddings', 'summary.json'), results)
    ctx.log('embed', json.dumps(results))


def stage_retrieve(ctx, args=None):
    from edp import embedding
    from edp.retrieval import retrieve
    units = _phase_units(ctx, 'eval')
    queries = [u for u in units if u['pilot_role'] == 'eval_query']
    gallery = [u for u in units if u['pilot_role'] == 'eval_gallery']
    rc = ctx.cfg['retrieval']
    results, status = [], {}
    for method in rc['methods']:
        if method == 'Random':
            vecs = None
        else:
            p = ctx.path('embeddings', 'eval', f'{method}.npz')
            if not p.exists():
                status[method] = 'pending_embeddings'
                continue
            ids, mat = embedding.load_vectors(ctx.run_dir, 'eval', method)
            vecs = dict(zip(ids, mat))
        res = retrieve(method, queries, gallery, vecs, rc['k'], rc['random_seed'])
        results.extend(res)
        status[method] = {'queries_ranked': sum(r['status'] == 'ranked' for r in res),
                          'queries': len(res), 'mean_usable_gallery': sum(r['usable_gallery_size'] for r in res) / len(res)}
    write_jsonl(ctx.path('retrieval', 'rankings.jsonl'), results)
    write_jsonl(ctx.path('retrieval', 'top5.jsonl'), [{k: v for k, v in r.items() if k != 'ranking'} for r in results])
    write_json(ctx.path('retrieval', 'summary.json'), status)
    ctx.log('retrieve', json.dumps(status))
    return status


def stage_review_package(ctx, args=None):
    summary_path = ctx.path('retrieval', 'summary.json')
    if not summary_path.exists() or any(isinstance(v, str) for v in read_json(summary_path).values()) \
            or set(read_json(summary_path)) != set(ctx.cfg['retrieval']['methods']):
        raise ProtocolError('All five retrieval methods must be complete before building the review package')
    retrievals = read_jsonl(ctx.path('retrieval', 'top5.jsonl'))
    units = _phase_units(ctx, 'eval')
    summary = review.build_review_package(ctx.run_dir, units, ctx.rows_by_id, ctx.overlays('eval'), retrievals, ctx.cfg)
    src = PACKAGE_DIR / 'REVIEWER_GUIDE.md'
    if src.exists():
        review.write_preserving(ctx.path('review_package', 'REVIEWER_GUIDE.md'), src.read_text(encoding='utf-8'), summary['files'])
    ctx.log('review-package', f'{summary["items"]} items, {summary["pairs_unique"]} unique pairs '
                              f'({summary["pairs_before_dedup"]} before dedup)')


def stage_cluster(ctx, args=None):
    from edp import embedding
    cc = ctx.cfg['clustering']
    phase = getattr(args, 'dev_phase', None) or 'dev'
    out = {}
    for rep in cc['representations']:
        p = ctx.path('embeddings', phase, f'{rep}.npz')
        if not p.exists():
            out[rep] = 'pending_embeddings'
            continue
        ids, mat = embedding.load_vectors(ctx.run_dir, phase, rep)
        out[rep] = {'n': len(ids), 'runs': cluster_bridge.kmeans_diagnostic(mat, ids, cc['k_values'], cc['seeds'], cc['n_init'])}
    write_json(ctx.path('clustering', 'dev_kmeans_diagnostic.json'),
               {'note': 'Development-only illustrative diagnostic; no K selected; not taxonomy validation.', **out})
    ctx.log('cluster-dev', 'wrote development K-means diagnostic')


def stage_bridge(ctx, args=None):
    manifest = ctx.manifest()
    overlays = {c: {} for c in ctx.cfg['annotation']['conditions']}
    dev_phase = getattr(args, 'dev_phase', None) or 'dev'
    for phase in (dev_phase, 'eval'):
        for c, d in ctx.overlays(phase).items():
            overlays[c].update(d)
    status_path = ctx.path('evaluation', 'item_review_status.json')
    rstat = read_json(status_path) if status_path.exists() else {}
    n = cluster_bridge.write_bridge(ctx.path('sae_bridge.jsonl'), manifest, overlays, rstat)
    split = {r['source_id']: r['sae_partition'] for r in read_jsonl(ctx.path('split_manifest.jsonl'))}
    write_json(ctx.path('sae_split_map.json'), split)
    sae = cluster_bridge.check_sae_run(ctx.cfg['sae'].get('results_dir'), split, manifest)
    write_json(ctx.path('sae_check.json'), sae)
    ctx.log('bridge', f'sae_bridge.jsonl rows={n}; SAE run: {sae["status"]}')


def stage_evaluate(ctx, args):
    key_dir = ctx.path('review_key_private')
    pair_key = read_jsonl(key_dir / 'pair_key.jsonl')
    item_key = read_jsonl(key_dir / 'item_label_key.jsonl')
    pair_codes = {p['pair_code'] for p in pair_key}
    item_codes = {i['item_code'] for i in item_key}
    reviews = review.load_reviews(Path(args.reviews), pair_codes, item_codes)
    adj = evaluate.load_adjudications(Path(args.adjudication) if args.adjudication else None, pair_codes)
    resolved = evaluate.resolve_pairs(reviews['human'], adj, sorted(pair_codes))
    code_of = {(p['query_id'], p['gallery_id']): p['pair_code'] for p in pair_key}
    retrievals = read_jsonl(ctx.path('retrieval', 'top5.jsonl'))
    yields = evaluate.verified_yield(retrievals, code_of, resolved, ctx.cfg['retrieval']['k'], ctx.cfg['retrieval']['methods'])
    states = {}
    for v in resolved.values():
        states[v['state']] = states.get(v['state'], 0) + 1
    complete = all(y['complete'] for y in yields.values()) and bool(reviews['human'])
    item_status = {}
    for i in item_key:
        rated = any((i['item_code'], i['description_label']) in r['items_stage2'] and
                    r['items_stage2'][(i['item_code'], i['description_label'])]['correctness'] for r in reviews['human'].values())
        item_status[i['source_id']] = 'human_reviewed' if rated else item_status.get(i['source_id'], 'pending')
    write_json(ctx.path('evaluation', 'item_review_status.json'), item_status)
    result = {
        'evaluated_at': now_utc(), 'reviews_dir': str(args.reviews),
        'human_reviewers': sorted(reviews['human']), 'ai_provisional_reviewers_not_counted': sorted(reviews['ai_provisional']),
        'pair_judgment_states': states, 'pairs_total': len(pair_codes),
        'semantic_evaluation_complete': complete,
        'status': 'semantic_evaluation_complete' if complete else 'human_review_pending',
        'verified_same_error_at_5': {m: {k: v for k, v in y.items() if k != 'per_query'} for m, y in yields.items()},
        'per_query': {m: y['per_query'] for m, y in yields.items()},
        'contrasts': evaluate.contrasts(yields, ctx.cfg) if complete else {'status': 'not_computed_missing_reviews'},
        'description_review': evaluate.description_metrics(reviews['human'], item_key),
        'problems': reviews['problems'],
        'interval_scope': 'conditional on this fixed gallery and review process; excludes gallery sampling, reviewer error, prompt selection',
    }
    write_json(ctx.path('evaluation', 'semantic_metrics.json'), result)
    ctx.log('evaluate-reviewed', f'status={result["status"]}; states={states}')
    stage_report(ctx, args)


def stage_test(ctx, args=None):
    import io
    import unittest
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE_DIR / 'tests'), top_level_dir=str(PACKAGE_DIR / 'tests'))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {'ran': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'skipped': len(result.skipped), 'recorded_at': now_utc(), 'python': platform.python_version(),
              'log_tail': stream.getvalue().splitlines()[-40:]}
    write_json(ctx.path('tests_result.json'), record)
    ctx.log('test', f'ran={record["ran"]} failures={record["failures"]} errors={record["errors"]}')
    if record['failures'] or record['errors']:
        raise ProtocolError('Unit tests failed; see tests_result.json')


def stage_report(ctx, args=None):
    status = report.compute_status(ctx)
    write_json(ctx.path('status.json'), status)
    report.write_results_ko(ctx, status)
    ctx.log('report', json.dumps({k: v for k, v in status['stages'].items()}))


def stage_all_auto(ctx, args):
    steps = []

    def attempt(name, fn, *a):
        try:
            fn(ctx, *a)
            steps.append({'stage': name, 'result': 'ok'})
            return True
        except ProtocolError as exc:
            steps.append({'stage': name, 'result': f'pending_or_blocked: {exc}'})
            ctx.log(name, f'not completed: {exc}')
            return False

    attempt('test', stage_test)
    if not attempt('audit', stage_audit) or not attempt('sample', stage_sample):
        write_json(ctx.path('all_auto_steps.json'), steps)
        return
    dev_phase = args.dev_phase or 'dev'
    ns = argparse.Namespace(phase=dev_phase, dev_phase=dev_phase)
    if not ctx.path('annotation', 'packets', dev_phase).exists():
        attempt('export-dev', stage_export, ns)
    attempt('import-dev', stage_import, ns)
    if ctx.frozen() or args.freeze:
        attempt('freeze', stage_freeze, ns)
    if ctx.frozen():
        ns_eval = argparse.Namespace(phase='eval', dev_phase=dev_phase)
        if not ctx.path('annotation', 'packets', 'eval').exists():
            attempt('export-eval', stage_export, ns_eval)
        attempt('import-eval', stage_import, ns_eval)
        attempt('embed', stage_embed, ns)
        attempt('retrieve', stage_retrieve)
        attempt('review-package', stage_review_package)
        attempt('cluster-dev', stage_cluster, ns)
    else:
        steps.append({'stage': 'freeze', 'result': 'awaiting_protocol_freeze (run stage freeze after dev review)'})
    attempt('bridge', stage_bridge, ns)
    attempt('report', stage_report)
    write_json(ctx.path('all_auto_steps.json'), steps)
    for s in steps:
        print(f'  {s["stage"]}: {s["result"]}')


STAGES = {'audit': stage_audit, 'sample': stage_sample, 'export-packets': stage_export,
          'import-annotations': stage_import, 'freeze': stage_freeze, 'embed': stage_embed,
          'retrieve': stage_retrieve, 'review-package': stage_review_package, 'cluster-dev': stage_cluster,
          'bridge': stage_bridge, 'report': stage_report, 'all-auto': stage_all_auto, 'test': stage_test,
          'evaluate-reviewed': stage_evaluate}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(PACKAGE_DIR / 'config.json'))
    parser.add_argument('--stage', required=True, choices=sorted(STAGES))
    parser.add_argument('--phase', default='dev')
    parser.add_argument('--dev-phase', default=None, help='development phase label whose annotations are used (default dev)')
    parser.add_argument('--reviews', help='directory of completed reviewer folders')
    parser.add_argument('--adjudication', help='optional adjudication CSV')
    parser.add_argument('--freeze', action='store_true', help='all-auto: freeze protocol if dev annotations are complete')
    args = parser.parse_args(argv)
    ctx = Ctx(args.config)
    if args.stage == 'evaluate-reviewed' and not args.reviews:
        parser.error('--reviews is required for evaluate-reviewed')
    try:
        STAGES[args.stage](ctx, args)
    except ProtocolError as exc:
        ctx.log(args.stage, f'ERROR: {exc}')
        sys.exit(2)


if __name__ == '__main__':
    main()
