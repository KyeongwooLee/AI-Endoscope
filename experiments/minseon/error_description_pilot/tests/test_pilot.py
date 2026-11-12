"""Unit tests on synthetic fixtures (no real-data accuracy is implied).

Run: python -I -m unittest discover -s experiments/minseon/error_description_pilot/tests -v
"""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
os.environ.setdefault('USE_TF', '0')

from edp import annotation, audit, evaluate, retrieval, review, sampling  # noqa: E402
from edp.embedding import build_inputs, check_vectors  # noqa: E402
from edp.io import ProtocolError, question_sha256, read_jsonl, sha256_text, write_jsonl  # noqa: E402

CFG = json.loads((PKG / 'config.json').read_text(encoding='utf-8'))

# --- clearly labeled SYNTHETIC math fixtures (not real evaluation data) ---------------------
SYN = {
    'alt_valid': ('Tom has 3 bags of 4 apples and 2 loose apples. How many apples?',
                  '2 + 3*4 = 14\n#### 14', '3*4=<<3*4=12>>12\n12+2=<<12+2=14>>14\n#### 14'),
    'arith': ('Tom has 3 bags of 4 apples and 2 loose apples. How many apples?',
              '3*4 + 2 = 16\n#### 16', '3*4=<<3*4=12>>12\n12+2=<<12+2=14>>14\n#### 14'),
    'relation': ('Tom has 3 bags of 4 apples and 2 loose apples. How many apples?',
                 '(3+2)*4 = 20\n#### 20', '3*4=<<3*4=12>>12\n12+2=<<12+2=14>>14\n#### 14'),
    'unexplained': ('A shop sold 7 pens on Monday and twice as many on Tuesday. Total?', '#### 19', '7*2=14\n7+14=21\n#### 21'),
    'disputed': ('A café sells “half a dozen” muffins per box; Ana buys 2 boxes. Muffins?',
                 'Half a dozen is 6, so 2 boxes is 12 muffins… wait, maybe 2*6=12\n#### 12', '#### 6'),
}


def synthetic_row(name, pid=1, sample=0):
    q, r, ref = SYN[name]
    return {'id': f'train-{pid}-{sample}', 'split': 'train', 'pid': pid, 'sample': sample,
            'mode': 'greedy' if sample == 0 else 'sampled', 'question': q, 'reference': ref, 'gold': 14.0,
            'pred': 99.0, 'response': r, 'finish': 'stop', 'has_correct_sibling': False, 'rule_type': 'SETUP',
            'rule_first_wrong_eq': 0, 'rule_n_eq': 1}


def prompt_stub(cond):
    return {'file': f'prompts/{cond}_v1.txt', 'sha256': 'x' * 64,
            'text': (PKG / CFG['annotation']['conditions'][cond]).read_text(encoding='utf-8')}


DESC = ('The response groups the two loose apples with the bags and multiplies the sum of bags and loose apples '
        'by four, treating loose apples as extra bags instead of adding them individually after computing three bags of four.')


class JsonlAndHashes(unittest.TestCase):
    def test_lf_safe_unicode_jsonl(self):
        rows = [{'a': 'line sep para', 'b': 'é'}, {'a': 'x\r\ny'}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.jsonl'
            write_jsonl(p, rows)
            self.assertIn(' ', p.read_text(encoding='utf-8'))
            self.assertEqual(read_jsonl(p), rows)
            p.write_bytes(p.read_bytes() + b'\n\n')
            with self.assertRaises(ProtocolError):
                read_jsonl(p)

    def test_exact_response_hash_not_normalized(self):
        a, b = 'café', 'café'
        self.assertNotEqual(sha256_text(a), sha256_text(b))
        self.assertEqual(question_sha256(a + '  \n'), question_sha256('café'))


class AuditChecks(unittest.TestCase):
    def rows(self):
        return [synthetic_row('arith', 1, 0), synthetic_row('arith', 1, 1), synthetic_row('relation', 2, 0)]

    def test_shuffled_meta_and_duplicates(self):
        rows = self.rows()
        audit.validate_rows(rows, [r['id'] for r in rows])
        with self.assertRaises(ProtocolError):
            audit.validate_rows(rows, [r['id'] for r in reversed(rows)])
        dup = rows + [copy.deepcopy(rows[0])]
        with self.assertRaises(ProtocolError):
            audit.validate_rows(dup, [r['id'] for r in dup])

    def test_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = {'sources': {'p': {'release': 'r', 'subdir': 's', 'base_url': 'http://invalid/',
                                     'assets': {'f.txt': '0' * 64}}}}
            (Path(d) / 's').mkdir()
            (Path(d) / 's' / 'f.txt').write_text('hello')
            with self.assertRaises(ProtocolError):
                audit.fetch_and_verify(cfg, d, allow_download=False)

    def test_exact_duplicate_mapping(self):
        rows = self.rows()
        rows[1]['response'] = rows[0]['response']
        recs = audit.build_source_records(rows, 'r', 'h')
        self.assertEqual(recs[1]['canonical_response_id'], 'train-1-0')
        counts = audit.audit_counts(rows, recs)
        self.assertEqual(counts['extra_identical_pid_response_copies'], 1)
        for rec in recs:
            rec['sae_partition'] = 'discovery'
        units, excl, _ = sampling.build_eligibility(rows, recs, {})
        unit = next(u for u in units if u['source_id'] == 'train-1-0')
        self.assertEqual(unit['all_source_ids'], ['train-1-0', 'train-1-1'])
        self.assertTrue(any(e['reason'] == 'exact_duplicate_same_question_response' for e in excl))


class SplitAndSampling(unittest.TestCase):
    def test_split_matches_kyeongwoo_logic(self):
        from sklearn.model_selection import GroupShuffleSplit
        pids = [i // 3 for i in range(300)] + [5, 7, 9]
        ours = sampling.assign_sae_partitions(pids)
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(np.zeros(len(pids)), groups=pids))
        self.assertEqual({i for i, p in enumerate(ours) if p == 'confirmation'}, set(te.tolist()))
        for pid in set(pids):
            self.assertEqual(len({ours[i] for i, p in enumerate(pids) if p == pid}), 1)

    def _units(self, n_dev=60, n_conf=130):
        units, pid = [], 0
        for part, n in (('discovery', n_dev), ('confirmation', n_conf)):
            for _ in range(n):
                for s in range(2):
                    units.append({'source_release': 'r', 'source_asset_sha256': 'h', 'source_row_index': len(units),
                                  'source_id': f'train-{pid}-{s}', 'namespaced_id': f'r:train-{pid}-{s}', 'source_pid': pid,
                                  'question_key': f'gsm8k:train:{pid}', 'question_sha256': f'q{pid}',
                                  'response_sha256': f'r{pid}-{s}', 'canonical_response_id': f'train-{pid}-{s}',
                                  'all_source_ids': [f'train-{pid}-{s}'], 'source_mode': 'greedy', 'source_finish': 'stop',
                                  'sae_partition': part, 'response_chars': 100 + pid, 'rule_type_hidden': ['ARITH', 'SETUP'][pid % 2],
                                  'eligible': pid % 10 != 3, 'exclusion_reasons': []})
                pid += 1
        return units

    def test_no_overlap_and_exclusions(self):
        units = self._units()
        manifest, _ = sampling.sample_pilot(units, CFG)
        pids = [m['source_pid'] for m in manifest]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertFalse(any(p % 10 == 3 for p in pids))
        roles = {m['pilot_role'] for m in manifest}
        self.assertEqual(roles, {'development', 'eval_query', 'eval_gallery'})
        again, _ = sampling.sample_pilot(units, CFG)
        self.assertEqual(manifest, again)
        bad = copy.deepcopy(manifest)
        bad[-1]['source_pid'] = bad[0]['source_pid']
        with self.assertRaises(ProtocolError):
            sampling.check_sample(bad, CFG)


class PacketsAndValidation(unittest.TestCase):
    def packet(self, name='relation', cond='G'):
        return annotation.build_packet(synthetic_row(name), 'dev', cond, prompt_stub(cond), CFG)

    def test_packet_whitelist(self):
        p = self.packet()
        self.assertNotIn('rule_type', p['rendered_prompt'])
        self.assertNotIn('SETUP', p['rendered_prompt'])
        bad = dict(p, rule_type='SETUP')
        with self.assertRaises(ProtocolError):
            annotation.check_packet(bad)
        with self.assertRaises(ProtocolError):
            annotation.check_packet(dict(p, has_correct_sibling=True))

    def test_prompt_injection_safe_render(self):
        out = annotation.render_prompt('Q={question} R={response}', {'question': '{response}', 'response': 'X'})
        self.assertEqual(out, 'Q={response} R=X')

    def g_out(self, p, **kw):
        base = {'source_id': p['source_id'], 'response_sha256': p['response_sha256'], 'status': 'supported_error',
                'evidence_quote': '(3+2)*4', 'question_evidence_quote': '2 loose apples',
                'observed_error': 'adds loose apples to bag count', 'violated_condition': 'loose items are not bags',
                'error_description': DESC, 'reference_review_needed': False, 'reference_review_reason': None,
                'secondary_errors': [], 'uncertainty_note': None}
        base.update(kw)
        return base

    def test_valid_grounded_record_and_span(self):
        p = self.packet()
        errors, warnings, ov = annotation.validate_output('G', self.g_out(p), p, CFG)
        self.assertEqual(errors, [])
        self.assertEqual(p['response'][ov['evidence_start']:ov['evidence_end']], '(3+2)*4')

    def test_repeated_quote_requires_explicit_occurrence(self):
        row = synthetic_row('relation')
        row['response'] = 'Bags: 3*4 = 12. Then é 3*4 again = 12\n#### 24'
        p = annotation.build_packet(row, 'dev', 'G', prompt_stub('G'), CFG)
        out = self.g_out(p, evidence_quote='3*4')
        errors, _, _ = annotation.validate_output('G', out, p, CFG)
        self.assertTrue(any('occurs 2 times' in e for e in errors))
        errors, _, ov = annotation.validate_output('G', dict(out, evidence_occurrence=2), p, CFG)
        self.assertEqual(errors, [])
        self.assertEqual(ov['evidence_start'], row['response'].rindex('3*4'))
        start = row['response'].rindex('3*4')
        errors, _, ov = annotation.validate_output('G', dict(out, evidence_start=start, evidence_end=start + 3), p, CFG)
        self.assertEqual((errors, ov['evidence_span_method']), ([], 'annotator_offsets'))
        errors, _, _ = annotation.validate_output('G', dict(out, evidence_start=start, evidence_end=start + 4), p, CFG)
        self.assertTrue(errors)

    def test_non_ascii_span_and_no_normalization(self):
        p = self.packet('disputed')
        quote = 'muffins… wait'
        errors, _, ov = annotation.validate_output('G', self.g_out(p, evidence_quote=quote, question_evidence_quote='“half a dozen”'), p, CFG)
        self.assertEqual(errors, [])
        self.assertEqual(p['response'][ov['evidence_start']:ov['evidence_end']], quote)
        errors, _, _ = annotation.validate_output('G', self.g_out(p, evidence_quote='muffins... wait', question_evidence_quote=None), p, CFG)
        self.assertTrue(any('not an exact substring' in e for e in errors))

    def test_word_range_status_and_abstention(self):
        p = self.packet()
        errors, _, _ = annotation.validate_output('G', self.g_out(p, error_description='Too short.'), p, CFG)
        self.assertTrue(any('words' in e for e in errors))
        p2 = self.packet('unexplained')
        out = self.g_out(p2, status='insufficient_evidence', evidence_quote=None, error_description=None,
                         observed_error=None, violated_condition=None, question_evidence_quote=None)
        errors, _, ov = annotation.validate_output('G', out, p2, CFG)
        self.assertEqual(errors, [])
        errors, _, _ = annotation.validate_output('G', dict(out, status='maybe'), p2, CFG)
        self.assertTrue(errors)

    def test_e_condition(self):
        p = self.packet('arith', 'E')
        out = {'source_id': p['source_id'], 'response_sha256': p['response_sha256'], 'status': 'supported_error',
               'explanation': DESC, 'reference_review_needed': False}
        self.assertEqual(annotation.validate_output('E', out, p, CFG)[0], [])
        self.assertTrue(annotation.validate_output('E', dict(out, evidence_quote='x'), p, CFG)[0])

    def test_embedding_input_excludes_metadata(self):
        row = synthetic_row('relation')
        units = [{'source_id': row['id']}]
        rec = {'source_id': row['id'], 'condition': 'G', 'status': 'supported_error', 'error_description': DESC,
               'embedding_text': DESC, 'evidence_quote': '(3+2)*4', 'reference_review_needed': True}
        items = build_inputs('G', units, {row['id']: row}, {row['id']: rec})
        self.assertEqual(items[0]['text'], DESC)
        self.assertNotIn('14', items[0]['text'])
        rec_abs = dict(rec, status='insufficient_evidence', embedding_text=None)
        self.assertIsNone(build_inputs('G', units, {row['id']: row}, {row['id']: rec_abs})[0]['text'])
        r = build_inputs('R', units, {row['id']: row})[0]['text']
        self.assertEqual(r, f'Question: {row["question"]}\nSolution: {row["response"]}')
        self.assertNotIn('<<', r)
        with self.assertRaises(ProtocolError):
            build_inputs('G', units, {row['id']: row}, {row['id']: dict(rec, embedding_text='other')})

    def test_import_attempts_and_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            rows = [synthetic_row('relation', 1), synthetic_row('arith', 2)]
            prompts = {c: prompt_stub(c) for c in ('E', 'G')}
            annotation.export_packets(run, 'dev', [{'source_id': r['id']} for r in rows], {r['id']: r for r in rows}, prompts, CFG)
            pk = {p['source_id']: p for p in read_jsonl(run / 'annotation/packets/dev/G.jsonl')}
            bad = self.g_out(pk['train-1-0'], evidence_quote='(3+2) * 4')
            good = self.g_out(pk['train-1-0'])
            write_jsonl(run / 'annotation/inbox/dev/G/attempt1.jsonl', [bad])
            write_jsonl(run / 'annotation/inbox/dev/G/attempt2.jsonl', [good])
            s = annotation.import_annotations(run, 'dev', CFG)
            self.assertEqual(s['G']['accepted_on_attempt2'], 1)
            self.assertEqual(s['G']['missing'], 1)
            self.assertFalse(s['G']['complete'])
            log = read_jsonl(run / 'annotation/attempts.jsonl')
            self.assertEqual([e['valid'] for e in log], [False, True])
            annotation.import_annotations(run, 'dev', CFG)
            self.assertEqual(len(read_jsonl(run / 'annotation/attempts.jsonl')), 2)  # idempotent log
            write_jsonl(run / 'annotation/inbox/dev/G/attempt3.jsonl', [good])
            with self.assertRaises(ProtocolError):
                annotation.import_annotations(run, 'dev', CFG)
            os.remove(run / 'annotation/inbox/dev/G/attempt3.jsonl')
            frozen = {'prompts': {'E': {'sha256': 'x' * 64}, 'G': {'sha256': 'y' * 64}}}
            with self.assertRaises(ProtocolError):
                annotation.import_annotations(run, 'dev', CFG, frozen=frozen)


class VectorsAndRetrieval(unittest.TestCase):
    def unit(self, i, role):
        return {'source_id': f'train-{i}-0', 'question_sha256': f'q{i}', 'source_pid': i,
                'response_sha256': f'r{i}', 'pilot_role': role}

    def test_vector_checks(self):
        with self.assertRaises(ProtocolError):
            check_vectors(np.ones((2, 5)), 2)
        v = np.zeros((1, 1024)); v[0, 0] = np.nan
        with self.assertRaises(ProtocolError):
            check_vectors(v, 1)
        with self.assertRaises(ProtocolError):
            check_vectors(np.ones((1, 1024)), 1)

    def test_ranking_ties_abstention_and_roles(self):
        q = [self.unit(1, 'eval_query'), self.unit(2, 'eval_query')]
        g = [self.unit(i, 'eval_gallery') for i in (10, 11, 12, 13)]
        e = np.eye(4, 1024)
        vec = {'train-1-0': e[0], 'train-10-0': e[1], 'train-11-0': e[0], 'train-12-0': e[0]}
        res = retrieval.retrieve('G', q, g, vec, 2, 1)
        self.assertEqual([s['id'] for s in res[0]['top_k']], ['train-11-0', 'train-12-0'])  # tie -> source_id
        self.assertEqual(res[0]['usable_gallery_size'], 3)  # abstaining gallery item excluded
        self.assertEqual((res[1]['status'], res[1]['top_k']), ('query_abstained_or_missing', []))
        empty = retrieval.retrieve('G', q[:1], g, {'train-1-0': e[0]}, 2, 1)
        self.assertEqual(empty[0]['status'], 'empty_gallery')
        with self.assertRaises(ProtocolError):
            retrieval.retrieve('R', q, g + [dict(self.unit(1, 'eval_gallery'), source_id='train-1-3')], vec, 2, 1)
        r1 = retrieval.retrieve('Random', q, g, None, 4, 5)
        self.assertEqual(r1, retrieval.retrieve('Random', q, g, None, 4, 5))
        self.assertEqual(sorted(s['id'] for s in r1[0]['top_k']), sorted(u['source_id'] for u in g))


class ReviewAndEvaluation(unittest.TestCase):
    def setup_package(self, d):
        rows = {f'train-{i}-0': synthetic_row(['arith', 'relation', 'alt_valid'][i % 3], i) for i in range(1, 7)}
        rows['train-6-0']['response'] = '=SUM(A1)\n#### 3'
        units = [{'source_id': f'train-{i}-0', 'pilot_role': 'eval_query' if i <= 2 else 'eval_gallery'} for i in range(1, 7)]
        ov = {'E': {u['source_id']: {'status': 'supported_error', 'embedding_text': 'desc E ' + u['source_id']} for u in units},
              'G': {u['source_id']: {'status': 'insufficient_evidence', 'embedding_text': None} for u in units}}
        retr = []
        for m in ('R', 'E', 'G'):
            for qi, q in enumerate(('train-1-0', 'train-2-0')):
                retr.append({'method': m, 'query_index': qi, 'query_id': q, 'status': 'ranked',
                             'top_k': [{'id': 'train-3-0', 'rank': 1}, {'id': f'train-{4 + qi}-0', 'rank': 2}]})
        cfg = copy.deepcopy(CFG)
        cfg['retrieval']['k'] = 2
        cfg['retrieval']['methods'] = ['R', 'E', 'G']
        cfg['bootstrap']['contrasts'] = [['G', 'E']]
        summary = review.build_review_package(d, units, rows, ov, retr, cfg)
        return cfg, retr, summary

    def test_blinding_dedup_escape_and_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg, retr, s = self.setup_package(d)
            self.assertEqual((s['pairs_before_dedup'], s['pairs_unique']), (12, 4))
            pairs_csv = (d / 'review_package/templates/reviewer_1/pairs.csv').read_text(encoding='utf-8')
            for word in ('"R"', 'train-', 'method', 'rank', 'score'):
                self.assertNotIn(word, pairs_csv)
            items2 = (d / 'review_package/templates/reviewer_1/items_stage2.csv').read_text(encoding='utf-8')
            self.assertIn(review.ABSTAIN_TEXT, items2)
            self.assertNotIn('condition', items2)
            self.assertIn('"\'=SUM(A1)', pairs_csv + (d / 'review_package/templates/reviewer_1/items_stage1.csv').read_text())
            self.assertIn('&lt;&lt;', (d / 'review_package/pairs.html').read_text())
            key = read_jsonl(d / 'review_key_private/pair_key.jsonl')
            shared = next(k for k in key if k['gallery_id'] == 'train-3-0' and k['query_id'] == 'train-1-0')
            self.assertEqual([m['method'] for m in shared['returned_by']], ['E', 'G', 'R'])
            p = d / 'review_package/templates/reviewer_1/pairs.csv'
            p.write_text(pairs_csv.replace('"judgment"', '"judgment"', 1) + '\n', encoding='utf-8')
            s2 = self.setup_package(d)[2]
            self.assertIn({'file': str(p), 'action': 'preserved_existing_differs'}, s2['files'])
            self.assertTrue(p.read_text(encoding='utf-8').endswith('\n\n'))

    def write_reviews(self, d, judgments, rtype='human', name='reviewer_1'):
        sub = Path(d) / 'completed' / name
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'reviewer.json').write_text(json.dumps({'reviewer_id': name, 'reviewer_type': rtype,
                                                       'saw_method_key_or_scores': False}))
        rows = [{'pair_code': c, 'judgment': j} for c, j in judgments.items()]
        (sub / 'pairs.csv').write_text(review.csv_text(rows, review.PAIR_FIELDS), encoding='utf-8')

    def evaluate(self, d, cfg, retr):
        key = read_jsonl(Path(d) / 'review_key_private/pair_key.jsonl')
        codes = {k['pair_code'] for k in key}
        rv = review.load_reviews(Path(d) / 'completed', codes, set())
        resolved = evaluate.resolve_pairs(rv['human'], {}, sorted(codes))
        code_of = {(k['query_id'], k['gallery_id']): k['pair_code'] for k in key}
        return key, evaluate.verified_yield(retr, code_of, resolved, 2, ['R', 'E', 'G']), rv

    def test_unreviewed_vs_uncertain_and_shared_judgments(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, retr, _ = self.setup_package(Path(d))
            key = read_jsonl(Path(d) / 'review_key_private/pair_key.jsonl')
            by = {(k['query_id'], k['gallery_id']): k['pair_code'] for k in key}
            j = {by[('train-1-0', 'train-3-0')]: 'same', by[('train-1-0', 'train-4-0')]: 'uncertain',
                 by[('train-2-0', 'train-3-0')]: 'different'}
            self.write_reviews(d, j)
            _, y, _ = self.evaluate(d, cfg, retr)
            self.assertFalse(y['G']['complete'])
            self.assertIsNone(y['G']['verified_same_error_at_k'])
            self.assertEqual(y['G']['per_query'][0]['conservative'], 0.5)
            self.assertEqual(y['G']['per_query'][0]['optimistic'], 1.0)
            self.assertIsNone(y['G']['per_query'][1]['conservative'])
            j[by[('train-2-0', 'train-5-0')]] = 'same'
            self.write_reviews(d, j)
            _, y, _ = self.evaluate(d, cfg, retr)
            self.assertTrue(y['G']['complete'])
            self.assertAlmostEqual(y['G']['verified_same_error_at_k'], 0.5)
            self.assertEqual(y['R']['per_query'], y['G']['per_query'])  # shared judgments reused

    def test_ai_reviews_not_counted_and_disagreement(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, retr, _ = self.setup_package(Path(d))
            key = read_jsonl(Path(d) / 'review_key_private/pair_key.jsonl')
            all_same = {k['pair_code']: 'same' for k in key}
            self.write_reviews(d, all_same, rtype='ai_provisional', name='ai')
            _, y, rv = self.evaluate(d, cfg, retr)
            self.assertEqual(list(rv['ai_provisional']), ['ai'])
            self.assertIsNone(y['E']['verified_same_error_at_k'])
            self.write_reviews(d, all_same, name='h1')
            self.write_reviews(d, {k: 'different' for k in all_same}, name='h2')
            _, y, _ = self.evaluate(d, cfg, retr)
            self.assertFalse(y['E']['complete'])
            bad = Path(d) / 'completed' / 'h3'
            self.write_reviews(d, {'P-999999': 'same'}, name='h3')
            with self.assertRaises(ProtocolError):
                self.evaluate(d, cfg, retr)

    def test_synthetic_metric_and_bootstrap(self):
        a = [1.0, 0.0, 0.5, 0.5]
        b = [0.0, 0.0, 0.5, 0.0]
        out = evaluate.paired_bootstrap(a, b, 2000, 1, 0.95)
        self.assertAlmostEqual(out['difference'], 0.375)
        self.assertLessEqual(out['ci_low'], 0.375)
        self.assertGreaterEqual(out['ci_high'], 0.375)
        same = evaluate.paired_bootstrap(a, a, 100, 1, 0.95)
        self.assertEqual((same['ci_low'], same['ci_high']), (0.0, 0.0))
        idx = np.random.default_rng(1).integers(0, 4, size=(2000, 4))
        diffs = np.asarray(a)[idx].mean(1) - np.asarray(b)[idx].mean(1)
        self.assertAlmostEqual(out['ci_low'], float(np.quantile(diffs, 0.025)))


class ResumeFingerprints(unittest.TestCase):
    def test_guarded_writes_refuse_changes(self):
        sys.path.insert(0, str(PKG))
        import run as runner
        from edp.io import guard_or_write_json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'f.json'
            self.assertTrue(guard_or_write_json(p, {'a': 1}, 'x'))
            self.assertFalse(guard_or_write_json(p, {'a': 1}, 'x'))
            with self.assertRaises(ProtocolError):
                guard_or_write_json(p, {'a': 2}, 'x')
            q = Path(d) / 'm.jsonl'
            runner.write_jsonl_guarded(q, [{'a': 1}], 'm')
            runner.write_jsonl_guarded(q, [{'a': 1}], 'm')
            with self.assertRaises(ProtocolError):
                runner.write_jsonl_guarded(q, [{'a': 2}], 'm')


if __name__ == '__main__':
    unittest.main()
