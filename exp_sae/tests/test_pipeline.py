"""Offline CPU checks: synthetic text, mock paid API, randomly initialized tiny Qwen."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from exp_sae import analyze, batch, extract, generate, label, prepare, report, sae, taxonomy
from exp_sae.common import (atomic_npz, digest, load_config, object_hash, prepared,
    read_json, read_problem_responses, response_path, rid, rows, sample_seed, save, space, write_rows)
from sae_error_clustering.extract import hidden_chunks, reasoning_tokens
from sae_error_clustering.tokens import label_answer, token_regions


class Characters:
    all_special_ids = [0]
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return {'input_ids':[1,2]} if kwargs.get('return_dict',True) else [1,2]

    def decode(self, ids, **kwargs):
        return ''.join(chr(i) for i in ids if i != 0)

    def __call__(self, text, **kwargs):
        return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i,i+1) for i in range(len(text))]}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cfg = load_config(Path(__file__).parents[1] / 'config.json')
        self.cfg.update(run_dir=str(root / 'run'), cache_dir=str(root / 'cache'),
            expected_problems=20, discovery_problems=2, pilot_problems=2)
        self.cfg['analysis'].update(min_train_positive_problems=1, min_test_positive_problems=1,
            bootstrap_runs=20, pca_components=2, top_features=2)
        self.problems = prepare.make_problems([{'question': f'Problem {i}: 2+2?', 'answer': '2+2=4\n#### 4'}
            for i in range(20)], self.cfg)
        write_rows(root / 'run/problems.jsonl', self.problems)
        save(root / 'run/manifest.json', {'config': self.cfg, 'dataset_revision': self.cfg['dataset_revision'],
            'problems_sha256': digest(root / 'run/problems.jsonl'), 'planned_responses': 400})
        for problem in self.problems:
            records = []
            for repeat in range(20):
                text = '2+2=5\n#### 4' if repeat % 2 else '2+2=4\n#### 4'
                records.append({**problem, 'id': rid(problem['pid'], repeat), 'repeat': repeat,
                    'response': text, 'canonical_id': rid(problem['pid'], repeat % 2),
                    'prompt_token_ids': [1,2], 'output_token_ids': list(map(ord, text)) + [0],
                    'finish': 'stop', 'seed': sample_seed(problem['pid'], repeat),
                    'generation': self.cfg['generation'], **label_answer(text, problem['gold'], False)})
            target = response_path(self.cfg, problem['pid'])
            write_rows(target, records)
            save(target.with_suffix('.meta.json'), {'sha256': digest(target), 'config_sha256': object_hash(self.cfg)})
        source = taxonomy.sample(self.cfg)[0]
        self.codebook = {'version': 'test-v1', 'reviewed_by': 'test reviewer', 'review_notes': 'Synthetic review only',
            'categories': [{'id': 'arith', 'name': 'Arithmetic', 'definition': 'Incorrect arithmetic equality',
                'include': 'Explicit wrong equality', 'exclude': 'Correct equality', 'examples': [
                    {'response_id': rid(source['pid'], 1), 'quote': '2+2=5', 'explanation': '2+2 equals 4'}]}],
            'multi_label_rules': 'Use all observed types', 'first_error_rules': 'Earliest wrong equality',
            'unclassified_rules': 'Keep insufficient evidence'}
        self.draft = root / 'draft.json'
        save(self.draft, self.codebook)
        taxonomy.freeze(self.cfg, self.draft)
        self.space_patch = patch('exp_sae.batch.space', return_value=100 * 2**30)
        self.space_patch.start()
        self.addCleanup(self.space_patch.stop)

    def good_label(self, error=False):
        return {'status': 'classified', 'types': ['arith'] if error else [], 'process_error': error,
            'first_error_char': 0 if error else None, 'first_error_step': 1 if error else None,
            'evidence': [{'type': 'arith', 'start': 0, 'end': 5, 'quote': '2+2=5', 'reason': '2+2=4'}] if error else [],
            'unclassified_reason': ''}

    def complete(self, key, labels=True):
        _, book = batch.ledger(self.cfg)
        entry = book['batches'][key]
        requests = list(rows(Path(self.cfg['run_dir']) / 'api' / f'{key}.input.jsonl'))
        outputs = []
        for req in requests:
            payload = json.loads(req['body']['input'])
            result = self.good_label('2+2=5' in payload['response']) if labels else {'test': 'draft'}
            outputs.append({'custom_id': req['custom_id'], 'error': None,
                'response': {'status_code': 200, 'body': {'status': 'completed',
                    'usage': {'input_tokens': 100, 'output_tokens': 100},
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(result)}]}]}}})
        outputs.reverse()
        api = MagicMock()
        api.files.create.return_value.id = 'file-test'
        remote = SimpleNamespace(id='batch-test', input_file_id='file-test', metadata={'exp_sae_key': key},
            status='completed', output_file_id='output-test', error_file_id=None,
            model_dump=lambda **kwargs: {'status': 'completed'})
        api.batches.create.return_value = remote
        api.batches.retrieve.return_value = remote
        api.files.content.return_value.text = '\n'.join(json.dumps(r) for r in outputs)
        batch.submit(self.cfg, key, api)
        batch.collect(self.cfg, key, api)
        return api


class DataTests(Fixture):
    def test_generate_workers_persist_20_responses_then_resume_without_regeneration(self):
        import torch
        from contextlib import ExitStack
        cfg = {**self.cfg, 'run_dir':str(Path(self.temp.name) / 'generated')}
        out = Path(cfg['run_dir'])
        write_rows(out / 'problems.jsonl',self.problems)
        save(out / 'manifest.json',{'config':cfg,'dataset_revision':cfg['dataset_revision'],
            'problems_sha256':digest(out / 'problems.jsonl')})
        real_tensor,real_ones = torch.tensor,torch.ones
        def tensor(*args,**kwargs):
            kwargs['device']='cpu'
            return real_tensor(*args,**kwargs)
        def ones(*args,**kwargs):
            kwargs['device']='cpu'
            return real_ones(*args,**kwargs)
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.generation_config.eos_token_id = 0
        model.config.max_position_embeddings = 4096
        model.generate.side_effect = lambda **kw: real_tensor([[1,2]+list(map(ord,'2+2=4\n#### 4'))+[0]])
        with ExitStack() as stack:
            for target,value in [('exp_sae.generate.environment',None),('exp_sae.generate.space',100*2**30),
                ('torch.cuda.is_available',True),('torch.cuda.is_bf16_supported',True),
                ('torch.cuda.get_device_name','mock GPU'),('torch.cuda.reset_peak_memory_stats',None),
                ('torch.cuda.max_memory_allocated',1024),('transformers.set_seed',None),
                ('transformers.AutoTokenizer.from_pretrained',Characters()),
                ('transformers.AutoModelForCausalLM.from_pretrained',model)]:
                stack.enter_context(patch(target,return_value=value))
            stack.enter_context(patch('torch.tensor',side_effect=tensor))
            stack.enter_context(patch('torch.ones',side_effect=ones))
            generate.run(cfg,0,1)
            generate.run(cfg,1,1)
            self.assertEqual(model.generate.call_count,40)
            generate.run(cfg,0,1)
            self.assertEqual(model.generate.call_count,40)
        self.assertEqual(len(read_problem_responses(cfg,self.problems[0])),20)
        self.assertEqual(len(read_problem_responses(cfg,self.problems[1])),20)
        self.assertFalse(response_path(cfg,2).exists())

    def test_seed_ids_and_problem_partition(self):
        self.assertEqual(len({rid(i,j) for i in range(7473) for j in range(20)}), 149460)
        self.assertEqual(sample_seed(2,3), sample_seed(2,3))
        self.assertNotEqual(sample_seed(2,3), sample_seed(2,4))
        self.assertEqual(CounterPartitions(self.problems), {'discovery': 4, 'confirmation': 16})
        for p in self.problems:
            self.assertEqual({r['partition'] for r in read_problem_responses(self.cfg,p)}, {p['partition']})

    def test_grading_format_truncation_and_process_are_distinct(self):
        self.assertTrue(label_answer('wrong process\n#### 1,000.00', '#### 1000', False)['answer_correct'])
        self.assertEqual(label_answer('answer: 4', '#### 4', False)['parse_status'], 'invalid_final_line')
        self.assertFalse(label_answer('#### 4', '#### 4', True)['analysis_eligible'])
        row = {'response': '2+2=5\n#### 4'}
        self.assertEqual(label.validate(self.good_label(True), row, self.codebook)['types'], ['arith'])

    def test_generation_bundle_corruption_preserved_and_other_bundles_resume(self):
        path = response_path(self.cfg, 0)
        with path.open('ab') as handle:
            handle.write(b'broken')
        damaged = path.read_bytes()
        with self.assertRaises(ValueError):
            read_problem_responses(self.cfg, self.problems[0])
        self.assertEqual(path.read_bytes(), damaged)
        self.assertEqual(len(read_problem_responses(self.cfg, self.problems[1])), 20)
        with self.assertRaises(ValueError):
            prepared({**self.cfg, 'seed': 7})

    def test_taxonomy_sample_and_freeze_evidence(self):
        self.assertEqual(len([r for p in taxonomy.sample(self.cfg) for r in read_problem_responses(self.cfg,p)]),40)
        self.assertTrue(all(p['partition'] == 'discovery' for p in taxonomy.sample(self.cfg)))
        taxonomy.freeze(self.cfg, self.draft)
        code = copy.deepcopy(self.codebook)
        code['categories'][0]['examples'][0]['quote'] = 'not observed'
        save(self.draft, code)
        with self.assertRaises(ValueError):
            taxonomy.freeze(self.cfg, self.draft)
        code = copy.deepcopy(self.codebook)
        code['reviewed_by'] = ''
        save(self.draft, code)
        with self.assertRaises(ValueError):
            taxonomy.freeze(self.cfg, self.draft)

    def test_codebook_changes_fail_closed(self):
        p = Path(self.cfg['run_dir']) / 'taxonomy/codebook.frozen.json'
        code = read_json(p)
        code['codebook']['version'] = 'modified'
        save(p, code)
        with self.assertRaises(ValueError):
            taxonomy.frozen(self.cfg)

    def test_evidence_offsets_first_error_and_unknown_type(self):
        row = {'response': '2+2=5\n#### 4'}
        for key, value in [('first_error_char', 1), ('types', ['invented'])]:
            annotation = self.good_label(True)
            annotation[key] = value
            with self.assertRaises(ValueError):
                label.validate(annotation, row, self.codebook)
        annotation = self.good_label(True)
        annotation['evidence'][0]['end'] = 4
        with self.assertRaises(ValueError):
            label.validate(annotation, row, self.codebook)

    def test_exact_reasoning_output_alignment_including_special_token(self):
        text = '2+2=5\n#### 4'
        row = {'response': text, 'output_token_ids': list(map(ord,text)) + [0], 'finish': 'stop'}
        decoded, regions = token_regions(Characters(), row['output_token_ids'])
        selected, _ = reasoning_tokens(Characters(), row)
        self.assertEqual(decoded,text)
        self.assertEqual(selected, list(range(5)))
        self.assertFalse(regions[-1]['included'])
        with self.assertRaises(ValueError):
            reasoning_tokens(Characters(), {**row,'response': 'changed'})


def CounterPartitions(problems):
    from collections import Counter
    return dict(Counter(p['partition'] for p in problems))


class BudgetTests(Fixture):
    def req(self, name='one'):
        return batch.request(name, 'classify', {'text': 'a'}, batch.obj({'value': batch.STR}), self.cfg)

    def test_shared_pending_budget_idempotent_plan_and_budget_limit(self):
        requests = [self.req()]
        key = batch.plan(self.cfg, 'taxonomy', requests, {})
        _, first = batch.ledger(self.cfg)
        self.assertEqual(batch.plan(self.cfg, 'taxonomy', requests, {}), key)
        self.assertEqual(batch.charged(first), batch.charged(batch.ledger(self.cfg)[1]))
        self.assertGreater(batch.charged(first), 0)
        with self.assertRaises(ValueError):
            batch.plan(self.cfg, 'labels', [self.req(str(i)) for i in range(4000)], {})
        self.assertEqual(len(batch.ledger(self.cfg)[1]['batches']),1)
        self.assertEqual(batch.cost(1000000,1000000,self.cfg['labeling']),700000)

    def test_submitted_once_and_recovery_after_ambiguous_timeout(self):
        key = batch.plan(self.cfg,'labels',[self.req()],{})
        api = MagicMock()
        api.files.create.return_value.id = 'file-test'
        api.batches.create.side_effect = TimeoutError('accepted but connection lost')
        with self.assertRaises(TimeoutError):
            batch.submit(self.cfg,key,api)
        self.assertEqual(batch.ledger(self.cfg)[1]['batches'][key]['state'],'submitting')
        api.batches.list.return_value = []
        with self.assertRaises(RuntimeError):
            batch.submit(self.cfg,key,api)
        api.batches.list.return_value = [SimpleNamespace(id='recovered',input_file_id='file-test',metadata={'exp_sae_key':key})]
        self.assertEqual(batch.submit(self.cfg,key,api),'recovered')
        self.assertEqual(batch.submit(self.cfg,key,api),'recovered')
        self.assertEqual(api.batches.create.call_count,1)
        self.assertEqual(api.files.create.call_count,1)

    def test_model_unavailable_never_submits_or_uploads(self):
        key = batch.plan(self.cfg,'taxonomy',[self.req()],{})
        api = MagicMock()
        api.models.retrieve.side_effect = RuntimeError('model unavailable')
        with self.assertRaises(RuntimeError):
            batch.submit(self.cfg,key,api)
        api.batches.create.assert_not_called()
        api.files.create.assert_not_called()
        self.assertEqual(batch.ledger(self.cfg)[1]['batches'][key]['state'],'planned')

    def test_upload_failure_can_retry_before_any_paid_submission(self):
        key = batch.plan(self.cfg,'taxonomy',[self.req()],{})
        api = MagicMock()
        api.files.create.side_effect = [ConnectionError('upload failed'),SimpleNamespace(id='uploaded')]
        api.batches.create.return_value = SimpleNamespace(id='paid-once',input_file_id='uploaded')
        with self.assertRaises(ConnectionError):
            batch.submit(self.cfg,key,api)
        self.assertEqual(batch.submit(self.cfg,key,api),'paid-once')
        self.assertEqual(api.batches.create.call_count,1)

    def test_confirmation_budget_cut_never_splits_a_problem(self):
        self.cfg['labeling']['budget_usd']='0.008'
        # Fixture manifest changes are deliberate; request costs stay the subject of this unit check.
        manifest = read_json(Path(self.cfg['run_dir']) / 'manifest.json')
        manifest['config']=self.cfg
        save(Path(self.cfg['run_dir']) / 'manifest.json',manifest)
        for problem in self.problems:
            path = response_path(self.cfg,problem['pid']).with_suffix('.meta.json')
            meta = read_json(path)
            meta['config_sha256']=object_hash(self.cfg)
            save(path,meta)
        with patch('exp_sae.label.review_pilot',return_value={}):
            key = label.build(self.cfg,'confirmation')
        entry = batch.ledger(self.cfg)[1]['batches'][key]
        self.assertEqual(len(entry['context']['pids']),1)
        self.assertEqual(len(entry['context']['mapping']),20)
        self.assertLessEqual(entry['reserved_microusd'],8000)

    def test_changed_input_prevents_submission(self):
        key = batch.plan(self.cfg,'taxonomy',[self.req()],{})
        target = Path(self.cfg['run_dir']) / 'api' / f'{key}.input.jsonl'
        target.write_text('{}',encoding='utf-8')
        api = MagicMock()
        with self.assertRaises(ValueError):
            batch.submit(self.cfg,key,api)
        api.batches.create.assert_not_called()

    def test_collect_usage_reconciled_and_duplicates_reuse_labels(self):
        key = label.build(self.cfg,'pilot')
        entry = batch.ledger(self.cfg)[1]['batches'][key]
        self.assertEqual(len(entry['request_ids']),4)
        self.assertEqual(len(entry['context']['mapping']),40)
        self.complete(key)
        label.materialize(self.cfg)
        annotations = label.load(self.cfg)
        self.assertEqual(len(annotations),40)
        self.assertEqual(sum(r['label']['process_error'] for r in annotations),20)
        self.assertTrue(batch.ledger(self.cfg)[1]['batches'][key]['usage_complete'])
        self.assertLess(batch.charged(batch.ledger(self.cfg)[1]),entry['reserved_microusd'])
        self.assertEqual(label.build(self.cfg,'pilot'),[key])

    def test_failed_results_keep_reservation_and_unclassified(self):
        key = label.build(self.cfg,'pilot')
        api = self.complete(key)
        path, book = batch.ledger(self.cfg)
        book['batches'][key]['state'] = 'submitted'
        book['batches'][key].pop('actual_microusd')
        save(path,book)
        api.files.content.return_value.text = json.dumps({'custom_id':book['batches'][key]['request_ids'][0],
            'response':None,'error':{'code':'batch_expired'}})
        batch.collect(self.cfg,key,api)
        entry = batch.ledger(self.cfg)[1]['batches'][key]
        self.assertFalse(entry['usage_complete'])
        self.assertNotIn('actual_microusd',entry)
        label.materialize(self.cfg)
        self.assertTrue(all(r['label']['status']=='unclassified' for r in label.load(self.cfg)))

    def test_pilot_review_gate_and_complete_problem_sampling(self):
        with self.assertRaises(FileNotFoundError):
            label.build(self.cfg,'confirmation',1)
        key = label.build(self.cfg,'pilot')
        self.complete(key)
        label.materialize(self.cfg)
        pilot = label.load(self.cfg)
        review = Path(self.temp.name) / 'review.json'
        save(review,{'approved':True,'reviewed_by':'test','evidence_quality_notes':'mock evidence checked',
            'actual_cost_notes':'mock usage checked','pilot_sha256':object_hash(pilot)})
        label.review_pilot(self.cfg,review)
        next_key = label.build(self.cfg,'confirmation',1)
        entry = batch.ledger(self.cfg)[1]['batches'][next_key]
        self.assertEqual(len(entry['context']['mapping']),20)
        self.assertFalse(set(entry['context']['pids']) & set(batch.ledger(self.cfg)[1]['batches'][key]['context']['pids']))


class TensorTests(unittest.TestCase):
    def weights(self):
        import torch
        return {'W_enc':torch.eye(2), 'W_dec':torch.eye(2), 'b_dec':torch.tensor([1.,2.]),'b_enc':torch.zeros(2)}

    def entry(self, architecture='standard'):
        return {'architecture':architecture,'d_in':2,'d_sae':2,'input_scale':1.,'apply_b_dec_to_input':True,'k':1}

    def test_relu_topk_jump_and_input_scale(self):
        import torch
        h = torch.tensor([[4.,4.]])
        w = self.weights()
        z,recon = sae.encode(h,w,self.entry())
        torch.testing.assert_close(z,torch.tensor([[3.,2.]]))
        torch.testing.assert_close(recon,h)
        z,_ = sae.encode(h,w,self.entry('topk'))
        torch.testing.assert_close(z,torch.tensor([[3.,0.]]))
        w['threshold'] = torch.tensor([4.,1.])
        z,_ = sae.encode(h,w,self.entry('jumprelu'))
        torch.testing.assert_close(z,torch.tensor([[0.,2.]]))
        z,recon = sae.encode(h,self.weights(),{**self.entry(),'input_scale':2.})
        torch.testing.assert_close(z,torch.tensor([[7.,6.]]))
        torch.testing.assert_close(recon,h)

    def test_streamed_mean_max_frequency_and_peaks(self):
        h = np.array([[4.,4.],[1.,2.],[2.,5.]],dtype='float32')
        a,ma = extract.aggregate(h,[0,2,4],self.weights(),self.entry(),3)
        b,mb = extract.aggregate(h,[0,2,4],self.weights(),self.entry(),1)
        for k in a:
            np.testing.assert_allclose(a[k],b[k])
        self.assertEqual(ma,mb)
        np.testing.assert_allclose(a['sae_frequency'],[2/3,2/3])
        np.testing.assert_equal(a['peaks'],[0,4])
        self.assertEqual(ma['reconstruction_mse'],0.)

    def test_native_and_brainpatch_tensor_orientation_and_shapes(self):
        import torch
        entry = {**self.entry('topk'),'format':'eleuther','d_sae':3}
        source = {'encoder.weight':torch.arange(6.).reshape(3,2),'encoder.bias':torch.zeros(3),
            'W_dec':torch.arange(6.).reshape(3,2),'b_dec':torch.zeros(2)}
        with patch.object(torch.Tensor,'to',lambda self,*args,**kwargs:self):
            actual = sae.normalize_weights(source,entry)
            torch.testing.assert_close(actual['W_enc'],source['encoder.weight'].T)
            brain_entry = {**entry,'format':'brainpatch','model_revision':'a'*40,'layer':18}
            source = {'config':brain_entry,'state_dict':{'W_enc':torch.ones(3,2),
                'W_dec':torch.ones(2,3),'b_enc':torch.zeros(3),'b_dec':torch.zeros(2)}}
            actual = sae.normalize_weights(source,brain_entry)
            self.assertEqual(actual['W_dec'].shape,(3,2))
            source['state_dict']['b_enc']=torch.zeros(4)
            with self.assertRaises(ValueError):
                sae.normalize_weights(source,brain_entry)

    def test_random_qwen_full_and_chunk_hidden_match(self):
        import torch
        from transformers import Qwen2Config,Qwen2Model
        torch.manual_seed(42)
        model = Qwen2Model(Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,attention_dropout=0.)).eval()
        prompt,output = [1,2,3],[4,5,6,7,8]
        captured = {}
        def hook(_m,_a,value):
            captured['h'] = value[0] if isinstance(value,tuple) else value
        with torch.inference_mode():
            handle = model.layers[0].register_forward_hook(hook)
            model(input_ids=torch.tensor([prompt+output]),use_cache=False)
            reference = captured['h'][0,len(prompt):].clone()
            handle.remove()
            actual = torch.cat([h.clone() for _,h in hidden_chunks(model,0,prompt,output,2,'cpu')])
            torch.testing.assert_close(reference,actual,rtol=1e-4,atol=1e-5)
            self.assertFalse(model.layers[0]._forward_hooks)


class StorageAndAnalysisTests(Fixture):
    def test_disk_reserve_stops_before_new_output_allocation(self):
        stat = lambda path,*a,**k: SimpleNamespace(st_dev=0 if str(path)=='/' else 1)
        with patch('pathlib.Path.mkdir'), patch('exp_sae.common.os.stat',side_effect=stat), \
                patch('exp_sae.common.shutil.disk_usage',return_value=SimpleNamespace(free=16*2**30)):
            self.assertEqual(space(self.cfg),16*2**30)
            with self.assertRaises(OSError):
                space(self.cfg,2*2**30)

    def test_extract_oom_preserves_hidden_and_resume_commits_then_deletes(self):
        import torch
        from contextlib import ExitStack
        entry = {'id':'synthetic','d_in':2,'d_sae':2,'layer':0,'status':'metadata_verified',
            'architecture':'standard','input_scale':1.,'apply_b_dec_to_input':True}
        weights = {'W_enc':torch.eye(2),'W_dec':torch.eye(2),'b_enc':torch.zeros(2),'b_dec':torch.zeros(2)}
        checkpoint = Path(self.cfg['cache_dir']) / 'sae/owned.safetensors'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'synthetic checkpoint')
        entry['weights_sha256']=digest(checkpoint)
        def load(*args):
            if not checkpoint.exists():
                checkpoint.write_bytes(b'synthetic checkpoint')
            return weights,checkpoint
        model = MagicMock()
        model.to.return_value=model
        model.eval.return_value=model
        model.config.hidden_size=1536
        model.model.layers=[None]*28
        model.config.max_position_embeddings=4096
        def chunks(_model,_layer,_prompt,output,*args):
            yield 0,torch.ones((len(output),2))
        with ExitStack() as stack:
            for target,value in [('exp_sae.extract.environment',None),('exp_sae.extract.space',100*2**30),
                ('torch.cuda.is_available',True),('torch.cuda.is_bf16_supported',True),
                ('torch.cuda.get_device_name','mock GPU'),
                ('torch.cuda.reset_peak_memory_stats',None),('torch.cuda.max_memory_allocated',1024),
                ('torch.cuda.empty_cache',None),('transformers.AutoTokenizer.from_pretrained',Characters()),
                ('transformers.AutoModelForCausalLM.from_pretrained',model),
                ('exp_sae.sae.registry',{'entries':[entry]})]:
                stack.enter_context(patch(target,return_value=value))
            stack.enter_context(patch('exp_sae.sae.load',side_effect=load))
            replay = stack.enter_context(patch('exp_sae.extract.hidden_chunks',side_effect=chunks))
            with patch('exp_sae.extract.aggregate',side_effect=torch.OutOfMemoryError('synthetic OOM')):
                with self.assertRaises(torch.OutOfMemoryError):
                    extract.run(self.cfg,limit=2)
            hidden = list((Path(self.cfg['run_dir']) / 'extraction').rglob('temporary_hidden/*.npz'))
            self.assertEqual(len(hidden),1)
            extract.run(self.cfg,limit=2)
            self.assertFalse(hidden[0].exists())
            self.assertFalse(checkpoint.exists())
            count=replay.call_count
            extract.run(self.cfg,limit=2)
            self.assertEqual(replay.call_count,count)
            _,_,quality,state = extract.load_entry(self.cfg,entry,require_complete=False)
            self.assertEqual(len(quality),40)
            self.assertFalse(state['complete'])
            with self.assertRaises(ValueError):
                extract.load_entry(self.cfg,entry)

    def test_hidden_deletion_requires_verified_shapes_ids_and_hashes(self):
        folder = Path(self.temp.name) / 'bundle'
        base = folder / '00000'
        hidden = folder / 'temporary_hidden/a.npz'
        hidden.parent.mkdir(parents=True)
        atomic_npz(hidden,hidden=np.ones((2,2),dtype='float32'))
        save(hidden.with_suffix('.json'),{'id':'a','sha256':digest(hidden)})
        with self.assertRaises(FileNotFoundError):
            extract.delete_hidden(hidden,base,['a'],'signature',{'d_sae':2,'d_in':2},['a'])
        self.assertTrue(hidden.exists())
        arrays = {k:np.ones((1,2),dtype='int32' if k=='peaks' else 'float32')
            for k in ['sae_mean','sae_max','sae_frequency','peaks','hidden_mean']}
        atomic_npz(base.with_suffix('.npz'),ids=np.array(['a']),**arrays)
        write_rows(base.with_suffix('.jsonl'),[{'id':'a','status':'ok','hidden_sha256':digest(hidden),'tokens':2}])
        meta = {'signature':'signature','feature_ids':['a'],'feature_sha256':digest(base.with_suffix('.npz')),
            'quality_sha256':digest(base.with_suffix('.jsonl'))}
        save(base.with_suffix('.meta.json'),meta)
        with self.assertRaises(ValueError):
            extract.delete_hidden(hidden,base,['wrong'],'signature',{'d_sae':2,'d_in':2},['a'])
        with self.assertRaises(ValueError):
            extract.delete_hidden(hidden,base,['a'],'signature',{'d_sae':3,'d_in':2},['a'])
        self.assertTrue(hidden.exists())
        extract.delete_hidden(hidden,base,['a'],'signature',{'d_sae':2,'d_in':2},['a'])
        self.assertFalse(hidden.exists())
        self.assertFalse(hidden.with_suffix('.json').exists())

    def test_confirmation_never_changes_scaler_or_pca(self):
        rng = np.random.default_rng(42)
        train,test = rng.random((12,8)),rng.random((5,8))
        a,_,first = analyze.transformed(train,test,3,42)
        b,_,second = analyze.transformed(train,test*10000,3,42)
        np.testing.assert_allclose(first['scaler'].mean_,second['scaler'].mean_)
        np.testing.assert_allclose(first['pca'].components_,second['pca'].components_)
        np.testing.assert_allclose(a,b)

    def test_bootstrap_resamples_problem_groups_and_sparse_cases(self):
        result = analyze.bootstrap_metrics([0,1]*8,[.1,.9]*8,np.repeat(np.arange(8),2),30,42)
        self.assertEqual(result['auprc'],1.)
        self.assertEqual(result['f1_ci95'],[1.,1.])
        support = analyze.support([1,1,1,0],[5,5,5,6])
        self.assertEqual(support['positive_problems'],1)

    def test_within_problem_contrasts_exclude_overlap(self):
        records = [{'id':str(i),'pid':i//3} for i in range(6)]
        annotations = {str(i):{'label':{'types':[['A'],['B'],['A','B']][i%3]}} for i in range(6)}
        x = np.array([[4.,2.],[1.,1.],[1000.,1000.]]*2)
        diffs = analyze.paired_differences(x,records,annotations,'A','B')
        np.testing.assert_allclose(diffs,[[3.,1.],[3.,1.]])

    def test_offline_labels_features_analysis_report(self):
        for scope in ('discovery','pilot'):
            key = label.build(self.cfg,scope)
            self.complete(key)
            label.materialize(self.cfg)
        annotations = label.load(self.cfg)
        ids = {r['id'] for r in annotations}
        entry = {'id':'synthetic','status':'metadata_verified','layer':0,'d_in':2,'d_sae':3,
            'revision_match':'synthetic fixture only','reason':''}
        out = Path(self.cfg['run_dir']) / 'extraction'
        directory = out / 'synthetic/test'
        signature = 'synthetic-signature'
        for problem in self.problems:
            records = read_problem_responses(self.cfg,problem)
            base = directory / f'{problem["pid"]:05d}'
            base.parent.mkdir(parents=True,exist_ok=True)
            kept = [r for r in records if r['id'] in ids]
            n = len(kept)
            x = np.array([[float(r['repeat']%2),float(1-r['repeat']%2),0.] for r in kept],dtype='float32').reshape(n,3)
            quality = [{'id':r['id'],'status':'ok','pid':r['pid'],
                'offsets':[(i,i+1) for i in range(len(r['response']))],
                'reconstruction_mse':0.,'relative_squared_error':0.,'mean_l0':1.,'active_fraction':1/3}
                for r in records]
            arrays = {'sae_mean':x,'sae_max':x*2,'sae_frequency':x,'peaks':np.zeros((n,3),dtype='int32'),
                'hidden_mean':x[:,:2]}
            atomic_npz(base.with_suffix('.npz'),ids=np.array([r['id'] for r in kept],dtype='U40'),**arrays)
            write_rows(base.with_suffix('.jsonl'),quality)
            save(base.with_suffix('.meta.json'),{'signature':object_hash([signature,digest(response_path(self.cfg,problem['pid']))]),
                'feature_sha256':digest(base.with_suffix('.npz')),'quality_sha256':digest(base.with_suffix('.jsonl')),
                'feature_ids':[r['id'] for r in kept]})
        save(directory / 'run.json',{'signature':signature,'labels_sha256':object_hash(annotations),'complete':True,
            'processed_problems':len(self.problems),'peak_cuda_bytes':0,'seconds_this_invocation':0,'disk_free_bytes':100*2**30})
        save(out / 'index.json',{'synthetic':'synthetic/test'})
        with patch('exp_sae.sae.registry',return_value={'entries':[entry]}), patch('exp_sae.analyze.space'):
            analyze.run(self.cfg)
            analyze.run(self.cfg)
            report.run(self.cfg)
        result = read_json(Path(self.cfg['run_dir']) / 'analysis/synthetic.json')
        self.assertEqual(result['types'][0]['baselines']['sae']['auprc'],1.)
        self.assertEqual(set(result['types'][0]['baselines']),{'sae','raw_hidden','length','correctness','length_correctness'})
        self.assertTrue(result['types'][0]['max_activation_examples'])
        self.assertEqual(read_json(Path(self.cfg['run_dir']) / 'report/quality.json')['generated_responses'],400)
        self.assertTrue((Path(self.cfg['run_dir']) / 'report/synthetic.png').exists())


if __name__ == '__main__':
    unittest.main()
