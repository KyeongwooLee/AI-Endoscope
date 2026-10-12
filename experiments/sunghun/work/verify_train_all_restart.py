"""Validate a replacement GPU environment against saved generation and feature data."""
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/tmp/gsm8k-train-all-v1')
spec = importlib.util.spec_from_file_location('frozen_extractor', ROOT / 'extract_train_all.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
config = json.loads((ROOT / 'input/config.json').read_text())
manifest = json.loads((ROOT / 'manifest.json').read_text())
questions = {r['id']: r for r in json.loads((ROOT / 'questions.json').read_text())}
records = json.loads((ROOT / 'resume_checks/records.json').read_text())
torch.set_num_threads(4)
model, tokenizer = module.load_model(config, config['model']['revision'])
assert module.digest(tokenizer.chat_template) == module.CHAT_HASH
checks = []
with np.load(ROOT / 'resume_checks/features.npz', allow_pickle=False) as saved:
    for index, record in enumerate(records):
        identifier = record['id']
        generation = record['generation']
        module.verify_prompt(tokenizer, generation, questions[identifier]['question'], config)
        current, quality = module.extract(model, generation, causal_check=True)
        assert quality['matches'] == quality['positions']
        selection = saved['ids'] == identifier
        np.testing.assert_array_equal(current['checkpoints'], saved['checkpoints'][selection])
        errors = {}
        for field in ('hidden', 'trajectory'):
            expected = saved[field][selection]
            np.testing.assert_allclose(current[field], expected, atol=1e-3, rtol=3e-4)
            errors[field] = float(np.abs(current[field] - expected).max())
        location = list(saved['readout_ids']).index(identifier)
        expected = saved['readouts'][saved['readout_offsets'][location]:saved['readout_offsets'][location + 1]]
        np.testing.assert_allclose(current['readouts'], expected, atol=1e-3, rtol=3e-4)
        errors['readouts'] = float(np.abs(current['readouts'] - expected).max())
        if index == 0:
            regenerated = module.generate_example(model, tokenizer, questions[identifier]['question'], config)
            assert regenerated['generated_token_ids'] == generation['generated_token_ids'], 'Greedy replay differs'
        checks.append({'id': identifier, 'quality': quality, 'max_abs_errors': errors,
                       'greedy_regenerated_exactly': index == 0})
        print('RESTART_CHECK_PASSED', identifier, errors, flush=True)
module.write_json(ROOT / 'restart_quality_gate.json', {
    'passed': True, 'fingerprint': manifest['fingerprint'],
    'environment': module.environment(), 'checks': checks})
del model, tokenizer
gc.collect()
torch.cuda.empty_cache()
print('RESTART_GPU_VALIDATION_PASSED', flush=True)
