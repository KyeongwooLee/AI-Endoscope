import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
original = root / 'outputs/vessl-qwen-gsm8k-fp32-v6'
ablation = root / 'outputs/feature-ablation-v1'
dest = root / 'work/confirmation-input'
dest.mkdir(exist_ok=True)
manifest = json.loads((original / 'manifest.json').read_text())
config = manifest['config']
config['model']['revision'] = manifest['model_revision']
config['dataset']['revision'] = manifest['dataset_revision']
(dest / 'config.json').write_text(json.dumps(config, indent=2))
excluded_ids, question_hashes = [], []
for phase in ('pilot', 'exploration', 'test'):
    for line in (original / 'data' / (phase + '.jsonl')).open(encoding='utf-8'):
        row = json.loads(line)
        excluded_ids.append(row['id'])
        question_hashes.append(hashlib.sha256(' '.join(row['question'].split()).encode()).hexdigest())
(dest / 'excluded.json').write_text(json.dumps({'ids': excluded_ids, 'question_hashes': question_hashes}, indent=2))
fields = ['current_stability_L10_entropy_prefix_mean', 'current_stability_L7_entropy_prefix_mean',
          'change_stability_L7_entropy_prefix_mean']
with np.load(ablation / 'input/designs.npz', allow_pickle=False) as data:
    ids, y, train = [data[key] for key in ('ids', 'y', 'train')]
frame = pd.read_parquet(ablation / 'input/engineered_features.parquet').loc[ids[train], fields]
np.savez_compressed(dest / 'train_candidates.npz', ids=ids[train], y=y[train], features=frame.to_numpy(dtype=np.float32))
(dest / 'original_inner_split.json').write_bytes((ablation / 'results/split_ids.json').read_bytes())
for name in ('raw_initial_mlp.pt', 'raw_initial_scaler.joblib'):
    (dest / name).write_bytes((ablation / 'results' / name).read_bytes())
protocol = {'sample_count': 1000, 'sampling_seed': 20261009, 'primary': 'negative_layer10_entropy_prefix_mean',
    'secondary': ['negative_layer7_entropy_prefix_mean', 'negative_layer7_entropy_change', 'combo_linear', 'combo_mlp', 'raw_initial_mlp'],
    'candidate_fields': fields, 'directions': [-1, -1, -1], 'checkpoint': 32,
    'selection': 'Remaining GSM8K train, excluding old pilot/exploration/test IDs and normalized question duplicates.',
    'fitting': 'Original train only, 639/160 inner split; no new-data tuning.',
    'chat_template_hash': manifest['chat_template_hash'], 'expected_layers': 28,
    'limits': ['Fresh to this project, not necessarily unseen in Qwen pretraining.',
              'Fixed risk discrimination evaluation, not an answer-accuracy intervention.',
              'Primary population still generating at t32; early completions reported separately.',
              'No labels, final-layer features, actual next-token identities, or future positions are predictors.']}
(dest / 'protocol.json').write_text(json.dumps(protocol, indent=2))
with zipfile.ZipFile(root / 'work/confirmation-input.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in dest.iterdir():
        archive.write(path, 'input/' + path.name)
    project = root / 'outputs/gsm8k-logit-study'
    for path in (project / 'gsm8k_lens').glob('*.py'):
        archive.write(path, 'gsm8k_lens/' + path.name)
    for name in ('train_hidden_probes.py', 'train_error_classifier.py'):
        archive.write(project / 'scripts' / name, name)
print('Prepared confirmation package:', (root / 'work/confirmation-input.zip').stat().st_size)
