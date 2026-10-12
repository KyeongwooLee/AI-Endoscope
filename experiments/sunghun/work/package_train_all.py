import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
original = root / 'outputs/vessl-qwen-gsm8k-fp32-v6'
confirmation = root / 'outputs/entropy-confirmation-v1'
project = root / 'outputs/gsm8k-logit-study'
destination = root / 'work/train-all-input/input'
destination.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = json.loads((original / 'manifest.json').read_text())
config = json.loads((confirmation / 'input/config.json').read_text())
assert config['model']['revision'] == manifest['model_revision']
assert config['dataset']['revision'] == manifest['dataset_revision']
for section in ('prompt', 'generation'):
    assert config[section] == manifest['config'][section]
assert config['model']['dtype'] == manifest['config']['model']['dtype'] == 'float32'
gold = json.loads((original / 'data/gold.json').read_text(encoding='utf-8'))
known, sources = [], {}
for phase in ('pilot', 'exploration'):
    for line in (original / 'data' / (phase + '.jsonl')).open(encoding='utf-8'):
        row = json.loads(line)
        path = original / 'generations' / phase / row['id'] / 'generation.json'
        receipt = json.loads(path.with_name('complete.json').read_text())
        assert sha(path) == receipt['files']['generation.json']
        generation = json.loads(path.read_text(encoding='utf-8'))
        known.append({'id': row['id'], 'question': row['question'], 'gold': gold[row['id']],
                      'origin': 'old_' + phase, 'generation': generation})
        sources[row['id']] = {'sha256': sha(path), 'source': path.relative_to(root).as_posix()}
backup = json.loads((confirmation / 'local_backup_receipt.json').read_text())
for row in json.loads((confirmation / 'questions.json').read_text(encoding='utf-8')):
    path = confirmation / 'records' / (row['id'] + '.json')
    assert sha(path) == backup['files']['records/' + path.name]['sha256']
    record = json.loads(path.read_text(encoding='utf-8'))
    known.append({'id': row['id'], 'question': row['question'], 'gold': row['gold'],
                  'origin': 'old_confirmation', 'generation': record['generation']})
    sources[row['id']] = {'sha256': sha(path), 'source': path.relative_to(root).as_posix()}
assert len(known) == len({r['id'] for r in known}) == 2128
assert all(r['id'].startswith('train-') for r in known)
(destination / 'known.jsonl').write_text('\n'.join(json.dumps(r) for r in known) + '\n', encoding='utf-8')
(destination / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
(destination / 'provenance.json').write_text(json.dumps({
    'known_count': len(known), 'new_count': 5345, 'official_test_included': False,
    'original_manifest_sha256': sha(original / 'manifest.json'),
    'confirmation_manifest_sha256': sha(confirmation / 'manifest.json'),
    'source_records': sources}, indent=2) + '\n')
package = root / 'work/train-all-input.zip'
with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in destination.iterdir():
        archive.write(path, 'input/' + path.name)
    for path in (project / 'gsm8k_lens').glob('*.py'):
        archive.write(path, 'gsm8k_lens/' + path.name)
    archive.write(project / 'scripts/extract_train_all.py', 'extract_train_all.py')
print(json.dumps({'bytes': package.stat().st_size, 'sha256': sha(package), 'known': len(known), 'new': 5345}))
