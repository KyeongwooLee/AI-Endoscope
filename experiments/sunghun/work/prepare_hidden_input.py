import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
run = root / 'outputs/vessl-qwen-gsm8k-fp32-v6'
dest = root / 'work/hidden_input'
dest.mkdir(exist_ok=True)
with (dest / 'examples.jsonl').open('w', encoding='utf-8') as stream:
    for phase in ('exploration', 'test'):
        for path in sorted((run / 'generations' / phase).glob('*/generation.json')):
            generation = json.loads(path.read_text(encoding='utf-8'))
            label = json.loads((run / 'labels' / phase / (generation['id'] + '.json')).read_text())
            record = {key: generation[key] for key in ('id', 'phase', 'prompt_token_ids', 'generated_token_ids')}
            record['incorrect'] = int(not label['answer_correct'])
            record['generation_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            stream.write(json.dumps(record) + '\n')
(dest / 'splits.json').write_bytes((root / 'outputs/error-classifier-v1/split_ids.json').read_bytes())
with zipfile.ZipFile(root / 'work/hidden_input.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in dest.iterdir():
        archive.write(path, path.name)
print('Prepared replay input:', (root / 'work/hidden_input.zip').stat().st_size)
