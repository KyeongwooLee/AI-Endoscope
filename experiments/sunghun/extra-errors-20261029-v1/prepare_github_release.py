"""Package verified original checkpoints without modifying their bytes."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'repo'
CODE = REPO / 'experiments/sunghun/extra-errors-20261029-v1'
OUT = ROOT / 'github_release'
PREFIX = 'qwen-extra-errors-20261029-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    CODE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=False)
    audit = json.loads((ROOT / 'FINAL_AUDIT.json').read_text())
    assert audit['complete'] and audit['selected_errors'] == audit['verified_features'] == 4000
    assert audit['generated'] == 18656 and audit['taxonomy_status'] == 'unannotated'
    frozen = json.loads((ROOT / 'backup/run_manifest.json').read_text())
    for name, digest in frozen['files'].items():
        p = ROOT / name if name in ['worker.py', 'protocol.py', 'control.py'] else ROOT / 'backup' / name
        assert sha(p) == digest, name
    # Preserve the exact producer code, including its historical parser limitation.
    # The separate, corrected audit uses LF-only JSONL iteration.
    for name in ['worker.py', 'protocol.py', 'control.py', 'audit_local.py',
                 'launch_worker.py', 'prefetch_model.py', 'prefetch_baseline.py',
                 'prepare_inputs.py', 'test_protocol.py', 'test_jsonl_unicode.py']:
        shutil.copyfile(ROOT / name, CODE / name)
    for name in ['FINAL_AUDIT.json', 'run_manifest.json', 'environment.json']:
        shutil.copyfile((ROOT if name == 'FINAL_AUDIT.json' else ROOT / 'backup') / name, CODE / name)
    for name in ['LICENSE-GSM8K.txt', 'LICENSE-SVAMP.txt']:
        shutil.copyfile(REPO / 'experiments/sunghun/posthoc-transfer-data-v1' / name, CODE / name)
    shutil.copyfile(Path(__file__), CODE / 'prepare_github_release.py')

    outbox = ROOT / 'backup/outbox'
    files = []
    for rp in sorted(outbox.glob('*.json')):
        receipt = json.loads(rp.read_text(encoding='utf-8'))
        zp = rp.with_suffix('.zip')
        assert zp.stat().st_size == receipt['bytes'] and sha(zp) == receipt['archive_sha256'], zp.name
        # The prior full audit checked every original inner member and feature link.
        # Check the complete inner member hash list again while packaging.
        with zipfile.ZipFile(zp) as z:
            assert set(z.namelist()) == set(receipt['members'])
            for n, h in receipt['members'].items():
                assert hashlib.sha256(z.read(n)).hexdigest() == h, n
        files += [(zp, f'{PREFIX}/backup/outbox/{zp.name}'),
                  (rp, f'{PREFIX}/backup/outbox/{rp.name}')]
    assert len(files) == 2332
    parts, current, size = [], [], 0
    for source, name in files:
        if current and size + source.stat().st_size > 1024**3 - 1024**2:
            parts.append(current)
            current, size = [], 0
        current.append((source, name))
        size += source.stat().st_size
    if current:
        parts.append(current)
    index = {'format': 'independent ZIPs; extract all into the same directory; do not concatenate',
             'source_commit': frozen['source_commit'], 'archives': {}}
    for number, entries in enumerate(parts, 1):
        name = f'qwen-extra-errors-full-part-{number:02d}.zip'
        path = OUT / name
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
            for source, member in entries:
                z.write(source, member)
        assert path.stat().st_size < 2 * 1024**3
        with zipfile.ZipFile(path) as z:
            for source, member in entries:
                assert hashlib.sha256(z.read(member)).hexdigest() == sha(source)
        index['archives'][name] = {'bytes': path.stat().st_size, 'sha256': sha(path),
            'members': {member: sha(source) for source, member in entries}}
        print(f'PACKAGED {name} {path.stat().st_size}', flush=True)
    save(OUT / 'FULL_INDEX.json', index)
    for name in ['README.md', 'DATA_SCHEMA.json', 'load_data.py', 'FINAL_AUDIT.json',
                 'LICENSE-GSM8K.txt', 'LICENSE-SVAMP.txt', 'ATTRIBUTION.md']:
        shutil.copyfile(CODE / name, OUT / name)
    shutil.copyfile(ROOT / 'new_errors.jsonl', OUT / 'new_errors.jsonl')
    shutil.copyfile(ROOT / 'backup/summary.json', OUT / 'dataset_summary.json')
    with zipfile.ZipFile(OUT / 'AI_STARTER.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['new_errors.jsonl', 'README.md', 'DATA_SCHEMA.json', 'load_data.py',
                     'FINAL_AUDIT.json', 'dataset_summary.json', 'ATTRIBUTION.md',
                     'LICENSE-GSM8K.txt', 'LICENSE-SVAMP.txt']:
            z.write(OUT / name, f'{PREFIX}/{name}')
    with zipfile.ZipFile(OUT / 'qwen-extra-errors-metadata.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / 'backup').rglob('*'):
            if p.is_file() and 'outbox' not in p.relative_to(ROOT / 'backup').parts:
                z.write(p, f'{PREFIX}/backup/{p.relative_to(ROOT / "backup").as_posix()}')
        for p in CODE.iterdir():
            if p.is_file():
                z.write(p, f'{PREFIX}/code/{p.name}')
        # Frozen imports are distributed alongside the pinned source commit for offline inspection.
        for sub in ['gsm8k-logit-study/gsm8k_lens', 'smollm2-detailed-v1']:
            for p in (REPO / 'experiments/sunghun' / sub).rglob('*.py'):
                z.write(p, f'{PREFIX}/source/{p.relative_to(REPO / "experiments/sunghun").as_posix()}')
    secret = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{24,}')
    for p in CODE.iterdir():
        if p.is_file():
            assert not secret.search(p.read_bytes()), p.name
    with zipfile.ZipFile(OUT / 'qwen-extra-errors-metadata.zip') as z:
        for n in z.namelist():
            assert not secret.search(z.read(n)), n
    hashes = {p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    (OUT / 'SHA256SUMS.txt').write_text(''.join(f'{h}  {n}\n' for n, h in hashes.items()), encoding='ascii')
    save(OUT / 'READY_TO_PUBLISH.json', {'complete': True, 'audit': audit, 'assets': hashes,
         'original_batch_archives': 1166, 'full_parts': len(parts)})
    shutil.copyfile(OUT / 'FULL_INDEX.json', CODE / 'FULL_INDEX.json')
    save(CODE / 'PUBLICATION_MANIFEST.json', {'release_tag': PREFIX, 'assets': hashes,
         'source_commit': frozen['source_commit'], 'original_batch_archives': 1166})
    print('READY', len(hashes), 'assets', flush=True)


if __name__ == '__main__':
    main()
