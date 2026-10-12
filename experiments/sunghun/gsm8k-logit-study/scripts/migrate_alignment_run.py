"""Create a fresh extraction run, preserving verified generation bytes unchanged."""
import argparse
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsm8k_lens.pipeline import stage_fingerprint
from gsm8k_lens.storage import complete, file_digest, read_json, source_digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--extra-generations', required=True)
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    source, extra, target = map(lambda p: Path(p).resolve(),
                                (args.source, args.extra_generations, args.target))
    if target.exists():
        raise ValueError('Target must not exist; preserve prior runs')
    manifest = read_json(source / 'manifest.json')
    fingerprint = stage_fingerprint(manifest, 'generate')
    ids = read_json(source / 'sample_ids.json')
    for name, checksum in manifest['data_files'].items():
        if file_digest(source / name) != checksum:
            raise ValueError(f'Corrupt prepared input: {name}')
    selected = {}
    for root in (source / 'generations', extra / 'generations'):
        for marker in sorted(root.glob('*/*/complete.json')):
            folder = marker.parent
            phase, sample = folder.parent.name, folder.name
            if sample not in ids[phase] or not complete(folder, fingerprint, verify=True):
                raise ValueError(f'Invalid generation: {folder}')
            record = read_json(folder / 'generation.json')
            if record['id'] != sample or record['phase'] != phase:
                raise ValueError('Generation identity mismatch')
            key = (phase, sample)
            checksum = file_digest(folder / 'generation.json')
            if key in selected and selected[key][1] != checksum:
                raise ValueError(f'Conflicting generation: {key}')
            selected[key] = (folder, checksum)
    if sum(p == 'pilot' for p, _ in selected) != manifest['config']['dataset']['pilot_size']:
        raise ValueError('Pilot generations must all be preserved')
    environment = source / 'generation_environment.json'
    if not environment.is_file():
        raise ValueError('Original generation environment record is required')
    target.mkdir()
    shutil.copy2(environment, target / environment.name)
    for name in manifest['data_files']:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    for (phase, sample), (folder, checksum) in selected.items():
        destination = target / 'generations' / phase / sample
        shutil.copytree(folder, destination)
        if file_digest(destination / 'generation.json') != checksum:
            raise ValueError('Copy integrity failure')
    origin_hash = file_digest(source / 'manifest.json')
    manifest.update(source_hash=source_digest(), created_unix=time.time(),
                    migration={'parent_manifest_sha256': origin_hash,
                               'parent_source_hash': read_json(source / 'manifest.json')['source_hash'],
                               'reason': 'exact original-token ByteLevel alignment repair'})
    write_json(target / 'manifest.json', manifest)
    write_json(target / 'migration.json', {
        'generation_fingerprint': fingerprint, 'parent_manifest_sha256': origin_hash,
        'generation_environment_sha256': file_digest(environment),
        'counts': {phase: sum(p == phase for p, _ in selected) for phase in ids},
        'generation_checksums': {f'{p}/{s}': checksum for (p, s), (_, checksum) in selected.items()},
        'features_reused': False, 'labels_reused': False, 'gates_reused': False})
    print(f'Migrated {len(selected)} unchanged generations to {target}')


if __name__ == '__main__':
    main()
