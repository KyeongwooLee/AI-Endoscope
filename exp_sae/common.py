import contextlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
import time
from pathlib import Path

from sae_error_clustering.common import (atomic_json, atomic_npz as _atomic_npz,
                                         digest, object_hash, read_json)


def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_rows(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def sync(path):
    with Path(path).open('r+b') as handle:
        os.fsync(handle.fileno())
    if os.name == 'posix':
        fd = os.open(Path(path).parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, value)
    sync(path)


def atomic_npz(path, **arrays):
    _atomic_npz(path, **arrays)
    sync(path)


def load_config(path):
    cfg = read_json(path)
    for key in ('model_revision', 'tokenizer_revision'):
        value = cfg[key]
        if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{key} must be a full immutable commit SHA')
    if len(cfg['dataset_revision']) != 40 or any(c not in '0123456789abcdef' for c in cfg['dataset_revision']):
        raise ValueError('dataset_revision must be a full immutable commit SHA')
    if cfg['model'] != 'Qwen/Qwen2.5-1.5B-Instruct':
        raise ValueError('This experiment is scoped to Qwen2.5-1.5B-Instruct')
    for key in ('repeats', 'workers', 'chunk_size', 'expected_problems', 'discovery_problems', 'pilot_problems'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if not math.isfinite(cfg['min_free_gib']) or cfg['min_free_gib'] < 15:
        raise ValueError('Keep at least 15 GiB free on the data disk')
    generation = cfg['generation']
    if (not 0 < generation['temperature'] < 10 or not 0 < generation['top_p'] <= 1
            or type(generation['max_new_tokens']) is not int or generation['max_new_tokens'] <= 0):
        raise ValueError('Invalid sampling parameters')
    if any(type(value) is not int or value <= 0 for value in cfg['analysis'].values()):
        raise ValueError('Analysis settings must be positive integers')
    if cfg['analysis']['bootstrap_runs'] < 10:
        raise ValueError('At least 10 problem bootstrap replicates are required')
    for key in ('run_dir', 'cache_dir'):
        cfg[key] = str(Path(cfg[key]).resolve())
    run, cache = Path(cfg['run_dir']), Path(cfg['cache_dir'])
    if run == cache or run in cache.parents or cache in run.parents:
        raise ValueError('Run and cache directories must be separate siblings')
    return cfg


def space(cfg, extra_bytes=0):
    run, cache = Path(cfg['run_dir']), Path(cfg['cache_dir'])
    for path in (run, cache):
        path.mkdir(parents=True, exist_ok=True)
    if os.stat(run).st_dev != os.stat(cache).st_dev:
        raise ValueError('Run and HF cache must be on the same data disk')
    if os.name == 'posix' and os.stat(run).st_dev == os.stat('/').st_dev:
        raise ValueError('Run/cache are on the root filesystem; mount the data disk before running')
    free = shutil.disk_usage(run).free
    if free - extra_bytes < cfg['min_free_gib'] * 2**30:
        raise OSError('Data disk reserve would fall below 15 GiB; existing results preserved')
    return free


def environment(cfg):
    cache = Path(cfg['cache_dir'])
    space(cfg)
    # Must run before importing HF/torch so root never receives model or kernel caches.
    for key, value in {'HF_HOME': cache / 'hf', 'HF_HUB_CACHE': cache / 'hf/hub',
                       'HF_DATASETS_CACHE': cache / 'datasets', 'TORCH_HOME': cache / 'torch',
                       'XDG_CACHE_HOME': cache, 'TMPDIR': cache / 'tmp',
                       'TEMP': cache / 'tmp', 'TMP': cache / 'tmp',
                       'TRITON_CACHE_DIR': cache / 'triton', 'CUDA_CACHE_PATH': cache / 'cuda',
                       'MPLCONFIGDIR': cache / 'matplotlib'}.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(value)
    tempfile.tempdir = str(cache / 'tmp')


@contextlib.contextmanager
def lock(directory):
    path = Path(directory) / '.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(json.dumps({'pid': os.getpid(), 'time': time.time()}))
        yield
    finally:
        path.unlink()


def runtime():
    packages = {}
    for name in ('numpy', 'torch', 'transformers', 'datasets', 'huggingface-hub',
                 'safetensors', 'scikit-learn', 'openai'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = list(Path(__file__).parent.glob('*.py'))
    source += [Path(__file__).parent.parent / 'sae_error_clustering' / name
               for name in ('tokens.py', 'common.py', 'prepare.py', 'extract.py')]
    return {'python': platform.python_version(), 'packages': packages,
            'code': {str(p.relative_to(Path(__file__).parent.parent)): digest(p) for p in source}}


def prepared(cfg):
    run = Path(cfg['run_dir'])
    manifest = read_json(run / 'manifest.json')
    if manifest['config'] != cfg or digest(run / 'problems.jsonl') != manifest['problems_sha256']:
        raise ValueError('Config or prepared dataset changed; use a new run directory')
    return list(rows(run / 'problems.jsonl')), manifest


def rid(pid, repeat):
    return f'gsm8k-train-{pid:05d}-r{repeat:02d}'


def sample_seed(pid, repeat, seed=42):
    return int(object_hash([seed, pid, repeat])[:8], 16)


def response_path(cfg, pid):
    return Path(cfg['run_dir']) / 'responses' / f'{pid:05d}.jsonl'


def read_problem_responses(cfg, problem):
    path = response_path(cfg, problem['pid'])
    meta = read_json(path.with_suffix('.meta.json'))
    records = list(rows(path))
    expected = [rid(problem['pid'], i) for i in range(cfg['repeats'])]
    if (digest(path) != meta['sha256'] or [r['id'] for r in records] != expected
            or meta['config_sha256'] != object_hash(cfg)):
        raise ValueError(f'Corrupt/incompatible response bundle: {path}')
    for record in records:
        if (record['pid'] != problem['pid'] or record['partition'] != problem['partition']
                or record['question'] != problem['question'] or record['gold'] != problem['gold']):
            raise ValueError('Response problem/partition mismatch')
        for key in ('prompt_token_ids', 'output_token_ids'):
            if not record[key] or any(type(t) is not int or t < 0 for t in record[key]):
                raise ValueError('Invalid response token IDs')
        from sae_error_clustering.tokens import label_answer
        grade = label_answer(record['response'], record['gold'], record['finish'] == 'length')
        if (record['finish'] not in ('stop', 'length') or any(record[k] != v for k,v in grade.items())
                or record['id'] != rid(record['pid'], record['repeat'])
                or record['seed'] != sample_seed(record['pid'], record['repeat'], cfg['seed'])
                or record['generation'] != cfg['generation']):
            raise ValueError('Response grading/seed/generation mismatch')
    return records


def responses(cfg, require_complete=True):
    problems, _ = prepared(cfg)
    for problem in problems:
        path = response_path(cfg, problem['pid'])
        if not path.exists() and not path.with_suffix('.meta.json').exists() and not require_complete:
            continue
        yield from read_problem_responses(cfg, problem)
