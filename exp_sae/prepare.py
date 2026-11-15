from pathlib import Path

from sae_error_clustering.prepare import assign_splits
from sae_error_clustering.tokens import parse_answer
from .common import (digest, environment, lock, object_hash, read_json, runtime,
                     save, space, write_rows)


def make_problems(records, cfg):
    if len(records) != cfg['expected_problems']:
        raise ValueError(f'Expected {cfg["expected_problems"]} GSM8K train problems, got {len(records)}')
    result = []
    for pid, row in enumerate(records):
        if not row['question'].strip() or parse_answer(row['answer'])['status'] != 'ok':
            raise ValueError(f'Invalid GSM8K record {pid}')
        result.append({'id': f'gsm8k-train-{pid:05d}', 'pid': pid, 'question': row['question'],
                       'gold': row['answer'], 'response': row['answer']})
    assign_splits(result, cfg['seed'], .8)
    for row in result:
        del row['response'], row['canonical_id']
    return result


def run(cfg):
    environment(cfg)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    out = Path(cfg['run_dir'])
    with lock(out / 'prepare'):
        target = out / 'manifest.json'
        if target.exists():
            from .common import prepared
            prepared(cfg)
            print('Prepared manifest already verified; no dataset download.')
            return
        revision = HfApi().dataset_info(cfg['dataset'], revision=cfg['dataset_revision']).sha
        if revision != cfg['dataset_revision']:
            raise ValueError('Dataset revision resolved to an unexpected commit')
        data = load_dataset(cfg['dataset'], 'main', split='train', revision=revision)
        problems = make_problems(data, cfg)
        space(cfg, 20 * 2**20)
        write_rows(out / 'problems.jsonl', problems)
        save(target, {'config': cfg, 'dataset_revision': revision,
                      'problems_sha256': digest(out / 'problems.jsonl'),
                      'planned_responses': len(problems) * cfg['repeats'],
                      'runtime': runtime(), 'scope': 'post-generation AI-label separability, not causal diagnosis'})
        save(out / 'splits.json', {str(p['pid']): p['partition'] for p in problems})
        print(f'Prepared {len(problems)} problems / {len(problems) * cfg["repeats"]} responses; dataset SHA {revision}')
