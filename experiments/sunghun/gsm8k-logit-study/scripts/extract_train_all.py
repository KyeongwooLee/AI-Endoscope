"""Compact, resumable feature extraction for the official GSM8K train split only."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

from datasets import load_dataset
import numpy as np
import torch

from gsm8k_lens.answers import label_answer
from gsm8k_lens.engine import environment, generate_example, load_model
from gsm8k_lens.storage import digest

CHECKPOINTS = (0, 32, 64, 128)
RAW_LAYERS = (7, 10, 13, 14)
READOUT_LAYERS = (7, 10, 14)
READOUT_METRICS = ('entropy', 'margin', 'top1_probability')
TRAJECTORY_METRICS = ('previous_norm', 'hidden_norm', 'delta_norm', 'relative_delta', 'cosine_distance')
BATCH_SIZE = 32
CHAT_HASH = '2e2d2512cfe46af53dc1eed45368ecaab26ac4461c480e6b69fe571cf0ceaa75'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path) + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def checkpoint_times(length):
    if length < 1:
        raise ValueError('An empty generation cannot supply prediction positions')
    return [t for t in CHECKPOINTS if t < length]


def residual_metrics(states):
    before, after = states[:-1], states[1:]
    previous, current = before.norm(dim=-1), after.norm(dim=-1)
    delta = (after - before).norm(dim=-1)
    cosine = 1 - ((before * after).sum(-1) / (previous * current).clamp_min(1e-12)).clamp(-1, 1)
    return torch.stack((previous, current, delta, delta / previous.clamp_min(1e-12), cosine), -1)


def readout(model, states):
    pieces = []
    for start in range(0, len(states), 16):
        logits = model.lm_head(model.model.norm(states[start:start + 16])).float()
        logp = logits.log_softmax(-1)
        top = logp.topk(2, dim=-1).values.exp()
        pieces.append(torch.stack((-(logp.exp() * logp).sum(-1), top[:, 0] - top[:, 1], top[:, 0]), -1))
    return torch.cat(pieces).cpu().numpy()


@torch.inference_mode()
def extract(model, generation, causal_check=False):
    prompt, generated = generation['prompt_token_ids'], generation['generated_token_ids']
    times = checkpoint_times(len(generated))
    count = min(129, len(generated))
    positions = torch.arange(len(prompt) - 1, len(prompt) - 1 + count, device=model.device)
    captured, curves = {}, {}

    def hook(layer):
        def collect(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            selected = value[0, positions if layer in RAW_LAYERS else positions[times]].detach().clone()
            if layer in RAW_LAYERS:
                captured[layer] = selected
                curves[layer] = selected[times]
            else:
                curves[layer] = selected
        return collect

    handles = [model.model.embed_tokens.register_forward_hook(hook(0))]
    handles += [block.register_forward_hook(hook(layer)) for layer, block in enumerate(model.model.layers, 1)]
    try:
        ids = prompt + generated[:count - 1]
        model.model(input_ids=torch.tensor([ids], device=model.device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    assert sorted(curves) == list(range(29))
    states = torch.stack([curves[layer] for layer in range(29)])
    assert torch.isfinite(states).all()
    hidden = torch.stack([captured[layer][times] for layer in RAW_LAYERS], 1).cpu().numpy()
    trajectory = residual_metrics(states).permute(1, 0, 2).cpu().numpy()
    readouts = np.stack([readout(model, captured[layer]) for layer in READOUT_LAYERS], 1)
    assert np.isfinite(hidden).all() and np.isfinite(trajectory).all() and np.isfinite(readouts).all()
    predicted = model.lm_head(model.model.norm(states[-1])).argmax(-1).cpu().tolist()
    quality = {'positions': len(times), 'matches': sum(a == generated[t] for a, t in zip(predicted, times)),
               'causal_max_abs_error': None}
    if causal_check and 32 in times and count > 33:
        reference = captured[14][32]
        replayed = []

        def short_hook(module, inputs, result):
            value = result[0] if isinstance(result, tuple) else result
            replayed.append(value[0, -1].detach().clone())

        handle = model.model.layers[13].register_forward_hook(short_hook)
        try:
            model.model(input_ids=torch.tensor([prompt + generated[:32]], device=model.device), use_cache=False)
        finally:
            handle.remove()
        quality['causal_max_abs_error'] = float((reference - replayed[0]).abs().max())
        if not torch.allclose(reference, replayed[0], atol=3e-4, rtol=3e-4):
            raise ValueError('Prefix causality gate failed')
    return {'checkpoints': np.asarray(times, dtype=np.int16), 'hidden': hidden,
            'trajectory': trajectory, 'readouts': readouts}, quality


def verify_prompt(tokenizer, generation, question, config):
    messages = [{'role': 'system', 'content': config['prompt']['system']},
                {'role': 'user', 'content': config['prompt']['user_template'].replace('{question}', question)}]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if generation['prompt_token_ids'] != expected:
        raise ValueError('Imported generation has a different prompt')
    if generation['generated_tokens'] != len(generation['generated_token_ids']):
        raise ValueError('Imported generation token count mismatch')
    text = tokenizer.decode(generation['generated_token_ids'], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    if text != generation['text']:
        raise ValueError('Imported generation text does not match its original token IDs')


def select_train(root, config, known):
    path = root / 'questions.json'
    if path.exists():
        rows = json.loads(path.read_text())
    else:
        dataset = load_dataset(config['dataset']['id'], config['dataset']['name'],
                               revision=config['dataset']['revision'], split='train')
        if len(dataset) != 7473:
            raise ValueError('Unexpected official train size')
        order = list(range(7473))
        random.Random(20261009).shuffle(order)
        rows = [{'id': f'train-{i:05d}', 'index': i, 'question': dataset[i]['question'],
                 'gold': dataset[i]['answer']} for i in order]
        write_json(path, rows)
    if len(rows) != 7473 or {r['id'] for r in rows} != {f'train-{i:05d}' for i in range(7473)}:
        raise ValueError('Must contain every official train ID exactly once')
    for row in rows:
        if row['id'] in known:
            if known[row['id']]['question'] != row['question'] or known[row['id']]['gold'] != row['gold']:
                raise ValueError('Imported question or gold differs from pinned dataset')
    if not set(known).issubset({r['id'] for r in rows}):
        raise ValueError('Imported data includes non-train IDs')
    return rows


def gate(summaries):
    positions = sum(row['quality']['positions'] for row in summaries)
    matches = sum(row['quality']['matches'] for row in summaries)
    if positions == 0 or matches / positions < .99:
        raise ValueError(f'Generation replay gate failed: {matches}/{positions}')
    if not any(row['quality']['causal_max_abs_error'] is not None for row in summaries):
        raise ValueError('No prefix causality check was performed')
    return {'passed': True, 'agreement': matches / positions, 'positions': positions,
            'checks': [{'id': row['id'], 'origin': row['origin'], **row['quality']} for row in summaries]}


def pack_batch(root, number, saved, fingerprint):
    name = f'batch_{number:04d}'
    output = root / 'outbox' / (name + '.zip')
    hidden, trajectory, checkpoints, ids, readouts, records, summaries = [], [], [], [], [], [], []
    offsets = [0]
    for path in saved:
        with np.load(path, allow_pickle=False) as value:
            record = json.loads(str(value['record']))
            assert record['fingerprint'] == fingerprint
            records.append(record)
            summaries.append(record['summary'])
            times = value['checkpoints']
            ids.extend([record['id']] * len(times))
            checkpoints.append(times)
            hidden.append(value['hidden'])
            trajectory.append(value['trajectory'])
            readouts.append(value['readouts'])
            offsets.append(offsets[-1] + len(value['readouts']))
    array = io.BytesIO()
    np.savez_compressed(array, ids=np.asarray(ids), checkpoints=np.concatenate(checkpoints),
                        hidden=np.concatenate(hidden), trajectory=np.concatenate(trajectory),
                        readout_ids=np.asarray([r['id'] for r in records]),
                        readout_offsets=np.asarray(offsets, dtype=np.int64), readouts=np.concatenate(readouts))
    contents = {name + '/features.npz': array.getvalue(),
                name + '/generations.jsonl': ('\n'.join(json.dumps(r, allow_nan=False) for r in records) + '\n').encode(),
                name + '/summary.json': json.dumps(summaries, allow_nan=False).encode()}
    partial = Path(str(output) + '.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_STORED) as archive:
        for member, content in contents.items():
            archive.writestr(member, content)
    os.replace(partial, output)
    receipt = {'fingerprint': fingerprint, 'archive_sha256': sha(output),
               'archive_bytes': output.stat().st_size,
               'files': {k: hashlib.sha256(v).hexdigest() for k, v in contents.items()},
               'ids': [r['id'] for r in records], 'summaries': summaries}
    write_json(root / 'outbox' / (name + '.json'), receipt)
    return receipt


def validate_committed(root, name, receipt, fingerprint, expected_ids):
    if receipt['fingerprint'] != fingerprint or receipt['ids'] != expected_ids:
        raise ValueError('Committed batch differs from this experiment')
    archive = root / 'outbox' / (name + '.zip')
    if archive.exists():
        if sha(archive) != receipt['archive_sha256']:
            raise ValueError('Batch archive hash mismatch')
    else:
        acknowledgement = root / 'acks' / (name + '.json')
        if not acknowledgement.exists() or json.loads(acknowledgement.read_text()) != {
                'archive_sha256': receipt['archive_sha256'], 'archive_bytes': receipt['archive_bytes']}:
            raise ValueError('Missing archive without a verified local backup acknowledgement')


def final_summary(root, summaries):
    incorrect = sum(row['incorrect'] for row in summaries)
    result = {'problems': len(summaries), 'correct': len(summaries) - incorrect, 'incorrect': incorrect,
              'accuracy': 1 - incorrect / len(summaries), 'official_test_loaded': False,
              'parse_failures': sum(r['parse_status'] != 'ok' for r in summaries),
              'truncated': sum(r['truncated'] for r in summaries),
              'eligible': {str(t): sum(r['generated_tokens'] > t for r in summaries) for t in CHECKPOINTS},
              'by_origin': {}}
    for origin in sorted({r['origin'] for r in summaries}):
        subset = [r for r in summaries if r['origin'] == origin]
        result['by_origin'][origin] = {'problems': len(subset), 'incorrect': sum(r['incorrect'] for r in subset)}
    write_json(root / 'results/generation_summary.json', result)
    write_json(root / 'results/labels.json', summaries)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for directory in ('outbox', 'pending', 'acks', 'results'):
        (root / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'input/config.json').read_text())
    known_rows = [json.loads(line) for line in (root / 'input/known.jsonl').read_text().splitlines()]
    known = {r['id']: r for r in known_rows}
    if len(known) != len(known_rows) or len(known) != 2128:
        raise ValueError('Expected 2,128 unique historical train generations')
    rows = select_train(root, config, known)
    source_hashes = {p.name: sha(p) for p in (root / 'gsm8k_lens').glob('*.py')}
    fingerprint = digest({'script': sha(__file__), 'input': sha(root / 'input/known.jsonl'),
                          'config': config, 'questions': sha(root / 'questions.json'), 'sources': source_hashes})
    manifest = {'fingerprint': fingerprint, 'config': config, 'source_hashes': source_hashes,
                'script_sha256': sha(__file__), 'input_sha256': sha(root / 'input/known.jsonl'),
                'questions_sha256': sha(root / 'questions.json'), 'problems': 7473,
                'reuse_generations': len(known), 'new_generations': 7473 - len(known),
                'official_test_loaded': False, 'checkpoints': list(CHECKPOINTS), 'raw_layers': list(RAW_LAYERS),
                'readout_layers': list(READOUT_LAYERS), 'readout_metrics': list(READOUT_METRICS),
                'trajectory_metrics': list(TRAJECTORY_METRICS), 'trajectory_layers': list(range(1, 29)),
                'max_readout_position': 128, 'batch_size': BATCH_SIZE,
                'feature_definition': 'Raw blocks before final RMSNorm; logit lens = final RMSNorm + frozen LM head.',
                'position_definition': 't is the prediction position after t generated tokens, before token t.',
                'leakage_warning': 'Predictors use only available positions and layers <=14. Full-depth trajectories, labels, lengths and replay checks are diagnostics only.',
                'evaluation_warning': 'This pool includes earlier probe fit/tuning/evaluation data. Freeze a new split before any probe fitting; prior confirmation is no longer an untouched holdout if used for training.'}
    if (root / 'manifest.json').exists() and json.loads((root / 'manifest.json').read_text()) != manifest:
        raise ValueError('Cannot resume with modified inputs or source')
    write_json(root / 'manifest.json', manifest)
    torch.set_num_threads(4)
    model, tokenizer = load_model(config, config['model']['revision'])
    assert digest(tokenizer.chat_template) == CHAT_HASH
    write_json(root / 'environment.json', environment())
    summaries = []
    started, initial_done = time.monotonic(), None
    for batch_start in range(0, len(rows), BATCH_SIZE):
        number = batch_start // BATCH_SIZE
        name = f'batch_{number:04d}'
        batch_rows = rows[batch_start:batch_start + BATCH_SIZE]
        receipt_path = root / 'outbox' / (name + '.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            validate_committed(root, name, receipt, fingerprint, [r['id'] for r in batch_rows])
            if number == 0:
                write_json(root / 'quality_gate.json', {**gate(receipt['summaries']), 'fingerprint': fingerprint})
            summaries.extend(receipt['summaries'])
            for row in batch_rows:
                (root / 'pending' / (row['id'] + '.npz')).unlink(missing_ok=True)
            continue
        if initial_done is None:
            initial_done = len(summaries)
        saved = []
        for row in batch_rows:
            if shutil.disk_usage(root).free < 6 * 2**30:
                raise RuntimeError('Under 6GiB free space; preserving committed and pending data')
            path = root / 'pending' / (row['id'] + '.npz')
            if path.exists():
                with np.load(path, allow_pickle=False) as value:
                    existing = json.loads(str(value['record']))
                assert existing['id'] == row['id'] and existing['fingerprint'] == fingerprint
                saved.append(path)
                continue
            if row['id'] in known:
                generation = known[row['id']]['generation']
                origin = known[row['id']]['origin']
            else:
                generation = generate_example(model, tokenizer, row['question'], config)
                origin = 'new_train'
            verify_prompt(tokenizer, generation, row['question'], config)
            features, quality = extract(model, generation, causal_check=number == 0)
            label = label_answer(generation['text'], row['gold'], generation['truncated'])
            summary = {'id': row['id'], 'origin': origin, 'incorrect': int(not label['answer_correct']),
                       'parse_status': label['parse_status'], 'truncated': label['truncated'],
                       'generated_tokens': generation['generated_tokens'], 'quality': quality}
            record = {'id': row['id'], 'fingerprint': fingerprint, 'generation': generation,
                      'label': label, 'summary': summary}
            partial = Path(str(path) + '.partial')
            with partial.open('wb') as stream:
                np.savez_compressed(stream, record=np.asarray(json.dumps(record, allow_nan=False)), **features)
            os.replace(partial, path)
            saved.append(path)
            done = len(summaries) + len(saved)
            elapsed = time.monotonic() - started
            remaining = elapsed / max(done - initial_done, 1) * (len(rows) - done)
            write_json(root / 'progress.json', {'stage': 'generation_and_extraction', 'completed': done,
                       'total': len(rows), 'committed': len(summaries), 'elapsed_session_seconds': elapsed,
                       'remaining_seconds_estimate': remaining, 'last_id': row['id'],
                       'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30})
            print(f'[{done}/{len(rows)}] {row["id"]} {origin} tokens={generation["generated_tokens"]} ETA={remaining/3600:.2f}h', flush=True)
        if number == 0:
            batch_summaries = []
            for path in saved:
                with np.load(path, allow_pickle=False) as value:
                    batch_summaries.append(json.loads(str(value['record']))['summary'])
            write_json(root / 'quality_gate.json', {**gate(batch_summaries), 'fingerprint': fingerprint})
        receipt = pack_batch(root, number, saved, fingerprint)
        summaries.extend(receipt['summaries'])
        for path in saved:
            path.unlink()
    assert len(summaries) == 7473
    result = final_summary(root, summaries)
    write_json(root / 'progress.json', {'stage': 'complete', 'completed': len(summaries), 'total': len(rows)})
    members = [root / name for name in ('manifest.json', 'questions.json', 'environment.json',
               'quality_gate.json', 'progress.json', 'input/config.json', 'input/provenance.json',
               'results/generation_summary.json', 'results/labels.json', 'extract_train_all.py')]
    members += sorted((root / 'outbox').glob('batch_*.json')) + sorted((root / 'gsm8k_lens').glob('*.py'))
    backup_manifest = {p.relative_to(root).as_posix(): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in members}
    write_json(root / 'backup_manifest.json', backup_manifest)
    archive_path = root / 'metadata-backup.zip'
    partial = Path(str(archive_path) + '.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in members + [root / 'backup_manifest.json']:
            archive.write(path, path.relative_to(root).as_posix())
    os.replace(partial, archive_path)
    write_json(root / 'job_complete.json', {'archive_bytes': archive_path.stat().st_size,
               'archive_sha256': sha(archive_path), 'batches': (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE,
               'summary': result})
    print('TRAIN_ALL_EXTRACTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
