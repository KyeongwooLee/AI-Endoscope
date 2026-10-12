"""Frozen entropy candidates on project-unseen GSM8K train problems."""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from datasets import load_dataset

import train_hidden_probes as hp
from gsm8k_lens.answers import label_answer
from gsm8k_lens.engine import generate_example, load_model, environment
from gsm8k_lens.storage import digest

ROOT = Path('/tmp/gsm8k-confirmation-v1')
SCORES = ('layer10_entropy', 'layer7_entropy', 'layer7_entropy_change', 'combo_linear', 'combo_mlp', 'raw_initial_mlp')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(partial, path)


def candidate_values(entropy7, entropy10):
    if len(entropy7) != 33 or len(entropy10) != 33:
        raise ValueError('Primary features require exactly prediction positions 0..32')
    return np.array([np.mean(entropy10), np.mean(entropy7), np.mean(entropy7) - entropy7[0]], dtype=np.float32)


def refit(x, y, kind, learning_rate, decay, seed, epochs):
    torch.manual_seed(seed)
    model = hp.build_model(x.shape[1], kind).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=decay)
    labels = torch.tensor(y, dtype=torch.float32, device='cuda').reshape(-1, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum()), device='cuda'))
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    return model.cpu()


def transform(scaler, x):
    return torch.tensor(np.clip(scaler.transform(x), -10, 10), dtype=torch.float32, device='cuda')


def freeze_models():
    source = ROOT / 'input'
    destination = ROOT / 'frozen'
    if (destination / 'complete.json').exists():
        manifest = json.loads((destination / 'complete.json').read_text())
        assert all(sha(destination / name) == checksum for name, checksum in manifest['files'].items())
        return
    destination.mkdir(exist_ok=True)
    data = np.load(source / 'train_candidates.npz', allow_pickle=False)
    ids, y, raw = [data[key] for key in ('ids', 'y', 'features')]
    splits = json.loads((source / 'original_inner_split.json').read_text())
    fit_set, tune_set = set(splits['fit_ids']), set(splits['tune_ids'])
    fit = np.array([i in fit_set for i in ids])
    tune = np.array([i in tune_set for i in ids])
    assert len(ids) == 799 and fit.sum() == 639 and tune.sum() == 160
    assert (fit | tune).all() and not (fit & tune).any()
    inner_scaler = StandardScaler().fit(raw[fit])
    scaler = StandardScaler().fit(raw)
    x, vx, full_x = transform(inner_scaler, raw[fit]), transform(inner_scaler, raw[tune]), transform(scaler, raw)
    joblib.dump(scaler, destination / 'combo_scaler.joblib')
    selections = {}
    torch.use_deterministic_algorithms(True)
    for kind in ('linear', 'mlp'):
        grid = [(.003, .1), (.003, 1.)] if kind == 'linear' else [(.0003, .1), (.001, .1)]
        best = -np.inf
        for lr, decay in grid:
            pairs = [hp.fit_probe(x, y[fit], vx, y[tune], kind, lr, decay, seed) for seed in hp.SEEDS]
            score = np.mean([p[2] for p in pairs], axis=0)
            auc = roc_auc_score(y[tune], score)
            if auc > best:
                best, selected = auc, (lr, decay, [p[1] for p in pairs])
            del pairs
        lr, decay, epochs = selected
        models = [refit(full_x, y, kind, lr, decay, seed, epoch) for seed, epoch in zip(hp.SEEDS, epochs)]
        torch.save([m.state_dict() for m in models], destination / f'combo_{kind}.pt')
        selections[kind] = {'learning_rate': lr, 'weight_decay': decay, 'epochs': epochs, 'inner_auroc': float(best)}
    for name in ('raw_initial_mlp.pt', 'raw_initial_scaler.joblib'):
        shutil.copyfile(source / name, destination / name)
    write_json(destination / 'selections.json', selections)
    write_json(destination / 'complete.json', {'files': {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != 'complete.json'},
        'train_sha256': sha(source / 'train_candidates.npz')})
    print('MODELS_FROZEN_BEFORE_NEW_DATA', flush=True)


def select_questions(config, protocol):
    path = ROOT / 'questions.json'
    if path.exists():
        return json.loads(path.read_text())
    old = json.loads((ROOT / 'input/excluded.json').read_text())
    excluded = set(old['ids'])
    used_hashes = set(old['question_hashes'])
    dataset = load_dataset(config['dataset']['id'], config['dataset']['name'],
                           revision=config['dataset']['revision'], split='train')
    assert len(dataset) == 7473
    order = list(range(len(dataset)))
    random.Random(protocol['sampling_seed']).shuffle(order)
    rows = []
    for index in order:
        identifier = f'train-{index:05d}'
        row = dataset[index]
        question_hash = hashlib.sha256(' '.join(row['question'].split()).encode()).hexdigest()
        if identifier in excluded or question_hash in used_hashes:
            continue
        rows.append({'id': identifier, 'index': index, 'question': row['question'], 'gold': row['answer'], 'question_sha256': question_hash})
        used_hashes.add(question_hash)
        if len(rows) == protocol['sample_count']:
            break
    assert len(rows) == protocol['sample_count']
    assert not set(row['id'] for row in rows) & excluded
    write_json(path, rows)
    write_json(ROOT / 'sample_ids.json', [row['id'] for row in rows])
    return rows


def project_entropy(model, states):
    output = []
    for start in range(0, len(states), 16):
        logits = model.lm_head(model.model.norm(states[start:start + 16])).float()
        logp = logits.log_softmax(-1)
        output.extend((-(logp.exp() * logp).sum(-1)).cpu().tolist())
    return output


def load_probes():
    destination = ROOT / 'frozen'
    models = {}
    for kind in ('linear', 'mlp'):
        states = torch.load(destination / f'combo_{kind}.pt', map_location='cpu', weights_only=True)
        models['combo_' + kind] = []
        for state in states:
            model = hp.build_model(3, kind).cuda()
            model.load_state_dict(state)
            model.eval()
            models['combo_' + kind].append(model)
    models['raw_initial_mlp'] = []
    states = torch.load(destination / 'raw_initial_mlp.pt', map_location='cpu', weights_only=True)
    for state in states:
        model = hp.build_model(1536, 'mlp').cuda()
        model.load_state_dict(state)
        model.eval()
        models['raw_initial_mlp'].append(model)
    return models, joblib.load(destination / 'combo_scaler.joblib'), joblib.load(destination / 'raw_initial_scaler.joblib')


def capture_generation(model, tokenizer, question, config, check_quality):
    states, initial = {7: [], 10: [], 28: []}, []
    handles = []

    def capture(layer):
        def hook(module, arguments, result):
            value = result[0] if isinstance(result, tuple) else result
            if layer == 14:
                if not initial:
                    initial.append(value[0, -1].detach().clone())
            elif len(states[layer]) < 33:
                states[layer].append(value[0, -1].detach().clone())
        return hook

    for layer in ((7, 10, 14, 28) if check_quality else (7, 10, 14)):
        handles.append(model.model.layers[layer - 1].register_forward_hook(capture(layer)))
    try:
        record = generate_example(model, tokenizer, question, config)
    finally:
        for handle in handles:
            handle.remove()
    count = min(33, len(record['generated_token_ids']))
    assert len(states[7]) == len(states[10]) == count and len(initial) == 1
    entropy7, entropy10 = project_entropy(model, torch.stack(states[7])), project_entropy(model, torch.stack(states[10]))
    quality = None
    if check_quality:
        raw_final = torch.stack(states[28])
        predicted = model.lm_head(model.model.norm(raw_final)).argmax(-1).cpu().tolist()
        actual = record['generated_token_ids'][:count]
        matched = sum(a == b for a, b in zip(predicted, actual))
        # Independent full-prefix replay checks cached-generation hooks against saved-feature semantics.
        replayed = {}

        def replay_hook(layer):
            def hook(module, arguments, result):
                value = result[0] if isinstance(result, tuple) else result
                start = len(record['prompt_token_ids']) - 1
                replayed[layer] = value[0, start:start + count].detach().clone()
            return hook

        hooks = [model.model.layers[layer - 1].register_forward_hook(replay_hook(layer)) for layer in (7, 10, 14, 28)]
        try:
            ids = record['prompt_token_ids'] + record['generated_token_ids'][:count - 1]
            model.model(input_ids=torch.tensor([ids], device='cuda'), use_cache=False)
        finally:
            for handle in hooks:
                handle.remove()
        errors = []
        for layer, original in ((7, entropy7), (10, entropy10)):
            errors.extend(np.abs(np.array(project_entropy(model, replayed[layer])) - original).tolist())
        raw_error = float((replayed[14][0] - initial[0]).abs().max())
        assert torch.allclose(replayed[14][0], initial[0], atol=3e-4, rtol=3e-4)
        assert max(errors) < .001
        replay_top1 = model.lm_head(model.model.norm(replayed[28])).argmax(-1).cpu().tolist()
        replay_matches = sum(a == b for a, b in zip(replay_top1, actual))
        quality = {'cached_argmax_matches': matched, 'replay_argmax_matches': replay_matches,
                   'positions': count, 'entropy_max_abs_error': max(errors), 'raw_initial_max_abs_error': raw_error}
    return record, entropy7, entropy10, initial[0].cpu().numpy(), quality


def make_batch(batch, entries):
    archive_path = ROOT / 'outbox' / f'batch_{batch:04d}.zip'
    receipt_path = ROOT / 'outbox' / f'batch_{batch:04d}.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert sha(archive_path) == receipt['archive_sha256']
        return
    archive_path.parent.mkdir(exist_ok=True)
    partial = Path(str(archive_path) + '.partial')
    files = {}
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            path = ROOT / 'records' / (entry['id'] + '.json')
            name = str(path.relative_to(ROOT))
            files[name] = sha(path)
            archive.write(path, name)
    os.replace(partial, archive_path)
    write_json(receipt_path, {'archive_sha256': sha(archive_path), 'archive_bytes': archive_path.stat().st_size,
        'files': files, 'ids': [entry['id'] for entry in entries]})


def check_gate(ledger, rows, fingerprint):
    if not all(row['id'] in ledger for row in rows[:8]):
        return
    checks = [ledger[row['id']]['quality'] for row in rows[:8]]
    count = sum(q['positions'] for q in checks)
    assert sum(q['cached_argmax_matches'] for q in checks) / count >= .99
    assert sum(q['replay_argmax_matches'] for q in checks) / count >= .99
    write_json(ROOT / 'quality_gate.json', {'passed': True, 'fingerprint': fingerprint, 'checks': checks,
        'replay_agreement': sum(q['replay_argmax_matches'] for q in checks) / count})


def evaluate(ledger, protocol):
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    frame = pd.DataFrame(ledger.values())
    frame.to_parquet(output / 'predictions.parquet', index=False)
    common = frame[frame.eligible_t32].copy()
    y = common.incorrect.to_numpy(dtype=int)
    rows = []
    for name in SCORES:
        score = common[name].to_numpy(dtype=float)
        indexes = np.argsort(-score, kind='stable')[:int(np.ceil(len(score) * .2))]
        rows.append({'score': name, 'n': len(y), 'coverage': len(y) / len(frame),
            'error_prevalence': float(y.mean()), 'auroc': float(roc_auc_score(y, score)),
            'auroc_ci95': hp.bootstrap_auc(y, score),
            'ap': float(average_precision_score(y, score)),
            'top20_precision': float(y[indexes].mean()), 'top20_recall': float(y[indexes].sum() / y.sum())})
    pairs = {name + '_vs_raw_initial': hp.base.paired_bootstrap(y, common.raw_initial_mlp.to_numpy(), common[name].to_numpy())
             for name in SCORES if name != 'raw_initial_mlp'}
    write_json(output / 'results.json', rows)
    write_json(output / 'paired_comparisons.json', pairs)
    pd.DataFrame(rows).to_csv(output / 'results.csv', index=False)
    write_json(output / 'generation_summary.json', {'problems': len(frame), 'accuracy': float(1 - frame.incorrect.mean()),
        'primary_eligible': len(common), 'parse_failures': int((frame.parse_status != 'ok').sum()),
        'truncated': int(frame.truncated.sum()), 'mean_generated_tokens': float(frame.generated_tokens.mean())})
    primary = rows[0]
    report = ['# Fresh Problem Entropy Confirmation', '',
        '1,000 project-unseen GSM8K train problems; no overlap with earlier problem IDs or normalized question text.',
        'Primary: negative layer-10 logit-lens entropy averaged over prediction positions 0..32.',
        'Candidates/models frozen before loading the new questions. Qwen weights and generation unchanged.', '',
        '| Score | AUROC (95% CI) | AP | Top20 precision |', '|---|---|---:|---:|']
    for row in rows:
        lo, hi = row['auroc_ci95']
        report.append(f'| {row["score"]} | {row["auroc"]:.3f} [{lo:.3f}, {hi:.3f}] | {row["ap"]:.3f} | {row["top20_precision"]:.3f} |')
    report += ['', 'All comparisons use the same t32-active population; early completions are separately reported.',
        'Secondary comparisons and pointwise CIs are not multiplicity-adjusted.',
        'This is held out from our experiments, not a claim of absence from Qwen pretraining.',
        'Lower intermediate logit-lens entropy is the frozen risk direction, not calibrated output confidence.',
        'No inference intervention or answer-accuracy improvement is evaluated.']
    (output / 'report.md').write_text('\n'.join(report) + '\n')
    fig, axis = plt.subplots(figsize=(11, 4.5), layout='constrained')
    mean = np.array([row['auroc'] for row in rows])
    ci = np.array([row['auroc_ci95'] for row in rows]).T
    axis.errorbar(np.arange(len(rows)), mean, yerr=[mean-ci[0], ci[1]-mean], fmt='o', capsize=4, color='#167a68')
    axis.set_xticks(np.arange(len(rows)), [row['score'] for row in rows], rotation=20, ha='right')
    axis.axhline(.5, color='#777777', linestyle='--')
    axis.set_ylabel('Fresh-problem AUROC, incorrect = positive')
    axis.set_title('Frozen candidates on 1,000 new GSM8K problems')
    axis.grid(axis='y', alpha=.2)
    fig.savefig(output / 'confirmation.png', dpi=160)
    plt.close(fig)
    write_json(output / 'complete.json', {'files': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    protocol = json.loads((ROOT / 'input/protocol.json').read_text())
    config = json.loads((ROOT / 'input/config.json').read_text())
    freeze_models()
    rows = select_questions(config, protocol)
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in list((ROOT / 'gsm8k_lens').glob('*.py'))
                     + [ROOT / 'train_hidden_probes.py', ROOT / 'train_error_classifier.py']}
    fingerprint = digest({'script': sha(__file__), 'source_hashes': source_hashes, 'protocol': protocol, 'config': config,
        'questions_sha256': sha(ROOT / 'questions.json'), 'frozen_sha256': sha(ROOT / 'frozen/complete.json')})
    manifest_path = ROOT / 'manifest.json'
    manifest = {'fingerprint': fingerprint, 'source_hashes': source_hashes, 'protocol': protocol, 'config': config,
                'input_hashes': {p.name: sha(p) for p in (ROOT / 'input').iterdir() if p.is_file()}}
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    model, tokenizer = load_model(config, config['model']['revision'])
    assert digest(tokenizer.chat_template) == protocol['chat_template_hash']
    torch.use_deterministic_algorithms(False)
    write_json(ROOT / 'environment.json', environment())
    probes, combo_scaler, raw_scaler = load_probes()
    ledger_path = ROOT / 'ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    check_gate(ledger, rows, fingerprint)
    started = time.monotonic()
    initial_done = len(ledger)
    with torch.inference_mode():
        for index, row in enumerate(rows):
            record_path = ROOT / 'records' / (row['id'] + '.json')
            if row['id'] in ledger:
                assert sha(record_path) == ledger[row['id']]['record_sha256']
                if (index + 1) % 32 == 0 or index == len(rows) - 1:
                    make_batch(index // 32, rows[index // 32 * 32:index + 1])
                continue
            if record_path.exists():
                saved = json.loads(record_path.read_text())
                assert saved['fingerprint'] == fingerprint and saved['id'] == row['id']
                entry = saved['summary']
                entry['record_sha256'] = sha(record_path)
                ledger[row['id']] = entry
                write_json(ledger_path, ledger)
                check_gate(ledger, rows, fingerprint)
                if (index + 1) % 32 == 0 or index == len(rows) - 1:
                    make_batch(index // 32, rows[index // 32 * 32:index + 1])
                continue
            if shutil.disk_usage(ROOT).free < 6 * 2**30:
                raise RuntimeError('Less than 6GiB free space; preserve records and stop')
            generation, entropy7, entropy10, raw_initial, quality = capture_generation(model, tokenizer,
                row['question'], config, check_quality=index < 8)
            label = label_answer(generation['text'], row['gold'], generation['truncated'])
            entry = {'id': row['id'], 'incorrect': int(not label['answer_correct']),
                     'eligible_t32': len(entropy7) == 33, 'parse_status': label['parse_status'],
                     'truncated': label['truncated'], 'generated_tokens': generation['generated_tokens'], 'quality': quality}
            raw_x = transform(raw_scaler, raw_initial.reshape(1, -1))
            entry['raw_initial_mlp'] = float(np.mean([hp.probability(probe, raw_x)[0] for probe in probes['raw_initial_mlp']]))
            if entry['eligible_t32']:
                features = candidate_values(entropy7, entropy10)
                for name, feature in zip(SCORES[:3], features):
                    entry[name] = float(-feature)
                cx = transform(combo_scaler, features.reshape(1, -1))
                for name in ('combo_linear', 'combo_mlp'):
                    entry[name] = float(np.mean([hp.probability(probe, cx)[0] for probe in probes[name]]))
            else:
                for name in SCORES[:-1]:
                    entry[name] = None
            write_json(record_path, {'id': row['id'], 'fingerprint': fingerprint, 'generation': generation,
                'label': label, 'entropy7_prefix': entropy7, 'entropy10_prefix': entropy10,
                'raw_initial_h14': raw_initial.tolist(), 'summary': entry})
            entry['record_sha256'] = sha(record_path)
            ledger[row['id']] = entry
            write_json(ledger_path, ledger)
            if index == 7:
                check_gate(ledger, rows, fingerprint)
            if (index + 1) % 32 == 0 or index == len(rows) - 1:
                make_batch(index // 32, rows[index // 32 * 32:index + 1])
            done = len(ledger)
            elapsed = time.monotonic() - started
            remaining = elapsed / max(done - initial_done, 1) * (len(rows) - done)
            write_json(ROOT / 'progress.json', {'stage': 'generation_and_feature_capture', 'completed': done,
                'total': len(rows), 'elapsed_session_seconds': elapsed, 'remaining_seconds_estimate': remaining,
                'fingerprint': fingerprint, 'last_id': row['id'], 'peak_gpu_gib': torch.cuda.max_memory_allocated()/2**30})
            print(f'[{done}/{len(rows)}] {row["id"]} tokens={generation["generated_tokens"]}; remaining estimate {remaining/60:.1f}min', flush=True)
    assert (ROOT / 'quality_gate.json').exists()
    evaluate(ledger, protocol)
    write_json(ROOT / 'progress.json', {'stage': 'complete', 'completed': len(ledger), 'total': len(rows)})
    members = [p for name in ('input', 'frozen', 'records', 'results') for p in (ROOT / name).rglob('*') if p.is_file()]
    members += [p for p in ROOT.glob('*.json')]
    members += list(ROOT.glob('*.py')) + list((ROOT / 'gsm8k_lens').glob('*.py'))
    manifest = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in members}
    write_json(ROOT / 'backup_manifest.json', manifest)
    archive_path = ROOT / 'final-backup.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in members + [ROOT / 'backup_manifest.json']:
            archive.write(p, str(p.relative_to(ROOT)))
    write_json(ROOT / 'job_complete.json', {'archive_bytes': archive_path.stat().st_size, 'archive_sha256': sha(archive_path)})
    print('ENTROPY_CONFIRMATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
