"""Create train/validation-only depth-14 engineered features from immutable saved data."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

spec = importlib.util.spec_from_file_location('hidden', Path(__file__).with_name('train_hidden_probes.py'))
hidden = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hidden)


def slope(values):
    values = np.asarray(values, dtype=float)
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(np.dot(x, values) / np.dot(x, x)) if len(values) > 1 else 0.


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def state_features(states, trajectory):
    assert states.shape[0] == 4
    h7, h10, h13, h14 = states.astype(float)
    curve = trajectory[:14].astype(float)
    norm = np.linalg.norm(h14)
    square = h14 ** 2
    energy = square.sum()
    count = max(1, int(np.ceil(len(h14) * .01)))
    size = {'norm': norm, 'mean': h14.mean(), 'std': h14.std(),
        'rms': np.sqrt(np.mean(square)), 'abs_max': np.abs(h14).max(),
        'abs_q90': np.quantile(np.abs(h14), .9), 'abs_q99': np.quantile(np.abs(h14), .99),
        'max_energy_share': square.max() / max(energy, 1e-12),
        'top1pct_energy_share': np.sort(square)[-count:].sum() / max(energy, 1e-12),
        'participation_fraction': energy ** 2 / max(len(h14) * np.square(square).sum(), 1e-12)}
    update = h14 - h13
    adjacent = {'delta_norm': np.linalg.norm(update),
        'relative_delta': np.linalg.norm(update) / max(np.linalg.norm(h13), 1e-12),
        'cosine_distance': 1 - cosine(h13, h14),
        'norm_ratio': norm / max(np.linalg.norm(h13), 1e-12),
        'update_alignment_previous': cosine(update, h13),
        'update_alignment_current': cosine(update, h14)}
    flow = {}
    for name, before, after, steps in [('7_10', h7, h10, 3), ('10_14', h10, h14, 4), ('7_14', h7, h14, 7)]:
        change = np.linalg.norm(after - before)
        flow.update({name + '_delta_per_layer': change / steps,
            name + '_relative_delta': change / max(np.linalg.norm(before), 1e-12),
            name + '_cosine_distance': 1 - cosine(before, after),
            name + '_norm_ratio': np.linalg.norm(after) / max(np.linalg.norm(before), 1e-12)})
    flow['update_direction_7_10_vs_10_14'] = cosine(h10 - h7, h14 - h10)
    acceleration = {}
    # Saved trajectory columns: previous norm, hidden norm, absolute/relative delta, cosine distance.
    for index, name in [(2, 'absolute'), (3, 'relative'), (4, 'direction')]:
        history = curve[1:14, index]
        acceleration.update({name + '_slope_2_14': slope(history),
            name + '_slope_11_14': slope(history[-4:]),
            name + '_change_13_14': history[-1] - history[-2],
            name + '_ratio_14_13': history[-1] / max(history[-2], 1e-12),
            name + '_mean_second_difference': np.diff(history, n=2).mean(),
            name + '_decreasing_step_fraction': (np.diff(history) < 0).mean()})
    return {'size': size, 'adjacent': adjacent, 'flow': flow, 'acceleration': acceleration}


def readout_features(frame, checkpoint):
    frame = frame[(frame.token_index <= checkpoint) & frame.layer.isin([7, 10, 14])]
    result = {}
    for layer in (7, 10, 14):
        group = frame[frame.layer == layer].sort_values('token_index')
        assert group.token_index.tolist() == list(range(checkpoint + 1))
        for metric in ('entropy', 'margin'):
            values = group[metric].to_numpy(dtype=float)
            name = f'L{layer}_{metric}'
            result.update({name + '_current': values[-1], name + '_prefix_mean': values.mean(),
                name + '_prefix_std': values.std(), name + '_last8_mean': values[-8:].mean(),
                name + '_last8_std': values[-8:].std(), name + '_last8_slope': slope(values[-8:]),
                name + '_change_from_initial': values[-1] - values[0]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    saved = hidden.load_phase(Path(args.hidden), 'exploration')
    config = json.loads((Path(args.hidden) / 'config.json').read_text())
    assert config['stored_layers'] == [7, 10, 13, 14]
    # Full-depth scalar trajectories are sliced to the first 14 blocks immediately.
    trajectories = {}
    complete = json.loads((Path(args.hidden) / 'complete.json').read_text())
    for entry in complete['batches']:
        with np.load(Path(args.hidden) / entry['file'], allow_pickle=False) as batch:
            mask = (batch['phases'] == 'exploration') & np.isin(batch['checkpoints'], [0, 32])
            for identifier, t, curve in zip(batch['ids'][mask], batch['checkpoints'][mask], batch['trajectory'][mask, :14]):
                trajectories[identifier, int(t)] = curve
    files = sorted((Path(args.run) / 'features/exploration').glob('*/token_features.parquet'))
    dataset = ds.dataset([str(p) for p in files], format='parquet')
    readouts = dataset.to_table(columns=['id', 'layer', 'token_index', 'entropy', 'margin'],
        filter=(ds.field('token_index') <= 32) & ds.field('layer').isin([7, 10, 14])).to_pandas()
    by_id = dict(tuple(readouts.groupby('id', sort=False)))
    indexes = {(str(identifier), int(t)): index for index, (identifier, t) in enumerate(zip(saved['ids'], saved['checkpoints']))}
    ids = sorted(identifier for identifier, t in indexes if t == 32)
    records, raw_initial, raw_current, target = [], [], [], []
    groups = {key: [] for key in ('size', 'adjacent', 'flow', 'acceleration', 'stability')}
    initial_fields, difference_fields = [], []
    for identifier in ids:
        values = {'id': identifier}
        initial, current = {}, {}
        for t, prefix, mapping in ((0, 'initial', initial), (32, 'current', current)):
            index = indexes[identifier, t]
            vector = saved['hidden'][index]
            blocks = state_features(vector, trajectories[identifier, t])
            blocks['stability'] = readout_features(by_id[identifier], t)
            for group, fields in blocks.items():
                for name, value in fields.items():
                    field = f'{group}_{name}'
                    mapping[field] = float(value)
                    full = f'{prefix}_{field}'
                    values[full] = float(value)
                    if len(records) == 0:
                        if t == 32:
                            groups[group].append(full)
                        elif group != 'stability' or name.endswith('_current'):
                            initial_fields.append(full)
        for field in initial:
            full = f'change_{field}'
            values[full] = current[field] - initial[field]
            if len(records) == 0:
                difference_fields.append(full)
        records.append(values)
        first_index, current_index = indexes[identifier, 0], indexes[identifier, 32]
        assert saved['y'][first_index] == saved['y'][current_index]
        target.append(saved['y'][current_index])
        raw_initial.append(saved['hidden'][first_index, 3])
        raw_current.append(saved['hidden'][current_index, 3])
    frame = pd.DataFrame(records).set_index('id')
    assert np.isfinite(frame.to_numpy()).all()
    structure = [field for group in ('size', 'adjacent', 'flow', 'acceleration') for field in groups[group]]
    current = structure + groups['stability']
    groups.update(current_structure=structure, current_all=current,
        initial_all=initial_fields, initial_plus_current=initial_fields + current,
        initial_plus_changes=initial_fields + difference_fields)
    raw_initial, raw_current = np.stack(raw_initial), np.stack(raw_current)
    matrices = {name: frame[fields].to_numpy(dtype=np.float32) for name, fields in groups.items()}
    matrices.update(raw_initial=raw_initial, raw_current=raw_current,
                    raw_initial_plus_current=np.concatenate([raw_initial, raw_current], axis=1))
    splits = json.loads(Path(args.splits).read_text())
    train_set, valid_set = set(splits['train_all']), set(splits['validation_all'])
    train = np.array([i in train_set for i in ids])
    valid = np.array([i in valid_set for i in ids])
    assert (train | valid).all() and not (train & valid).any()
    np.savez_compressed(output / 'designs.npz', ids=np.asarray(ids), y=np.asarray(target), train=train, valid=valid, **matrices)
    frame.to_parquet(output / 'engineered_features.parquet')
    hidden.base.write_json(output / 'groups.json', groups)
    hidden.base.write_json(output / 'provenance.json', {'primary_checkpoint': 32,
        'train_n': int(train.sum()), 'validation_n': int(valid.sum()), 'test_loaded': False,
        'source_sha256': hidden.base.sha(Path(__file__)), 'split_sha256': hidden.base.sha(Path(args.splits)),
        'hidden_config_sha256': hidden.base.sha(Path(args.hidden) / 'config.json'),
        'hidden_complete_sha256': hidden.base.sha(Path(args.hidden) / 'complete.json'),
        'readout_columns': ['id', 'layer', 'token_index', 'entropy', 'margin'], 'max_layer': 14,
        'temporal_change_warning': 't0 and t32 are different prediction positions and contexts, not the same token representation.',
        'group_dimensions': {name: matrix.shape[1] for name, matrix in matrices.items()}})
    print(f'Prepared {len(ids)} exploration problems, {len(matrices)} feature groups; test untouched', flush=True)


if __name__ == '__main__':
    main()
