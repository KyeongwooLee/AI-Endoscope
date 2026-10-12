"""Audit the new compact schema against immutable previous confirmation data."""
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
output = root / 'outputs/gsm8k-train-all-v1'
previous = root / 'outputs/entropy-confirmation-v1'
count, entropy_error, hidden_error, checkpoint_count = 0, 0., 0., 0
for path in sorted(output.glob('batch_*/features.npz')):
    with np.load(path, allow_pickle=False) as data:
        assert data['hidden'].shape[1:] == (4, 1536)
        assert data['trajectory'].shape[1:] == (28, 5)
        assert data['readouts'].shape[1:] == (3, 3)
        for key in ('hidden', 'trajectory', 'readouts'):
            assert np.isfinite(data[key]).all()
        assert data['readout_offsets'][0] == 0
        assert data['readout_offsets'][-1] == len(data['readouts'])
        np.testing.assert_allclose(np.linalg.norm(data['hidden'][:, 3], axis=-1),
                                   data['trajectory'][:, 13, 1], atol=.001, rtol=1e-5)
        checkpoint_count += len(data['ids'])
        for i, identifier in enumerate(data['readout_ids']):
            saved = previous / 'records' / (str(identifier) + '.json')
            if not saved.exists():
                continue
            record = json.loads(saved.read_text(encoding='utf-8'))
            prefix = data['readouts'][data['readout_offsets'][i]:data['readout_offsets'][i+1]]
            for layer_index, field in ((0, 'entropy7_prefix'), (1, 'entropy10_prefix')):
                values = record[field]
                error = np.max(np.abs(prefix[:len(values), layer_index, 0] - values))
                entropy_error = max(entropy_error, float(error))
                assert error < .001
            selected = (data['ids'] == identifier) & (data['checkpoints'] == 0)
            hidden = data['hidden'][selected, 3][0]
            reference = np.asarray(record['raw_initial_h14'])
            hidden_error = max(hidden_error, float(np.abs(hidden - reference).max()))
            np.testing.assert_allclose(hidden, reference, atol=3e-4, rtol=3e-4)
            count += 1
assert count > 0
result = {'passed': True, 'compared_confirmation_examples': count,
          'checked_checkpoint_rows': checkpoint_count,
          'entropy_max_abs_error': entropy_error, 'initial_h14_max_abs_error': hidden_error}
(output / 'previous_feature_agreement.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
