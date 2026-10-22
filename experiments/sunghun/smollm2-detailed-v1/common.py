import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + '.partial')
    partial.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(partial, path)


def pack_batch(root, name, paths, ids):
    out = root / 'outbox'
    out.mkdir(exist_ok=True)
    manifest = {p.relative_to(root).as_posix(): sha(p) for p in paths}
    archive = out / (name + '.zip')
    partial = Path(str(archive) + '.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            z.write(path, path.relative_to(root).as_posix())
    os.replace(partial, archive)
    write_json(out / (name + '.json'), dict(name=name, count=len(ids), ids=ids,
               archive_sha256=sha(archive), bytes=archive.stat().st_size, members=manifest))


def confidence_stats(logp, entropy, margin):
    """All lower/tail statistics are within this answer's explicitly selected mask."""
    import numpy as np
    names = ['logp_mean', 'logp_q10', 'logp_min', 'logp_last16_mean',
             'prob_lt_01_ratio', 'prob_lt_05_ratio', 'entropy_mean',
             'entropy_q90', 'entropy_max', 'entropy_last16_mean',
             'margin_mean', 'margin_q10', 'margin_min']
    if not len(logp):
        return names, np.full(len(names), np.nan, dtype=np.float32)
    values = [logp.mean(), np.quantile(logp, .1), logp.min(), logp[-16:].mean(),
              (logp < np.log(.1)).mean(), (logp < np.log(.5)).mean(),
              entropy.mean(), np.quantile(entropy, .9), entropy.max(), entropy[-16:].mean(),
              margin.mean(), np.quantile(margin, .1), margin.min()]
    return names, np.asarray(values, dtype=np.float32)
