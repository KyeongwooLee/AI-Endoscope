"""BGE-M3 dense embeddings with exact-input provenance and no silent truncation."""
import os
import time
from pathlib import Path

import numpy as np

from .annotation import embedding_text
from .io import ProtocolError, read_jsonl, sha256_text, write_json, write_jsonl

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')
DIM = 1024


def build_inputs(cond, units, rows_by_id, overlays=None):
    """Return list of dicts: source_id, text or None, reason. Only allowed text enters."""
    items = []
    for u in units:
        row = rows_by_id[u['source_id']]
        if cond == 'R':
            text, reason = f'Question: {row["question"]}\nSolution: {row["response"]}', None
        elif cond == 'Q':
            text, reason = row['question'], None
        elif cond in ('E', 'G'):
            rec = (overlays or {}).get(u['source_id'])
            if rec is None:
                text, reason = None, 'annotation_missing'
            else:
                text = embedding_text(rec)
                reason = None if text is not None else f'abstained_or_invalid:{rec["status"]}'
        else:
            raise ProtocolError(f'Unknown representation {cond}')
        items.append({'source_id': u['source_id'], 'condition': cond, 'text': text,
                      'text_sha256': sha256_text(text) if text is not None else None, 'not_embedded_reason': reason})
    return items


def check_vectors(vectors, n):
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape != (n, DIM):
        raise ProtocolError(f'Embedding shape {vectors.shape} != ({n}, {DIM})')
    if not np.all(np.isfinite(vectors)):
        raise ProtocolError('Non-finite embedding values')
    norms = np.linalg.norm(vectors, axis=1)
    if n and np.max(np.abs(norms - 1)) > 1e-3:
        raise ProtocolError('Embeddings are not L2-normalized')
    return vectors.astype(np.float32)


class Embedder:
    def __init__(self, cfg):
        import torch
        from sentence_transformers import SentenceTransformer
        e = cfg['embedding']
        if not e.get('revision'):
            raise ProtocolError('Embedding revision must be pinned before evaluation')
        device = e['device'] if e['device'] != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg, self.device = e, device
        self.model = SentenceTransformer(e['model'], revision=e['revision'], device=device)
        if e['precision'] != 'float32':
            raise ProtocolError('Only float32 is configured for this pilot')
        self.model = self.model.float()
        self.model.max_seq_length = e['max_tokens']
        self.tokenizer = self.model.tokenizer
        import importlib.metadata as md
        self.provenance = {
            'model': e['model'], 'revision': e['revision'], 'device': device, 'precision': e['precision'],
            'max_seq_length': self.model.max_seq_length, 'modules': [type(m).__name__ for m in self.model],
            'pooling': self.model[1].get_config_dict() if len(self.model) > 1 else None,
            'normalize_embeddings': True, 'similarity': 'cosine (dot product of L2-normalized vectors)',
            'versions': {p: md.version(p) for p in ('sentence-transformers', 'transformers', 'torch', 'numpy')},
            'tokenizer_class': type(self.tokenizer).__name__}

    def token_length(self, text):
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids'])

    def encode(self, texts):
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        vec = self.model.encode(texts, batch_size=self.cfg['batch_size'], normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return check_vectors(vec, len(texts))


def embed_condition(run_dir, phase, cond, items, embedder, cfg):
    """Embed allowed texts, caching by (model revision, precision, exact text hash)."""
    e = cfg['embedding']
    cache = Path(run_dir) / 'embeddings' / 'cache' / f'{e["revision"][:12]}_{e["precision"]}'
    cache.mkdir(parents=True, exist_ok=True)
    todo = []
    for it in items:
        if it['text'] is None:
            it['status'] = 'not_embedded'
            continue
        it['token_count'] = embedder.token_length(it['text'])
        if it['token_count'] > e['max_tokens']:
            it['status'], it['not_embedded_reason'] = 'blocked', f'exceeds_max_tokens:{it["token_count"]}'
            continue
        it['status'] = 'embedded'
        if not (cache / f'{it["text_sha256"]}.npy').exists():
            todo.append(it)
    unique = {}
    for it in todo:
        unique.setdefault(it['text_sha256'], it['text'])
    start = time.time()
    if unique:
        vectors = embedder.encode(list(unique.values()))
        for h, v in zip(unique, vectors):
            tmp = cache / f'{h}.tmp.npy'
            np.save(tmp, v)
            os.replace(tmp, cache / f'{h}.npy')
    elapsed = time.time() - start
    embedded = [it for it in items if it['status'] == 'embedded']
    vectors = np.stack([np.load(cache / f'{it["text_sha256"]}.npy') for it in embedded]) if embedded else np.zeros((0, DIM), np.float32)
    vectors = check_vectors(vectors, len(embedded))
    out = Path(run_dir) / 'embeddings' / phase
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f'{cond}.tmp.npz'
    with tmp.open('wb') as handle:
        np.savez(handle, ids=np.asarray([it['source_id'] for it in embedded]),
                 text_sha256=np.asarray([it['text_sha256'] for it in embedded]), vectors=vectors)
    os.replace(tmp, out / f'{cond}.npz')
    write_jsonl(out / f'{cond}_inputs.jsonl', items)
    return {'items': len(items), 'embedded': len(embedded), 'newly_encoded_texts': len(unique),
            'not_embedded': sum(it['status'] == 'not_embedded' for it in items),
            'blocked': sum(it['status'] == 'blocked' for it in items), 'encode_seconds': round(elapsed, 2),
            'max_token_count': max([it.get('token_count', 0) for it in items] or [0])}


def load_vectors(run_dir, phase, cond):
    path = Path(run_dir) / 'embeddings' / phase / f'{cond}.npz'
    with np.load(path) as data:
        ids, hashes, vectors = data['ids'].tolist(), data['text_sha256'].tolist(), data['vectors']
    inputs = {it['source_id']: it for it in read_jsonl(path.with_name(f'{cond}_inputs.jsonl'))}
    for sid, h in zip(ids, hashes):
        if inputs[sid]['text_sha256'] != h or inputs[sid]['status'] != 'embedded':
            raise ProtocolError(f'Vector/input provenance mismatch for {sid} in {path}')
    return ids, check_vectors(vectors, len(ids))


def write_provenance(run_dir, provenance):
    write_json(Path(run_dir) / 'embeddings' / 'provenance.json', provenance)
