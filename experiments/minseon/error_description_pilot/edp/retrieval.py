"""Fixed query -> gallery top-k retrieval on normalized dense embeddings (no PCA/clustering)."""
import numpy as np

from .io import ProtocolError


def rank_gallery(query_vec, gallery_ids, gallery_vecs):
    """Cosine (dot of L2-normalized vectors); ties broken by stable source ID."""
    if len(gallery_ids) == 0:
        return []
    scores = gallery_vecs @ query_vec
    pairs = sorted(zip(gallery_ids, scores.tolist()), key=lambda p: (-p[1], p[0]))
    return [{'id': gid, 'score': float(s), 'rank': i + 1} for i, (gid, s) in enumerate(pairs)]


def random_ranking(gallery_ids, seed, query_index):
    order = sorted(gallery_ids)
    perm = np.random.default_rng([seed, 7, query_index]).permutation(len(order))
    return [{'id': order[j], 'score': None, 'rank': i + 1} for i, j in enumerate(perm)]


def check_roles(queries, gallery):
    qq = {u['question_sha256'] for u in queries} | {u['source_pid'] for u in queries}
    for g in gallery:
        if g['question_sha256'] in qq or g['source_pid'] in qq:
            raise ProtocolError(f'Gallery item {g["source_id"]} shares a question with a query')
    if {u['response_sha256'] for u in queries} & {g['response_sha256'] for g in gallery}:
        raise ProtocolError('A query response is duplicated in the gallery')


def retrieve(method, queries, gallery, vectors_by_id, k, seed):
    """vectors_by_id: source_id -> vector for usable (non-abstaining) items; None for Random."""
    check_roles(queries, gallery)
    gallery_ids_all = [g['source_id'] for g in gallery]
    results = []
    for qi, q in enumerate(queries):
        if method == 'Random':
            ranking = random_ranking(gallery_ids_all, seed, qi)
            status, size = 'ranked', len(gallery_ids_all)
        else:
            usable = [gid for gid in gallery_ids_all if gid in vectors_by_id]
            size = len(usable)
            if q['source_id'] not in vectors_by_id:
                ranking, status = [], 'query_abstained_or_missing'
            else:
                mat = np.stack([vectors_by_id[g] for g in usable]) if usable else np.zeros((0, 1))
                ranking = rank_gallery(vectors_by_id[q['source_id']], usable, mat)
                status = 'ranked' if ranking else 'empty_gallery'
        results.append({'method': method, 'query_index': qi, 'query_id': q['source_id'], 'status': status,
                        'usable_gallery_size': size, 'gallery_size': len(gallery_ids_all),
                        'top_k': ranking[:k], 'ranking': ranking})
    return results
