"""Resolve exact evidence locations in code. Never edit explanation or normalize text."""
from copy import deepcopy

def attach_unique_quote_offsets(output, response):
    fixed = deepcopy(output)
    quote = fixed.get('evidence_quote')
    if quote is None and fixed.get('status') != 'supported_error':
        fixed.update(evidence_start=None, evidence_end=None, evidence_occurrence=None)
        return fixed
    if not isinstance(quote, str) or not quote:
        raise ValueError('A non-empty exact quote is required')
    # Include overlapping occurrences: repeated or ambiguous quotes must be rejected.
    positions = [i for i in range(len(response)) if response.startswith(quote, i)]
    if len(positions) != 1:
        raise ValueError(f'Expected one exact occurrence, found {len(positions)}; do not guess or normalize')
    start = positions[0]
    fixed.update(evidence_start=start, evidence_end=start + len(quote), evidence_occurrence=1)
    assert response[fixed['evidence_start']:fixed['evidence_end']] == quote
    return fixed

if __name__ == '__main__':
    text = '앞부분 🙂\n80 - 16 = 40\n끝'
    v = {'status': 'supported_error', 'evidence_quote': '80 - 16 = 40', 'error_description': 'unchanged', 'evidence_start': 999}
    x = attach_unique_quote_offsets(v, text)
    assert text[x['evidence_start']:x['evidence_end']] == v['evidence_quote']
    assert x['error_description'] == 'unchanged' and v['evidence_start'] == 999
    for response, quote in [('a a', 'a'), ('aaa', 'aa'), ('x - y', 'x−y'), ('abc', '')]:
        try:
            attach_unique_quote_offsets({'status':'supported_error','evidence_quote':quote}, response)
        except ValueError:
            pass
        else:
            raise AssertionError('Ambiguous, changed, or empty quote was accepted')
    print('Exact-quote checks passed: Unicode offsets, immutable input, duplicates including overlaps, no normalization, missing quote.')
