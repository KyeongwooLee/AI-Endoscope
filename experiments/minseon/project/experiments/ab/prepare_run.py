#!/usr/bin/env python3
"""Prepare a new run with the pilot-validated canonical settings; preserve resumes."""
from pathlib import Path
import hashlib
from pipeline import prepare,RUN,dump,read
if not (RUN/'manifest.json').exists():
    prepare()
    dump(RUN/'config.json',read(Path(__file__).with_name('config.json')))
    files=list(Path(__file__).parent.glob('*.py'))+[RUN/'config.json',RUN/'manifest.json']
    dump(RUN/'protocol_frozen.json',dict(frozen_before_collection=True,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
elif not (RUN/'config.json').exists():
    raise RuntimeError('Manifest exists but config is missing; restore the recorded config before resuming.')
else:
    print('Existing manifest and settings retained.')

(RUN/'source_code').mkdir(exist_ok=True)
