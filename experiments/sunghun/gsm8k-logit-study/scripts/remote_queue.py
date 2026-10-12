"""SSH-independent test producer and receipt-gated queue access."""
import argparse
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from gsm8k_lens import pipeline
from gsm8k_lens.config import load_config, manifest_for_run
from gsm8k_lens.offload import acknowledge, pack_batch, public_record
from gsm8k_lens.storage import file_digest, read_json, source_digest, write_json
import json


def queued_records(run, expected):
    records = [read_json(p) for p in sorted((run / 'offload/records').glob('*.json'))]
    for record in records:
        if record['run_id'] != expected or record['source_hash'] != source_digest():
            raise ValueError('Queue identity/source mismatch')
    return records


def next_batch(run, records, ids, phase, size):
    exported = {sid for r in records if r['phase'] == phase for sid in r['sample_ids']}
    remaining = [sid for sid in ids if sid not in exported]
    active = run / 'offload/active.json'
    if active.exists():
        plan = read_json(active)
        if plan['phase'] != phase or plan['source_hash'] != source_digest():
            raise ValueError('Active batch mismatch')
        selected = [sid for sid in plan['sample_ids'] if sid not in exported]
        if not set(selected).issubset(remaining):
            raise ValueError('Active sample mismatch')
        if selected:
            return selected
    return remaining[:size]


def produce(args, cfg, run, manifest):
    phase = 'test'
    pipeline.require_gate(run, manifest, phase)
    ids = [r['id'] for r in pipeline.rows_for_phase(run, manifest, phase)]
    status = {'state': 'running', 'phase': phase, 'expected': len(ids),
              'started_unix': time.time(), 'mode': 'detached_queue',
              'run_id': args.expected_run_id}
    def save(**values):
        status.update(values, updated_unix=time.time())
        write_json(run / 'server_job_status.json', status)
    with FileLock(str(run / '.producer.lock'), timeout=0):
        try:
            while True:
                with FileLock(str(run / '.writer.lock'), timeout=1800):
                    records = queued_records(run, args.expected_run_id)
                    exported = {s for r in records if r['phase'] == phase for s in r['sample_ids']}
                    pending_bytes = sum(r['bytes'] for r in records if r['state'] != 'offloaded')
                    selected = next_batch(run, records, ids, phase, args.batch_size)
                    save(packed=len(exported), queued_bytes=pending_bytes)
                    if not selected:
                        (run / 'offload/active.json').unlink(missing_ok=True)
                        save(state='complete', step='awaiting_local_transfer_and_analysis')
                        return
                    # Keep room for another worst-case batch; never delete unverified data.
                    usage = shutil.disk_usage(run)
                    if pending_bytes >= args.max_queue_gib * 2**30 or usage.free < 6 * 2**30:
                        save(state='waiting_for_space', step='awaiting_transfer')
                        wait = True
                    else:
                        wait = False
                        save(state='running', step='generate_extract_pack', active_ids=selected)
                        write_json(run / 'offload/active.json', {
                            'phase': phase, 'sample_ids': selected, 'source_hash': source_digest()})
                        pipeline.generate(cfg, run, phase, selected_ids=selected)
                        pipeline.label(cfg, run, phase, selected_ids=selected)
                        pipeline.extract(cfg, run, phase, selected_ids=selected)
                        record = pack_batch(cfg, run, phase, selected)
                        (run / 'offload/active.json').unlink(missing_ok=True)
                        print('QUEUED ' + json.dumps(record), flush=True)
                        save(packed=len(exported) + len(selected), active_ids=[])
                time.sleep(30 if wait else 3)
        except Exception as error:
            save(state='failed', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['produce', 'list', 'ack'], required=True)
    parser.add_argument('--config', default='configs/rtx3090.json')
    parser.add_argument('--run', required=True)
    parser.add_argument('--phase', choices=['test'], default='test')
    parser.add_argument('--expected-run-id', required=True)
    parser.add_argument('--receipt')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-queue-gib', type=float, default=8)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 128 or not 0 < args.max_queue_gib <= 8:
        parser.error('Invalid bounded queue parameters')
    run = Path(args.run).resolve()
    if file_digest(run / 'manifest.json') != args.expected_run_id:
        raise ValueError('Run identity mismatch')
    cfg = load_config(args.config)
    manifest = manifest_for_run(run, cfg)
    if manifest['source_hash'] != source_digest():
        raise ValueError('Source mismatch')
    if args.action == 'produce':
        produce(args, cfg, run, manifest)
        return
    if args.action == 'ack':
        if not args.receipt:
            parser.error('Receipt required')
        with FileLock(str(run / '.writer.lock'), timeout=1800):
            result = acknowledge(run, args.receipt)
    else:
        records = queued_records(run, args.expected_run_id)
        ready = [r for r in records if r['phase'] == args.phase and r['state'] == 'ready']
        if ready:
            result = public_record(run, ready[0])
        else:
            exported = {s for r in records if r['phase'] == args.phase for s in r['sample_ids']}
            expected = {r['id'] for r in pipeline.rows_for_phase(run, manifest, args.phase)}
            result = {'status': 'phase_complete' if expected == exported else 'waiting',
                      'phase': args.phase}
            if any(r['state'] == 'deleting' for r in records):
                raise ValueError('Interrupted cleanup requires receipt replay')
    print('OFFLOAD_RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
