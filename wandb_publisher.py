#!/usr/bin/env python3
"""Publish durable local telemetry in a CPU-only process; network never gates training.

Replay after a prolonged outage:
  python wandb_publisher.py --out /path/to/training/output --once
The saved W&B identity is reused and already accepted history is skipped.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time

from monitoring_io import complete_records, process_identity, read_json, utc_now, write_json


def emit(message):
    print(json.dumps(message), flush=True)


def payload(record, wandb):
    result = dict(record['metrics'])
    if record.get('publish_layers') and record.get('layers'):
        rows = record['layers']
        columns = ['layer'] + sorted(set().union(*(r.keys() for r in rows)) - {'layer'})
        result['layers/snapshot'] = wandb.Table(columns=columns,
            data=[[r.get(k) for k in columns] for r in rows], optional=True)
    return result


def submit_available(stream, run, wandb):
    last = None
    for record in complete_records(stream):
        if record['step'] < run.step:
            continue
        run.log(payload(record, wandb), step=record['step'], commit=True)
        last = record['step']
    return last


def publisher_config(directory):
    data = read_json(directory / 'config.json')
    manifest = data['manifest']
    groups = [{k: v for k, v in g.items() if k != 'parameter_names'}
              for g in manifest['optimizer_groups']]
    return dict(model=manifest['config'], parameters=manifest['parameter_count'],
        training=manifest['contract']['arguments'], environment=manifest['contract']['environment'],
        optimizer_groups=groups, source_sha256=manifest['contract']['source_sha256'],
        tokenizer=manifest['contract']['tokenizer'], monitoring=data['monitoring'],
        metric_notes='Loss is response-token-weighted within each logging window; not validation loss. '
        'Norms are measured before clipping. Update ratios are deterministic samples of actual '
        'stored updates. Layer ratios compare weight RMS, not activation contributions. '
        'Use train/step or train/tokens_seen as X axes, not upload wall time.')


def main(args):
    directory = args.directory
    if args.out:
        directory = Path(read_json(args.out / 'wandb_run.json')['directory'])
    if directory is None:
        raise ValueError('Supply --out or --directory')
    directory = directory.resolve()
    identity = read_json(directory / 'identity.json')
    mode = args.mode or identity['mode']
    if mode == 'disabled':
        return
    # Prevent two local workers from writing this one W&B stream on resume.
    lock = (directory / 'publisher.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        emit({'event': 'publisher_already_running', 'id': identity['id']})
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import wandb  # No torch/model imports; no GPU context or training callbacks.
    status = dict(pid=os.getpid(), process_start=process_identity(os.getpid()),
                  id=identity['id'], project=identity['project'], mode=mode)

    def save_status(state, **fields):
        status.update(state=state, updated_utc=utc_now(), **fields)
        write_json(directory / 'publisher_status.json', status)

    started = time.monotonic()
    ended_since = None
    retry_delay = 5
    while True:
        run = None
        session = read_json(directory / 'session.json', {})
        alive = process_identity(session.get('pid', 0)) == session.get('process_start') is not None
        ended = session.get('state') != 'running' or not alive
        if not ended:
            ended_since = None
        elif ended_since is None:
            ended_since = time.monotonic()
        if args.once and time.monotonic() - started >= args.drain_seconds:
            save_status('pending_sync', reason='Replay retry budget expired; local JSONL retained')
            return
        if ended_since is not None and time.monotonic() - ended_since >= args.drain_seconds:
            save_status('pending_sync', reason='Post-training retry budget expired; local JSONL retained')
            return
        try:
            save_status('connecting')
            run = wandb.init(entity=identity['entity'] or None, project=identity['project'],
                id=identity['id'], name=identity['name'], job_type='pretrain',
                mode=mode, resume='allow' if mode == 'online' else None,
                dir=str(directory), config=publisher_config(directory),
                allow_val_change=True, settings=wandb.Settings(init_timeout=30,
                    x_service_wait=30, x_disable_stats=True, disable_git=True,
                    console='off', quiet=True))
            run.define_metric('train/step')
            run.define_metric('train/tokens_seen')
            run.define_metric('*', step_metric='train/step', step_sync=False)
            emit({'event': 'connected', 'id': run.id, 'url': run.url, 'next_step': run.step})
            last = run.step - 1
            with (directory / 'metrics.jsonl').open() as stream:
                while True:
                    submitted = submit_available(stream, run, wandb)
                    if submitted is not None:
                        last = submitted
                    session = read_json(directory / 'session.json', {})
                    alive = process_identity(session.get('pid', 0)) == session.get('process_start') is not None
                    ended = session.get('state') != 'running' or not alive
                    save_status('offline_saved' if mode == 'offline' else 'streaming',
                                last_submitted_step=last, url=run.url,
                                note='SDK submission is not a remote acknowledgement; JSONL is authoritative')
                    if args.once or ended:
                        # Catch the final complete row after the end marker became visible.
                        submitted = submit_available(stream, run, wandb)
                        if submitted is not None:
                            last = submitted
                        run.summary['trainer_state'] = session.get('state', 'unknown') if alive else (
                            session.get('state') if session.get('state') != 'running' else 'exited_without_finish')
                        run.summary['final_recorded_optimizer_step'] = last
                        run.summary['local_telemetry_preserved'] = True
                        if session.get('summary'):
                            run.summary['training_summary'] = session['summary']
                        save_status('finishing', last_submitted_step=last)
                        run.finish()
                        save_status('offline_saved' if mode == 'offline' else 'finished', last_submitted_step=last)
                        return
                    time.sleep(2)
        except Exception as exc:
            # No secrets/config/environment dumps; the SDK's own log holds details.
            save_status('retrying', error_type=type(exc).__name__)
            emit({'event': 'retrying', 'error_type': type(exc).__name__, 'wait_seconds': retry_delay})
            if run is not None:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            time.sleep(retry_delay)
            retry_delay = min(60, retry_delay * 2)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--mode', choices=['online', 'offline', 'disabled'])
    p.add_argument('--once', action='store_true', help='Replay existing JSONL and exit; do not wait for the trainer')
    p.add_argument('--drain-seconds', type=int, default=300,
                   help='Retry connections this long after training ends; telemetry always remains on disk')
    main(p.parse_args())
