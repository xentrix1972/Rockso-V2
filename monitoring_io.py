"""Small, torch-free helpers shared by training telemetry and its CPU publisher."""
import datetime
import json
import os
from pathlib import Path
import uuid


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def process_identity(pid):
    """PID plus Linux start ticks avoids mistaking a reused PID for this trainer."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if text[0] == 'Z' else text[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def last_record(path):
    """Read the last complete JSONL record without scanning a long run."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0, 2)
        end = f.tell()
        data = b''
        while end:
            size = min(end, 65536)
            end -= size
            f.seek(end)
            data = f.read(size) + data
            complete = data.split(b'\n')[:-1]
            if end:
                complete = complete[1:]
            for line in reversed(complete):
                if line.strip():
                    return json.loads(line)
    return None


def complete_records(stream):
    """A writer can still be midway through the last line; retry it next poll."""
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line.endswith('\n'):
            stream.seek(offset)
            return
        if line.strip():
            yield json.loads(line)


def repair_partial_tail(path):
    """Preserve an interrupted write separately before appending after a crash."""
    path = Path(path)
    if not path.exists() or not path.stat().st_size:
        return
    with path.open('rb+') as f:
        f.seek(-1, 2)
        if f.read(1) == b'\n':
            return
        end = f.tell()
        offset, tail = end, b''
        while offset:
            size = min(offset, 65536)
            offset -= size
            f.seek(offset)
            tail = f.read(size) + tail
            boundary = tail.rfind(b'\n')
            if boundary >= 0:
                offset += boundary + 1
                tail = tail[boundary + 1:]
                break
        backup = path.with_name(path.name + '.partial.' + uuid.uuid4().hex[:8])
        backup.write_bytes(tail)
        f.truncate(offset)
        f.flush()
        os.fsync(f.fileno())


def prepare_identity(out, *, entity, project, name, mode, resume_step=None):
    out = Path(out)
    old = read_json(out / 'wandb_run.json')
    reason = None
    if old:
        last = last_record(Path(old['directory']) / 'metrics.jsonl')
        if (old['entity'], old['project']) != (entity, project):
            reason = 'W&B destination changed'
        elif resume_step is None or (last and resume_step < last['step']):
            reason = 'New lineage or resume earlier than already recorded updates'
        else:
            old.update(mode=mode, name=name)
            write_json(out / 'wandb_run.json', old)
            write_json(Path(old['directory']) / 'identity.json', old)
            return old
    run_id = uuid.uuid4().hex[:12]  # OS entropy; never touches training RNG.
    directory = out / 'monitoring' / run_id
    directory.mkdir(parents=True)
    identity = dict(id=run_id, entity=entity, project=project, name=name, mode=mode,
                    directory=str(directory), created_utc=utc_now(),
                    parent_run_id=old['id'] if old else None, branch_reason=reason)
    write_json(directory / 'identity.json', identity)
    write_json(out / 'wandb_run.json', identity)
    return identity
