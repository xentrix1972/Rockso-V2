import json
from pathlib import Path
import numpy as np
import torch
from .runtime import sha256


def collate(rows, *, seq_len, vocab_size):
    ids = torch.tensor([r['input_ids'] for r in rows], dtype=torch.long)
    if ids.ndim != 2 or ids.shape[1] != seq_len or torch.any(ids < 0) or torch.any(ids >= vocab_size):
        raise ValueError('Input IDs violate block size or vocabulary contract')
    if any('metadata_mask' not in r or 'span_ends' not in r for r in rows):
        raise ValueError('Packed training requires metadata_mask and span_ends')
    meta = torch.tensor([r['metadata_mask'] for r in rows], dtype=torch.int8)
    if meta.shape != ids.shape or not ((meta == 0) | (meta == 1)).all():
        raise ValueError('metadata_mask must match input_ids and contain only 0/1')
    ends = [list(r['span_ends']) for r in rows]
    for spans in ends:
        positive = [int(e) for e in spans if e > 0]
        if not positive or positive != sorted(set(positive)) or positive[-1] > seq_len:
            raise ValueError('span_ends must contain increasing document boundaries')
    width = max(map(len, ends))
    spans = torch.tensor([r + [0] * (width - len(r)) for r in ends], dtype=torch.int32)
    supplied = ['position_ids' in r for r in rows]
    if any(supplied) and not all(supplied):
        raise ValueError('Mixed position_ids contract within batch')
    pos = torch.tensor([r['position_ids'] for r in rows], dtype=torch.long) if all(supplied) else torch.arange(seq_len)[None].expand_as(ids)
    if pos.shape != ids.shape or torch.any(pos < 0):
        raise ValueError('Invalid position_ids')
    return {'input_ids': ids, 'position_ids': pos, 'span_ends': spans, 'metadata_mask': meta}


class PackedSource:
    def __init__(self, path, *, split='train', ledger=None, ledger_meta=None, sequential=False,
                 seq_len=2048, vocab_size=65536):
        from datasets import load_from_disk, DatasetDict
        self.path = Path(path).resolve()
        self.split = split
        self.ledger_path = Path(ledger).resolve() if ledger is not None else None
        data = load_from_disk(str(self.path))
        self.data = data[split] if isinstance(data, DatasetDict) else data
        self.seq_len, self.vocab_size = seq_len, vocab_size
        if ledger is None and not sequential:
            raise ValueError('Supply a ledger or explicitly select sequential order')
        if ledger is not None and sequential:
            raise ValueError('Ledger and sequential order are mutually exclusive')
        if ledger is not None and ledger_meta is None:
            raise ValueError('Ledger metadata is required to preserve its coordinate contract')
        self.order = np.load(ledger, mmap_mode='r', allow_pickle=False) if ledger else None
        self.length = len(self.order) if self.order is not None else len(self.data)
        self.receipt = {'path': str(self.path), 'split': split, 'dataset_fingerprint': self.data._fingerprint,
                        'rows': len(self.data), 'sequence_length': seq_len, 'vocab_size': vocab_size,
                        'order_length': self.length, 'order': 'ledger' if ledger else 'sequential'}
        # Arrow payloads are read-only. Record identity/size/mtime; the producer's
        # content receipt remains authoritative (this is not a fresh full corpus hash).
        self.receipt['cache_files'] = [dict(path=f['filename'], size=Path(f['filename']).stat().st_size,
            mtime_ns=Path(f['filename']).stat().st_mtime_ns) for f in self.data.cache_files]
        if self.order is not None:
            if self.order.ndim != 1 or self.order.dtype.kind not in 'iu':
                raise ValueError('Ledger must be a one-dimensional integer array')
            metadata = json.loads(Path(ledger_meta).read_text())
            declared = metadata.get('block_size', metadata.get('contract', {}).get('block_size'))
            if declared is not None and int(declared) != seq_len:
                raise ValueError('Ledger block size differs from trainer')
            self.receipt.update(ledger_sha256=sha256(ledger), ledger_meta_sha256=sha256(ledger_meta),
                                ledger_metadata=metadata)

    def __getstate__(self):
        # Spawn workers with paths/metadata, not a pickled Arrow dataset or a
        # copied NumPy memmap payload. Each worker opens the same read-only files.
        state = dict(self.__dict__)
        state.pop('data')
        state.pop('order')
        return state

    def __setstate__(self, state):
        from datasets import load_from_disk, DatasetDict
        self.__dict__.update(state)
        data = load_from_disk(str(self.path))
        self.data = data[self.split] if isinstance(data, DatasetDict) else data
        self.order = np.load(self.ledger_path, mmap_mode='r', allow_pickle=False) if self.ledger_path else None
        length = len(self.order) if self.order is not None else len(self.data)
        if (length != self.length or len(self.data) != self.receipt['rows'] or
                self.data._fingerprint != self.receipt['dataset_fingerprint']):
            raise ValueError('Worker dataset identity differs from the parent source')

    def get_batch(self, cursor, batch_size):
        if cursor < 0 or cursor + batch_size > self.length:
            raise StopIteration('Insufficient rows for another complete microbatch')
        indices = self.order[cursor:cursor + batch_size].tolist() if self.order is not None else list(range(cursor, cursor + batch_size))
        if any(i < 0 or i >= len(self.data) for i in indices):
            raise ValueError('Ledger row index lies outside its dataset')
        return collate([self.data[int(i)] for i in indices], seq_len=self.seq_len, vocab_size=self.vocab_size)


class SyntheticSource:
    """Deterministic per-row synthetic packing, explicitly excluded from quality claims."""
    def __init__(self, seq_len, vocab_size, seed):
        self.seq_len, self.vocab_size, self.seed = seq_len, vocab_size, seed
        self.length = 2**40
        self.receipt = {'kind': 'synthetic', 'seq_len': seq_len, 'vocab_size': vocab_size, 'seed': seed}

    def get_batch(self, cursor, batch_size):
        rows = []
        for i in range(cursor, cursor + batch_size):
            g = torch.Generator().manual_seed(self.seed + i)
            ids = torch.randint(1, self.vocab_size, (self.seq_len,), generator=g).tolist()
            prefix = max(1, self.seq_len // 4)
            rows.append({'input_ids': ids, 'metadata_mask': [1]*prefix + [0]*(self.seq_len-prefix),
                         'span_ends': [self.seq_len]})
        return collate(rows, seq_len=self.seq_len, vocab_size=self.vocab_size)


def tokenizer_receipt(path, vocab_size):
    from transformers import AutoTokenizer
    root = Path(path).resolve()
    tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    if len(tok) != vocab_size:
        raise ValueError(f'Tokenizer has {len(tok)} tokens; model has {vocab_size}')
    if tok.pad_token_id is not None and tok.pad_token_id == tok.eos_token_id:
        raise ValueError('Pad token equals EOS; refusing to silently mask supervised endings')
    files = {p.name: sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
    return tok.pad_token_id, {'path': str(root), 'files': files, 'vocab_size': len(tok),
                              'pad_id': tok.pad_token_id, 'eos_id': tok.eos_token_id}
