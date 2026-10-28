"""Ordered CPU microbatch preparation; checkpoint progress belongs to the trainer.

Workers never access CUDA. A private CPU generator keeps worker startup out of
the training RNG stream, including NorMuon's stochastic-rounding sequence.
Prefetched rows are disposable: resume constructs a new iterator at the saved
consumed cursor, not the iterator's dispatched or delivered position.
"""
import signal
import torch
from torch.utils.data import DataLoader, Dataset
from .masking import response_labels


def _identity(item):
    return item


def _worker_init(worker_id):
    # The parent handles Ctrl-C by finishing the optimizer update and saving.
    # Its data workers must remain available until that update has finished.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


class _Microbatches(Dataset):
    def __init__(self, source, start_cursor, batch_size, microbatches, pad_id):
        self.source, self.start_cursor = source, start_cursor
        self.batch_size, self.microbatches, self.pad_id = batch_size, microbatches, pad_id

    def __len__(self):
        return self.microbatches

    def __getitem__(self, index):
        if not 0 <= index < self.microbatches:
            raise IndexError(index)
        cursor = self.start_cursor + index * self.batch_size
        batch = self.source.get_batch(cursor, self.batch_size)
        labels = response_labels(batch['input_ids'], batch['metadata_mask'],
                                 batch['span_ends'], self.pad_id)
        return {'cursor': cursor, 'batch': batch, 'labels': labels,
                'supervised_targets': int((labels[:, 1:] != -100).sum())}


class BatchPipeline:
    def __init__(self, source, *, start_cursor, batch_size, grad_accum,
                 updates, pad_id, num_workers=0, prefetch_factor=2,
                 persistent_workers=True, pin_memory=False):
        if (num_workers < 0 or prefetch_factor < 1 or batch_size < 1 or
                grad_accum < 1 or updates < 0 or not 0 <= start_cursor <= source.length):
            raise ValueError('Invalid data pipeline dimensions or starting cursor')
        self.length = source.length
        self.batch_size, self.next_cursor = batch_size, start_cursor
        complete_updates = min(updates, (source.length - start_cursor) // (batch_size * grad_accum))
        self.microbatches = complete_updates * grad_accum
        dataset = _Microbatches(source, start_cursor, batch_size, self.microbatches, pad_id)
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        options = dict(dataset=dataset, batch_size=None, shuffle=False,
                       collate_fn=_identity, num_workers=num_workers,
                       # PyTorch's pin-memory helper calls set_num_threads(1).
                       # In this runtime that changes the state seen by compiled
                       # backward, duplicates block specializations and exhausts
                       # the compile cache. Pin on the consuming thread instead.
                       pin_memory=False, generator=torch.Generator().manual_seed(0))
        if num_workers:
            options.update(multiprocessing_context='spawn', worker_init_fn=_worker_init,
                           persistent_workers=self.persistent_workers,
                           prefetch_factor=prefetch_factor, in_order=True)
        self.loader = DataLoader(**options)
        self.iterator = None
        self.closed = False

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Data pipeline has been closed')
        if self.iterator is None and self.microbatches:
            self.iterator = iter(self.loader)
        return self

    @property
    def worker_pids(self):
        return [w.pid for w in getattr(self.iterator, '_workers', ())]

    def get_training_batch(self, cursor, batch_size):
        if cursor != self.next_cursor or batch_size != self.batch_size:
            raise ValueError('Data pipeline request differs from the next ledger microbatch')
        if self.closed:
            raise RuntimeError('Data pipeline has been closed')
        if self.iterator is None:
            self.__enter__()
        if self.iterator is None:
            raise StopIteration('No complete optimizer update remains')
        item = next(self.iterator)
        if item['cursor'] != cursor:
            raise RuntimeError('DataLoader returned an out-of-order ledger microbatch')
        if self.pin_memory:
            item['batch'] = {k: v.pin_memory() for k, v in item['batch'].items()}
            item['labels'] = item['labels'].pin_memory()
        self.next_cursor += batch_size  # delivery only; never used in a checkpoint
        return item['batch'], item['labels'], item['supervised_targets']

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Persistent iterators otherwise retain their worker processes. PyTorch
        # has no public close API; keep its shutdown call isolated here.
        shutdown = getattr(self.iterator, '_shutdown_workers', None)
        if shutdown is not None:
            shutdown()
        self.iterator = None
        self.loader = None

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
