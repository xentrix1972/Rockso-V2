"""Read-only optimizer-boundary diagnostics; no hooks in compiled model forwards."""
from collections import defaultdict
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import json

import torch

from monitoring_io import prepare_identity, process_identity, read_json, repair_partial_tail, utc_now, write_json
from rockso_hybrid.runtime import raw_time


def parameter_family(name):
    if name in ('embed.weight', 'lm_head.weight'):
        return 'embed'
    if name == 'value_embed.weight':
        return 'value_embed'
    if '.mudd.' in name:
        return 'mudd'
    if '.q_norm.' in name or '.k_norm.' in name:
        return 'qk_norm'
    if 'norm' in name:
        return 'norm'
    if '.mlp.' in name:
        return 'mlp'
    if any(s in name for s in ('q_proj.', 'k_proj.', 'v_proj.', 'out_proj.')):
        return 'attn'
    if '_gate.' in name:
        return 'gates'
    return 'scalars'


class UpdateDiagnostics:
    """Exact weight/gradient L2 reductions; deterministic samples of actual updates.

    Gradients are read after response-token normalization and BEFORE clipping.
    NorMuon can mutate .grad in step(), so post-step gradients must not be used.
    Only small sampled weights are cloned, not full optimizer/model tensors.
    """
    def __init__(self, model, optimizer, sample_size=2048):
        self.parameters = list(model.named_parameters())  # tied weights deduplicated
        self.masters = {n: p for n, _, p in optimizer.adam_masters}
        self.sample_size = sample_size
        self.cuda = next(model.parameters()).is_cuda
        self.active = False
        self.reset()

    def reset(self):
        self.stats, self.samples, self.events = {}, {}, []
        self.host_seconds = 0.0

    def _begin(self):
        started = raw_time()
        if self.cuda:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            self.events.append((a, b))
        return started

    def _end(self, started):
        if self.cuda:
            self.events[-1][1].record()
        self.host_seconds += raw_time() - started

    def sample(self, p):
        # Offset plus an odd stride samples different columns of aligned matrices.
        stride = max(1, math.ceil(p.numel() / self.sample_size))
        if stride > 1 and stride % 2 == 0:
            stride += 1
        return p.detach().view(-1)[stride // 2::stride][:self.sample_size].float()

    @torch.no_grad()
    def before_step(self):
        if not self.active:
            return
        self.reset()
        started = self._begin()
        for name, p in self.parameters:
            # Reduction accumulates in FP32; no full-size .float().square() temporary.
            w2 = torch.linalg.vector_norm(p.detach(), dtype=torch.float32).square()
            g2 = (torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32).square()
                  if p.grad is not None else w2.new_zeros(()))
            rec = self.stats[name] = {'weight_sq': w2, 'grad_sq': g2,
                                     'count': p.numel(), 'has_grad': p.grad is not None}
            if parameter_family(name) in ('embed', 'value_embed', 'attn', 'mlp'):
                before = self.sample(p).clone()
                self.samples[name] = before
                rec['sample_weight_sq'] = before.square().sum()
                rec['sample_count'] = before.numel()
                if name in self.masters:
                    master = self.sample(self.masters[name]).clone()
                    self.samples[name + ':master'] = master
                    rec['master_weight_sq'] = master.square().sum()
        self._end(started)

    @torch.no_grad()
    def after_step(self):
        if not self.active:
            return
        started = self._begin()
        for name, p in self.parameters:
            rec = self.stats[name]
            if name in self.samples:
                delta = self.sample(p) - self.samples[name]
                rec['update_sq'] = delta.square().sum()
                rec['unchanged_count'] = (delta == 0).sum().float()
                if name in self.masters:
                    delta = self.sample(self.masters[name]) - self.samples[name + ':master']
                    rec['master_update_sq'] = delta.square().sum()
                    rec['master_unchanged_count'] = (delta == 0).sum().float()
            if '_lambda_' in name:
                rec['value'] = p.detach().float().reshape(())
            elif 'xsa_alpha' in name:
                alpha = p.detach().float().tanh()
                rec.update(value=alpha.mean(), value_min=alpha.min(), value_max=alpha.max())
        self.samples.clear()
        self._end(started)

    def result(self):
        if not self.active:
            return {}, []
        slots = [(name, key, value) for name, rec in self.stats.items()
                 for key, value in rec.items() if isinstance(value, torch.Tensor)]
        values = torch.stack([x[2] for x in slots]).detach().cpu().tolist()
        records = {n: {k: v for k, v in rec.items() if not isinstance(v, torch.Tensor)}
                   for n, rec in self.stats.items()}
        for (name, key, _), value in zip(slots, values):
            records[name][key] = value
        families, layers = defaultdict(list), defaultdict(lambda: defaultdict(list))
        scalar_values = defaultdict(list)
        for name, rec in records.items():
            family = parameter_family(name)
            families[family].append(rec)
            if name.startswith('blocks.'):
                layer = int(name.split('.')[1])
                layers[layer][family].append(rec)
                if 'value' in rec:
                    label = name.split('.')[-1]
                    layers[layer][label].append(rec)
                    scalar_values[label].append(rec['value'])
            if 'xsa_alpha' in name:
                scalar_values['xsa_min'].append(rec['value_min'])
                scalar_values['xsa_max'].append(rec['value_max'])

        def norms(items):
            count = sum(r['count'] for r in items)
            return (math.sqrt(sum(r['weight_sq'] for r in items) / count),
                    math.sqrt(sum(r['grad_sq'] for r in items)))

        def ratio(items, master=False):
            prefix = 'master_' if master else ''
            # Each tensor's sample estimates its population sum, then aggregate.
            scale = lambda r: r['count'] / r['sample_count']
            dw = sum(r[prefix + 'update_sq'] * scale(r) for r in items)
            wkey = 'master_weight_sq' if master else 'sample_weight_sq'
            w = sum(r[wkey] * scale(r) for r in items)
            zero = sum(r[prefix + 'unchanged_count'] * scale(r) for r in items)
            return math.sqrt(dw / max(w, 1e-30)), zero / sum(r['count'] for r in items)

        metrics = {}
        for family, items in sorted(families.items()):
            w, g = norms(items)
            metrics[f'weights/{family}_rms'] = w
            metrics[f'grads/{family}_norm_preclip'] = g
            if family in ('embed', 'value_embed', 'attn', 'mlp'):
                r, z = ratio(items)
                metrics[f'updates/{family}_relative_sampled'] = r
                metrics[f'updates/{family}_unchanged_fraction_sampled'] = z
                if all('master_update_sq' in r for r in items):
                    r, z = ratio(items, master=True)
                    metrics[f'updates/{family}_master_relative_sampled'] = r
                    metrics[f'updates/{family}_master_unchanged_fraction_sampled'] = z
        for label, values in scalar_values.items():
            if label in ('xsa_min', 'xsa_max'):
                continue
            for stat, function in (('min', min), ('mean', statistics.mean), ('max', max)):
                metrics[f'balance/{label}_{stat}'] = function(values)
        if scalar_values['xsa_min']:
            metrics['balance/xsa_alpha_min'] = min(scalar_values['xsa_min'])
            metrics['balance/xsa_alpha_max'] = max(scalar_values['xsa_max'])
        table = []
        for layer, groups in sorted(layers.items()):
            row = {'layer': layer}
            for family in ('attn', 'mlp', 'norm', 'qk_norm', 'gates', 'mudd'):
                if family in groups:
                    w, g = norms(groups[family])
                    row[family + '_weight_rms'] = w
                    row[family + '_grad_norm_preclip'] = g
                    if family in ('attn', 'mlp'):
                        row[family + '_relative_update_sampled'] = ratio(groups[family])[0]
            row['grad_norm_preclip'] = math.sqrt(sum(r['grad_sq'] for k, items in groups.items()
                if k in families for r in items))
            if 'attn' in groups and 'mlp' in groups:
                row['mlp_to_attn_weight_rms'] = row['mlp_weight_rms'] / max(row['attn_weight_rms'], 1e-30)
            for label in scalar_values:
                if label in groups:
                    row[label] = statistics.mean(r['value'] for r in groups[label])
            table.append(row)
        for column in ('grad_norm_preclip', 'mlp_to_attn_weight_rms'):
            available = [r for r in table if column in r]
            if available:
                vals = [r[column] for r in available]
                for stat, fn in (('min', min), ('median', statistics.median), ('max', max)):
                    metrics[f'layers/{column}_{stat}'] = fn(vals)
        if table:
            metrics['layers/largest_gradient_layer'] = max(table, key=lambda r: r['grad_norm_preclip'])['layer']
        metrics['monitor/diagnostic_host_seconds'] = self.host_seconds
        if self.cuda:
            intervals = [a.elapsed_time(b) for a, b in self.events]
            # Some WSL/driver event timestamps can be negative. Do not chart an
            # invalid timing as a measured speedup; host/wall timing stays visible.
            if all(math.isfinite(ms) and ms >= 0 for ms in intervals):
                metrics['monitor/diagnostic_gpu_ms'] = sum(intervals)
        self.reset()
        return metrics, table


class TrainingMonitor:
    def __init__(self, model, optimizer, args, manifest, progress, *, spawn=True):
        self.args = args
        self.diag = UpdateDiagnostics(model, optimizer, args.monitor_sample_size)
        self.identity = prepare_identity(args.out, entity=args.wandb_entity, project=args.wandb_project,
            name=args.wandb_name or args.out.name, mode=args.wandb_mode,
            resume_step=progress['step'] if args.resume else None)
        self.directory = Path(self.identity['directory'])
        self.session = dict(pid=os.getpid(), process_start=process_identity(os.getpid()),
                            started_utc=utc_now(), state='running', start_step=progress['step'],
                            target_step=args.steps, arguments={k: str(v) if isinstance(v, Path) else v
                                                             for k, v in vars(args).items()})
        write_json(self.directory / 'session.json', self.session)
        write_json(self.directory / 'config.json', dict(manifest=manifest, monitoring=dict(
            basic_every=args.wandb_every, diagnostics_every=args.monitor_every,
            layer_table_every=args.monitor_layer_every, update_samples_per_tensor=args.monitor_sample_size)))
        repair_partial_tail(self.directory / 'metrics.jsonl')
        self.stream = (self.directory / 'metrics.jsonl').open('a', buffering=1)
        self.pending = []
        self.last_row = None
        self.publisher = None
        self.publish_enabled = spawn and args.wandb_mode != 'disabled'
        if self.publish_enabled:
            self.start_publisher()

    def start_publisher(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            with (self.directory / 'publisher.log').open('a') as log:
                self.publisher = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('wandb_publisher.py')),
                    '--directory', str(self.directory)], env=env, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            print(json.dumps({'event': 'wandb_publisher_unavailable', 'error': str(exc),
                              'local_metrics': str(self.directory / 'metrics.jsonl')}), flush=True)

    def begin(self, step):
        self.diag.active = bool(self.args.monitor_every and
            (step == 1 or step % self.args.monitor_every == 0 or step == self.args.steps))

    def record(self, row, *, final=False):
        self.last_row = row
        self.pending.append(row)
        if not (final or row['step'] == 1 or row['step'] % self.args.wandb_every == 0 or self.diag.active):
            return
        started = raw_time()
        items = self.pending
        targets = sum(r['supervised_targets'] for r in items)
        slots = sum(r['slot_tokens'] for r in items)
        # Use the trainer's contiguous intervals: clock and overhead accounting
        # must agree with the terminal. Legacy fixture rows contain only raw time.
        seconds = sum(r.get('wall_seconds', r['raw_seconds']) for r in items)
        update_seconds = sum(r['raw_seconds'] for r in items)
        norms = [r['grad_norm'] for r in items]
        metrics = {
            'train/step': row['step'], 'train/tokens_seen': row['tokens_seen'],
            'train/loss': sum(r['loss'] * r['supervised_targets'] for r in items) / targets,
            'train/loss_last': row['loss'], 'train/grad_norm_preclip': norms[-1],
            'train/grad_norm_max': max(norms),
            'train/clipped_fraction': sum(g > self.args.clip_grad for g in norms) / len(norms),
            'train/clip_scale': min(1., self.args.clip_grad / (row['grad_norm'] + 1e-6)),
            'train/supervised_fraction': targets / slots,
            'train/updates_in_window': len(items), 'schedule/lr_factor': row['lr_factor'],
            'perf/packed_tokens_per_second': slots / seconds,
            'perf/supervised_tokens_per_second': targets / seconds,
            'perf/update_seconds': seconds / len(items),
            'perf/wall_tokens_per_second': slots / seconds,
            'perf/update_only_tokens_per_second': slots / update_seconds,
            'perf/update_only_seconds': update_seconds / len(items),
        }
        if row['muon_momentum'] is not None:
            metrics['schedule/muon_momentum'] = row['muon_momentum']
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            available = [r[key] for r in items if r.get(key) is not None]
            if available:
                metrics['memory/' + key.replace('_bytes', '_gib')] = max(available) / 2**30
        metrics.update({'lr/' + k: v for k, v in row['learning_rates'].items() if v is not None})
        if row['learning_rates'].get('normuon') is not None:
            # Rockso V2 gate/up uses the configured NorMuon rate; no extra shape factor.
            metrics['lr/normuon_gate_up_effective'] = row['learning_rates']['normuon']
        diagnostics, layers = self.diag.result()
        metrics.update(diagnostics)
        metrics['monitor/serialize_seconds'] = raw_time() - started
        record = {'step': row['step'], 'recorded_utc': utc_now(), 'metrics': metrics}
        if layers:
            record['layers'] = layers  # all layers locally every diagnostic interval
            record['publish_layers'] = (row['step'] == 1 or final or row['step'] == self.args.steps or
                (self.args.monitor_layer_every > 0 and row['step'] % self.args.monitor_layer_every == 0))
        self.stream.write(json.dumps(record, allow_nan=False) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.pending = []
        # A failed publisher is restarted only at the normal metric cadence.
        if self.publish_enabled and (self.publisher is None or self.publisher.poll() is not None):
            status = read_json(self.directory / 'publisher_status.json', {})
            active = process_identity(status.get('pid', 0)) == status.get('process_start') is not None
            if not active:
                self.start_publisher()

    def finish(self, state, summary):
        if self.pending:
            last = self.pending.pop()
            self.diag.active = False
            self.record(last, final=True)
        self.stream.close()
        self.session.update(state=state, finished_utc=utc_now(), summary=summary)
        write_json(self.directory / 'session.json', self.session)
