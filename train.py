#!/usr/bin/env python3
"""Rockso V2 production trainer with editable run settings and compatible checkpoint resume."""
import argparse
import contextlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import time
import torch

from rockso_hybrid import ModelConfig, RocksoHybrid
from rockso_hybrid.data import PackedSource, SyntheticSource, tokenizer_receipt
from rockso_hybrid.input_pipeline import BatchPipeline
from rockso_hybrid.masking import build_masks, response_labels
from rockso_hybrid.loss import causal_loss
from rockso_hybrid.runtime import (HybridOptimizer, apply_fp8, atomic_save, gpu_ownership,
    import_rockso_weights, load_checkpoint, save_checkpoint, seed_all, sha256, source_receipt, raw_time)
from telemetry import TrainingMonitor

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Optional JSON base; explicit architecture flags override it')
    geometry = p.add_argument_group('model architecture (editable directly in the launcher)')
    for name in ('vocab_size', 'n_layer', 'n_embd', 'n_head', 'n_kv_head', 'mlp_hidden',
                 'max_seq_len', 'attention_gate_channels', 've_start', 've_every',
                 've_gate_channels', 'mudd_rank', 'local_window', 'global_every'):
        geometry.add_argument('--' + name.replace('_', '-'), type=int, default=None,
                              dest='model_' + name)
    geometry.add_argument('--d-ff', '--d_ff', type=int, default=None, dest='model_d_ff',
                          help='Original Rockso width convention: actual ReLU-squared hidden width is int(1.5*d_ff)')
    geometry.add_argument('--mlp-activation', choices=['relu2', 'swiglu'], default=None, dest='model_mlp_activation')
    for name in ('tie_embeddings', 'value_residual', 'attention_gate', 'xsa',
                 'value_embedding', 'global_nope', 'sigmoid_logit_cap', 'smoothing', 'residual_multipliers'):
        geometry.add_argument('--' + name.replace('_', '-'), action=argparse.BooleanOptionalAction,
                              default=None, dest='model_' + name)
    geometry.add_argument('--mudd-layers', nargs='*', type=int, default=None, dest='model_mudd_layers',
                          help='Zero-based sites, e.g. 16 31; no numbers disables MUDD')
    geometry.add_argument('--rope-fraction', type=float, default=None, dest='model_rope_fraction')
    geometry.add_argument('--rope-base', type=float, default=None, dest='model_rope_base')
    geometry.add_argument('--rope-math', choices=['rockso', 'float32'], default=None, dest='model_rope_math')
    geometry.add_argument('--attention-scale', default=None, dest='model_attention_scale',
                          help='auto (1/sqrt(head_dim)), or a literal numeric scale')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--split', default='train')
    p.add_argument('--ledger', type=Path)
    p.add_argument('--ledger-meta', type=Path)
    p.add_argument('--sequential', action='store_true')
    p.add_argument('--start-row', type=int, default=0, help='Explicit coordinate within this ledger; new lineage only')
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--tokenizer', type=Path)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--check-gpu', action='store_true', help='Opt-in ownership check for benchmark automation')
    p.add_argument('--dtype', choices=['float32','bfloat16'], default='bfloat16')
    p.add_argument('--backend', choices=['sdpa','flex'], default='flex')
    p.add_argument('--loss', choices=['reference','liger'], default='liger')
    p.add_argument('--optimizer', choices=['normuon','adamw'], default='normuon')
    p.add_argument('--fp8', action='store_true')
    p.add_argument('--fp8-mlp-pad-to', type=int, default=0, help='Optional GEMM-only zero padding; keeps logical SwiGLU width, scales and optimizer shapes unchanged')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--keep-last', type=int, default=0)
    p.add_argument('--native-gqa', action='store_true')
    p.add_argument('--attention-mode', choices=['prefixlm','doc_causal','causal'], default='prefixlm')
    p.add_argument('--batch-size', type=int, default=20)
    p.add_argument('--grad-accum', type=int, default=8)
    p.add_argument('--num-workers', type=int, default=0, help='Ordered CPU data workers; 0 prepares batches on the training thread')
    p.add_argument('--prefetch-factor', type=int, default=2, help='Microbatches queued per worker')
    p.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--non-blocking', action=argparse.BooleanOptionalAction, default=False,
                   help='Enqueue device copies without synchronizing the host after each tensor')
    p.add_argument('--steps', type=int, required=True, help='Absolute final update index, including on resume')
    p.add_argument('--schedule-steps', type=int, default=0, help='0=constant LR; editable on resume')
    p.add_argument('--lr-schedule', choices=['cosine', 'wsd'], default='cosine',
                   help='Cosine after warmup, or warmup/stable/decay; editable on resume')
    p.add_argument('--wsd-decay-style', choices=['cosine', 'linear'], default='cosine',
                   help='Decay shape for WSD only; editable on resume')
    p.add_argument('--stable-until-step', type=int, default=0,
                   help='WSD only: last update at peak LR before decay; 0 for a constant-LR benchmark')
    p.add_argument('--warmup-steps', type=int, default=0)
    p.add_argument('--min-lr-ratio', type=float, default=0.1)
    p.add_argument('--muon-lr', type=float, default=0.0029)
    p.add_argument('--muon-momentum-start', type=float, default=0.85,
                   help='NorMuon momentum on update 1 when momentum warmup is enabled')
    p.add_argument('--muon-momentum', type=float, default=0.95,
                   help='NorMuon momentum after warmup; constant when warmup steps is 0')
    p.add_argument('--muon-momentum-warmup-steps', type=int, default=0,
                   help='Absolute update that reaches target momentum; 0 disables the ramp; editable on resume')
    p.add_argument('--adam-lr', type=float, default=0.0003)
    p.add_argument('--ve-lr-mult', type=float, default=1.0, help='Multiplier for value_embed.weight only, preserved by the scheduler and resume')
    p.add_argument('--ve-weight-decay', type=float, default=None, help='Explicit decay coefficient for value_embed.weight; default matches --weight-decay')
    p.add_argument('--residual-lr-mult', type=float, default=1.0, help='LR multiplier for FP32 residual scalars; no decay')
    p.add_argument('--branch-lr-mult', type=float, default=4.0, help='LR multiplier for FP32 attention/MLP branch scalars; no decay')
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--fp32-master-weights', action='store_true', help='FP32 embedding optimizer weights and moments; NorMuon masters unless stochastic mode is selected')
    p.add_argument('--normuon-stochastic-rounding', action='store_true', help='Stochastic BF16 NorMuon updates instead of full FP32 matrix masters')
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--save-every', type=int, default=0)
    p.add_argument('--log-every', type=int, default=1,
                   help='Print metrics every N updates plus the final update; metrics.jsonl keeps every update')
    monitoring = p.add_argument_group('compact local diagnostics and asynchronous W&B publishing')
    monitoring.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'], default='disabled')
    monitoring.add_argument('--wandb-entity', default='xentrix1972-nick-bochy')
    monitoring.add_argument('--wandb-project', default='Rockso_v2')
    monitoring.add_argument('--wandb-name', default=None)
    monitoring.add_argument('--wandb-every', type=int, default=10,
                            help='Windowed loss/performance/LR metrics; independent of console logging')
    monitoring.add_argument('--monitor-every', type=int, default=0,
                            help='Exact weight/preclip gradient norms and sampled actual updates; 0 disables')
    monitoring.add_argument('--monitor-layer-every', type=int, default=250,
                            help='Publish full-layer tables at this cadence; local layer data follows monitor-every')
    monitoring.add_argument('--monitor-sample-size', type=int, default=2048,
                            help='Maximum fixed samples per weight tensor for update ratios; no RNG consumed')
    p.add_argument('--resume', type=Path,
                   help='Checkpoint path, or auto to select the highest saved step in --out; an empty output starts fresh')
    p.add_argument('--warm-start-rockso', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--benchmark', action='store_true', help='Disposable random-init steps; no weight export or checkpoint')
    p.add_argument('--warmup-updates', type=int, default=2, help='Excluded from benchmark summary, but still recorded')
    return p


def resolve_resume_path(out, checkpoint):
    """Select by saved progress; latest.pt can be older than a periodic save."""
    if checkpoint != Path('auto'):
        return checkpoint
    candidates = [p for p in Path(out).glob('step*.pt')
                  if p.is_file() and p.stem[4:].isdigit()]
    latest = Path(out) / 'latest.pt'
    if latest.is_file():
        candidates.append(latest)
    best, best_key = None, (-1, False)
    for path in sorted(candidates):
        try:
            state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
            if state.get('schema') != 'rockso_hybrid_checkpoint_1':
                raise ValueError('Not a complete hybrid training checkpoint')
            for name in ('model', 'optimizer', 'rng', 'config', 'contract', 'tokens_seen', 'cursor'):
                if name not in state:
                    raise ValueError(f'Checkpoint is missing {name}')
            step = state['step']
            if type(step) is not int or step < 0:
                raise ValueError('Invalid saved step')
            del state  # metadata inspection only; no model allocation or CUDA access
        except Exception as exc:
            raise RuntimeError(f'Cannot inspect resume checkpoint {path}; specify --resume PATH explicitly') from exc
        key = (step, path.name == 'latest.pt')
        if key > best_key:
            best, best_key = path, key
    return best


def resolve_model_config(args):
    values = ModelConfig.load(args.config).to_dict() if args.config is not None else {}
    if args.model_d_ff is not None and (args.model_mlp_activation or values.get('mlp_activation', 'relu2')) == 'swiglu':
        raise ValueError('SwiGLU requires explicit --mlp-hidden; --d_ff is the historical ReLU-squared convention')
    for name, value in vars(args).items():
        if name.startswith('model_') and value is not None:
            key = name.removeprefix('model_')
            if key == 'attention_scale':
                value = None if value == 'auto' else float(value)
            elif key == 'd_ff':
                if args.model_mlp_hidden is not None:
                    raise ValueError('Specify --d_ff or --mlp-hidden, not both')
                key, value = 'mlp_hidden', int(1.5 * value)
            values[key] = value
    return ModelConfig.from_dict(values)


def create_model(c, args):
    model = RocksoHybrid(c)
    model.reset_training_parameters()
    warm_start = None
    if args.warm_start_rockso:
        state = torch.load(args.warm_start_rockso, map_location='cpu', mmap=True, weights_only=False)
        warm_start = import_rockso_weights(model, state['model'])
        warm_start.update(path=str(args.warm_start_rockso.resolve()), sha256=sha256(args.warm_start_rockso))
        del state
    model.to(device=args.device, dtype=getattr(torch, args.dtype)).preserve_precision()
    fp8_names = apply_fp8(model) if args.fp8 else []
    if args.fp8_mlp_pad_to and (not args.fp8 or not args.compile):
        raise ValueError('Qualified MLP padding requires --fp8 --compile')
    if args.fp8:
        from rockso_hybrid.fp8_execution import install_mlp_padding
        install_mlp_padding(model, args.fp8_mlp_pad_to)
    model.configure_runtime(backend=args.backend, checkpointing=args.checkpointing, keep_last=args.keep_last,
                            compile_blocks=args.compile, native_gqa=args.native_gqa)
    return model, fp8_names, warm_start


def update(model, opt, source, cursor, args, pad_id, diagnostics=None):
    opt.zero_grad()
    slots = args.batch_size * model.config.max_seq_len * args.grad_accum
    valid_total = 0
    loss_sum = torch.zeros((), device=args.device, dtype=torch.float32)
    for micro in range(args.grad_accum):
        micro_cursor = cursor + micro * args.batch_size
        if isinstance(source, BatchPipeline):
            batch, labels, count = source.get_training_batch(micro_cursor, args.batch_size)
        else:
            batch = source.get_batch(micro_cursor, args.batch_size)
            labels = response_labels(batch['input_ids'], batch['metadata_mask'], batch['span_ends'], pad_id)
            count = (labels[:, 1:] != -100).sum().item()
        valid_total += count
        non_blocking = getattr(args, 'non_blocking', False)
        batch = {k:v.to(args.device, non_blocking=non_blocking) for k,v in batch.items()}
        labels = labels.to(args.device, non_blocking=non_blocking)
        masks = build_masks(batch['span_ends'], batch['metadata_mask'], args.attention_mode,
                            args.backend, model.config.local_window)
        h = model(batch['input_ids'], batch['position_ids'], masks, return_hidden=True)
        loss = causal_loss(h, model.lm_head.weight, labels, backend=args.loss,
                           sigmoid_cap=model.config.sigmoid_logit_cap)
        (loss * (count / slots)).backward()
        loss_sum += loss.detach().float() * count
    if not valid_total:
        raise ValueError('Accumulation window has no supervised targets')
    groups = {}
    for p in model.parameters():
        if p.grad is not None:
            groups.setdefault((p.device, p.grad.dtype), []).append(p.grad)
    for gradients in groups.values():
        torch._foreach_mul_(gradients, slots / valid_total)
    if diagnostics is not None:
        diagnostics.before_step()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True, foreach=True)
    opt.step()
    if diagnostics is not None:
        diagnostics.after_step()
    return loss_sum / valid_total, valid_total, slots, grad_norm


def validate_schedule(args):
    for name in ('muon_momentum_start', 'muon_momentum'):
        value = getattr(args, name, 0.85 if name.endswith('_start') else 0.95)
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError('NorMuon momentum must be finite and in [0, 1)')
    if getattr(args, 'muon_momentum_warmup_steps', 0) < 0:
        raise ValueError('NorMuon momentum warmup steps must be nonnegative')
    if args.lr_schedule != 'wsd' and args.wsd_decay_style != 'cosine':
        raise ValueError('A linear tail requires the WSD schedule')
    if min(args.schedule_steps, args.warmup_steps, args.stable_until_step) < 0:
        raise ValueError('Schedule coordinates must be nonnegative')
    if args.schedule_steps and args.warmup_steps >= args.schedule_steps:
        raise ValueError('Warmup must end before the schedule horizon')
    if args.lr_schedule == 'cosine' or args.schedule_steps == 0:
        if args.stable_until_step:
            raise ValueError('A stable phase requires WSD with a positive schedule length')
    elif not args.warmup_steps <= args.stable_until_step < args.schedule_steps:
        raise ValueError('WSD stable phase must end after warmup and before the schedule ends')


def lr_factor(step, args):
    """One-based update clock, including resumed updates; no optimizer-group changes."""
    if args.warmup_steps and step <= args.warmup_steps:
        return step / args.warmup_steps
    if not args.schedule_steps:
        return 1.0
    if args.lr_schedule == 'wsd':
        if step <= args.stable_until_step:
            return 1.0
        decay_start = args.stable_until_step
    else:
        decay_start = args.warmup_steps
    frac = max(0.0, min(1.0, (step - decay_start) / (args.schedule_steps - decay_start)))
    if args.lr_schedule == 'wsd' and args.wsd_decay_style == 'linear':
        return args.min_lr_ratio + (1-args.min_lr_ratio)*(1-frac)
    return args.min_lr_ratio + (1-args.min_lr_ratio)*0.5*(1+math.cos(math.pi*frac))


def muon_momentum(step, args):
    """Absolute one-based update: start at 1, target at warmup_steps and later."""
    start = getattr(args, 'muon_momentum_start', 0.85)
    target = getattr(args, 'muon_momentum', 0.95)
    warmup = getattr(args, 'muon_momentum_warmup_steps', 0)
    if warmup <= 1 or step >= warmup:
        return target
    fraction = max(0.0, (step - 1) / (warmup - 1))
    return start + (target - start) * fraction


def apply_optimizer_schedule(opt, step, args):
    """Set scalar coefficients before an update without touching optimizer buffers."""
    factor = lr_factor(step, args)
    opt.scale_lr(factor)
    momentum = muon_momentum(step, args) if opt.muon else None
    if opt.muon:
        for group in opt.muon.param_groups:
            group['momentum'] = momentum
    return factor, momentum


def main(args):
    if args.steps <= 0 or min(args.batch_size, args.grad_accum, args.threads, args.log_every) < 1 or args.start_row < 0:
        raise ValueError('Steps/batch/accumulation/threads/log-every must be positive; cursor must be nonnegative')
    if args.resume and args.warm_start_rockso:
        raise ValueError('Resume and weights-only warm start are different operations')
    if args.benchmark and (args.resume or args.warm_start_rockso):
        raise ValueError('Benchmark is a disposable random-init experiment')
    if args.benchmark and args.steps <= args.warmup_updates:
        raise ValueError('Benchmark needs measured steps after warmup')
    if args.synthetic == (args.data is not None):
        raise ValueError('Choose exactly one of --synthetic and --data')
    if args.data and args.tokenizer is None:
        raise ValueError('Real packed data requires its explicit tokenizer')
    if min(args.wandb_every, args.monitor_sample_size) < 1 or min(args.monitor_every, args.monitor_layer_every) < 0:
        raise ValueError('Logging intervals and sample size must be positive; diagnostics/table intervals may be zero')
    if args.num_workers < 0 or args.prefetch_factor < 1:
        raise ValueError('Data workers must be nonnegative and prefetch factor must be positive')
    validate_schedule(args)
    if not 0 <= args.min_lr_ratio <= 1 or args.clip_grad <= 0:
        raise ValueError('Invalid LR ratio or clipping threshold')
    if args.device == 'cpu' and (args.backend == 'flex' or args.loss == 'liger' or args.fp8 or args.dtype != 'float32'):
        raise ValueError('CPU smoke uses float32, SDPA and reference loss')
    args.out = args.out.resolve()
    automatic_resume = args.resume == Path('auto')
    args.resume = resolve_resume_path(args.out, args.resume)
    if automatic_resume:
        print(json.dumps({'event':'automatic_resume', 'checkpoint':str(args.resume) if args.resume else None}), flush=True)
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f'Resume checkpoint does not exist: {args.resume}')
        args.out.mkdir(parents=True, exist_ok=True)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        if any(args.out.iterdir()):
            raise FileExistsError(f'Fresh-run output directory is not empty: {args.out}; use an empty directory or --resume')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, os.sched_getaffinity(0) - {5})
    torch.set_num_threads(args.threads)
    # Allow TF32 for small FP32 gate GEMMs; recorded in the replay contract.
    torch.set_float32_matmul_precision('high')
    seed_all(args.seed)
    c = resolve_model_config(args)
    source = SyntheticSource(c.max_seq_len, c.vocab_size, args.seed) if args.synthetic else PackedSource(
        args.data, split=args.split, ledger=args.ledger, ledger_meta=args.ledger_meta,
        sequential=args.sequential, seq_len=c.max_seq_len, vocab_size=c.vocab_size)
    pad_id, tok_receipt = (None, None) if args.tokenizer is None else tokenizer_receipt(args.tokenizer, c.vocab_size)
    excluded = {'out','config','steps','resume','warm_start_rockso','save_every','log_every','benchmark','warmup_updates','threads','check_gpu'}
    # Architecture is recorded once in the resolved model config, irrespective
    # of whether its values came from a JSON file or literal launcher flags.
    excluded.update(name for name in vars(args) if name.startswith('model_'))
    excluded.update(name for name in vars(args) if name.startswith(('wandb_', 'monitor_')))
    contract = {'arguments': {k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items() if k not in excluded},
                'data':source.receipt, 'tokenizer':tok_receipt, 'torch':torch.__version__,
                'matmul_precision':'high', 'source_sha256':source_receipt(),
                'environment': {name:importlib.metadata.version(name) for name in
                    ('torch','torchao','triton','liger-kernel','transformers','datasets','numpy')}}
    model, fp8_names, imported = create_model(c, args)
    opt = HybridOptimizer(model, args.muon_lr, args.adam_lr, args.weight_decay, kind=args.optimizer, fp32_master_weights=args.fp32_master_weights,
                          normuon_stochastic_rounding=args.normuon_stochastic_rounding,
                          ve_lr_mult=args.ve_lr_mult, ve_weight_decay=args.ve_weight_decay,
                          residual_lr_mult=args.residual_lr_mult, branch_lr_mult=args.branch_lr_mult)
    progress = {'step':0, 'tokens_seen':0, 'cursor':args.start_row}
    if args.resume:
        progress = load_checkpoint(args.resume, model, opt, contract)
    elif args.start_row >= source.length:
        raise ValueError('Start row exceeds data order')
    _, next_momentum = apply_optimizer_schedule(opt, progress['step'] + 1, args)
    run = {'config':c.to_dict(), 'contract':contract, 'parameter_count':sum(p.numel() for p in model.parameters()),
           'fp8_modules':fp8_names, 'weights_only_import':imported, 'optimizer_groups':opt.group_receipt(),
           'benchmark':args.benchmark, 'gpu':torch.cuda.get_device_name() if args.device=='cuda' else None,
           'throughput_reporting':{'version':2, 'clock':'CLOCK_MONOTONIC_RAW',
               'headline':'completed packed slots / training-loop wall seconds',
               'terminal_window':'completed updates since the previous printed training record',
               'includes':'worker startup, input waits, updates, diagnostics, logging and checkpoint saves',
               'excludes':'initial dataset/checkpoint loading before the training loop',
               'save_accounting':'charged to the next update; terminal saves included in final summary'}}
    if not args.resume or not (args.out/'run.json').exists():
        (args.out/'run.json').write_text(json.dumps(run, indent=2)+'\n')
    if args.resume:
        # Preserve the initial run manifest; record each continuation's chosen settings.
        event = {'event':'resume', 'checkpoint':str(args.resume.resolve()), 'progress':progress,
                 'arguments':contract['arguments'], 'source_sha256':contract['source_sha256'],
                 'environment':contract['environment'], 'optimizer_groups':run['optimizer_groups'],
                 'throughput_reporting':run['throughput_reporting']}
        with (args.out/'resume_history.jsonl').open('a') as history:
            history.write(json.dumps(event)+'\n')
    print(json.dumps({'event':'ready', 'parameters':run['parameter_count'], 'out':str(args.out),
                      'starting_step':progress['step'], 'cursor':progress['cursor'],
                      'next_muon_momentum':next_momentum,
                      'throughput_reporting':run['throughput_reporting']}), flush=True)
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
        print(json.dumps({'event':'stop_requested','action':'finish_current_update_then_checkpoint'}), flush=True)
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    measured = []
    display_window = []
    # Contiguous intervals include work between updates. Never reset this after
    # logging or saving: that overhead belongs to the following completed update.
    wall_boundary = raw_time()
    monitor = TrainingMonitor(model, opt, args, run, progress) if (
        args.wandb_mode != 'disabled' or args.monitor_every > 0) else None
    with BatchPipeline(source, start_cursor=progress['cursor'], batch_size=args.batch_size,
                       grad_accum=args.grad_accum, updates=max(0, args.steps-progress['step']),
                       pad_id=pad_id, num_workers=args.num_workers,
                       prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers,
                       pin_memory=args.pin_memory and args.device == 'cuda') as batches, \
            (args.out/'metrics.jsonl').open('a') as log:
        print(json.dumps({'event':'input_pipeline', 'num_workers':args.num_workers,
                          'worker_pids':batches.worker_pids, 'prefetch_factor':args.prefetch_factor,
                          'persistent_workers':batches.persistent_workers,
                          'pin_memory':batches.pin_memory, 'non_blocking':args.non_blocking,
                          'start_cursor':progress['cursor'], 'microbatches':batches.microbatches}), flush=True)
        for step in range(progress['step']+1, args.steps+1):
            if progress['cursor'] + args.batch_size * args.grad_accum > source.length:
                print(json.dumps({'event':'ledger_exhausted','cursor':progress['cursor']}), flush=True)
                break
            factor, momentum = apply_optimizer_schedule(opt, step, args)
            if monitor:
                monitor.begin(step)
            if args.device == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
            started_raw, started_perf = raw_time(), time.perf_counter()
            loss, count, slots, grad_norm = update(model, opt, batches, progress['cursor'], args, pad_id,
                                                  diagnostics=monitor.diag if monitor else None)
            if args.device == 'cuda':
                end.record(); torch.cuda.synchronize()
            completed_raw = raw_time()
            elapsed, perf = completed_raw-started_raw, time.perf_counter()-started_perf
            wall_elapsed = completed_raw-wall_boundary
            wall_boundary = completed_raw
            progress = {'step':step, 'tokens_seen':progress['tokens_seen']+slots,
                        'cursor':progress['cursor']+args.batch_size*args.grad_accum}
            row = dict(progress, loss=float(loss), supervised_targets=count, slot_tokens=slots,
                raw_seconds=elapsed, perf_seconds=perf, wall_seconds=wall_elapsed,
                slot_tokens_per_second=slots/wall_elapsed,
                supervised_tokens_per_second=count/wall_elapsed,
                update_only_slot_tokens_per_second=slots/elapsed,
                update_only_supervised_tokens_per_second=count/elapsed,
                throughput_reporting_version=2, grad_norm=float(grad_norm), lr_factor=factor,
                muon_momentum=momentum,
                cuda_interval_seconds=begin.elapsed_time(end)/1000 if args.device=='cuda' else None,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None,
                warmup=args.benchmark and step <= args.warmup_updates)
            row['learning_rates'] = {group['name']:group['lr'] for group in opt.adam.param_groups}
            row['learning_rates']['normuon'] = opt.muon.param_groups[0]['lr'] if opt.muon else None
            display_window.append(row)
            encoded = json.dumps(row)
            log.write(encoded+'\n'); log.flush()
            if monitor:
                monitor.record(row, final=step == args.steps or stop[0])
            if step % args.log_every == 0 or step == args.steps or stop[0]:
                # A save after step 1000 must affect the next printed window,
                # even when step 1001 itself is not printed.
                window_wall = sum(r['wall_seconds'] for r in display_window)
                window_update = sum(r['raw_seconds'] for r in display_window)
                window_slots = sum(r['slot_tokens'] for r in display_window)
                window_targets = sum(r['supervised_targets'] for r in display_window)
                display = dict(row, log_window_updates=len(display_window),
                    log_window_wall_seconds=window_wall, log_window_slot_tokens=window_slots,
                    slot_tokens_per_second=window_slots/window_wall,
                    supervised_tokens_per_second=window_targets/window_wall,
                    update_only_slot_tokens_per_second=window_slots/window_update,
                    update_only_supervised_tokens_per_second=window_targets/window_update)
                print(json.dumps(display), flush=True)
                display_window.clear()
            if not row['warmup']:
                measured.append(row)
            if not args.benchmark and args.save_every and step % args.save_every == 0:
                save_checkpoint(args.out/f'step{step:08d}.pt', model, opt, **progress, contract=contract)
            if stop[0]:
                break
    if not args.benchmark:
        save_checkpoint(args.out/'latest.pt', model, opt, **progress, contract=contract)
        atomic_save({'schema':'rockso_hybrid_export_1', 'config':c.to_dict(), 'model':model.state_dict(),
                     'tokenizer':tok_receipt, 'attention_mode':args.attention_mode,
                     'source_sha256':source_receipt()}, args.out/'model.pt')
    # Include the final row's logging, worker shutdown, checkpoints and export.
    wall_seconds = sum(r['wall_seconds'] for r in measured) + (raw_time()-wall_boundary) if measured else 0.
    update_seconds = sum(r['raw_seconds'] for r in measured)
    summary = {'completed':progress, 'measured_steps':len(measured), 'benchmark':args.benchmark,
        'total_raw_seconds':sum(r['raw_seconds'] for r in measured),
        'total_wall_seconds':wall_seconds, 'throughput_reporting_version':2,
        'slot_tokens_per_second':sum(r['slot_tokens'] for r in measured)/wall_seconds if measured else None,
        'supervised_tokens_per_second':sum(r['supervised_targets'] for r in measured)/wall_seconds if measured else None,
        'update_only_slot_tokens_per_second':sum(r['slot_tokens'] for r in measured)/update_seconds if measured else None,
        'update_only_supervised_tokens_per_second':sum(r['supervised_targets'] for r in measured)/update_seconds if measured else None,
        'peak_allocated_bytes':max((r['peak_allocated_bytes'] or 0 for r in measured), default=0),
        'quality_evidence':False}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if monitor:
        monitor.finish('complete' if progress['step'] == args.steps else 'stopped', summary)
    print(json.dumps({'event':'finished', **summary}), flush=True)


if __name__ == '__main__':
    args = parser().parse_args()
    with gpu_ownership() if args.device == 'cuda' and args.check_gpu else contextlib.nullcontext():
        main(args)
