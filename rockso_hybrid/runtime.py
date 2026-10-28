import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import torch
from .vendor.normuon_optimizer_v3 import NorMuon


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def apply_fp8(model):
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    def selected(module, fqn):
        return isinstance(module, torch.nn.Linear) and any(fqn.endswith(s) for s in
            ("q_proj", "k_proj", "v_proj", "out_proj", "mlp.w1", "mlp.w2", "mlp.w3"))
    for name, module in model.named_modules():
        if selected(module, name) and (module.in_features % 16 or module.out_features % 16):
            raise ValueError(f"FP8 requires dimensions divisible by 16: {name}")
    names = [name for name, m in model.named_modules() if selected(m, name)]
    convert_to_float8_training(model, module_filter_fn=selected, config=Float8LinearConfig())
    return names


class HybridOptimizer:
    """Rockso NorMuon mathematics, standard PyTorch AdamW for small/tied parameters.

    No optional dependency can silently switch Adam to 8-bit. In legacy mode,
    Adam moments follow parameter dtype. With FP32 masters, BF16 embedding
    tables use FP32 optimizer weights and moments while their model weights
    remain BF16. NorMuon keeps its original BF16 momentum and FP32 per-neuron
    variance, with an additional FP32 master weight when requested. The explicit
    stochastic-rounding mode instead stores NorMuon updates directly into BF16;
    embedding optimizer weights and moments remain FP32.
    """
    def __init__(self, model, muon_lr=0.0029, adam_lr=0.0003, weight_decay=0.1, kind="normuon", fp32_master_weights=False, normuon_stochastic_rounding=False, ve_lr_mult=1.0, ve_weight_decay=None, residual_lr_mult=1.0, branch_lr_mult=4.0):
        muon, adam = model.parameter_groups()
        self.ve_lr_mult = float(ve_lr_mult)
        self.ve_weight_decay = float(weight_decay if ve_weight_decay is None else ve_weight_decay)
        if not math.isfinite(self.ve_lr_mult) or self.ve_lr_mult <= 0:
            raise ValueError("VE LR multiplier must be finite and positive")
        if not math.isfinite(self.ve_weight_decay) or self.ve_weight_decay < 0:
            raise ValueError("VE weight decay must be finite and nonnegative")
        if any(not math.isfinite(x) or x <= 0 for x in (residual_lr_mult, branch_lr_mult)):
            raise ValueError("Residual and branch LR multipliers must be finite and positive")
        if not any(n == "value_embed.weight" for n, _ in adam) and (self.ve_lr_mult != 1.0 or self.ve_weight_decay != weight_decay):
            raise ValueError("Separate VE settings require value_embed.weight in AdamW")
        if kind not in ("normuon", "adamw"):
            raise ValueError("Unknown optimizer")
        if kind == "normuon" and not next(model.parameters()).is_cuda:
            raise ValueError("The frozen NorMuon kernels require CUDA; use explicit adamw for CPU smoke tests")
        self.kind = kind
        self.fp32_master_weights = bool(fp32_master_weights)
        self.normuon_stochastic_rounding = bool(normuon_stochastic_rounding)
        if self.normuon_stochastic_rounding and (not self.fp32_master_weights or kind != "normuon"):
            raise ValueError("Stochastic NorMuon requires FP32 embedding masters and kind=normuon")
        self.adam_masters = []
        if kind == "adamw":
            adam = muon + adam
            muon = []
        self.names = {"muon": [n for n, _ in muon], "adam": [n for n, _ in adam]}
        self.muon = NorMuon([p for _, p in muon], lr=muon_lr, weight_decay=weight_decay,
                           momentum=0.95, beta2=0.95, num_polar_iters=5,
                           fp32_master_weights=self.fp32_master_weights and not self.normuon_stochastic_rounding,
                           stochastic_rounding=self.normuon_stochastic_rounding) if muon else None
        if self.fp32_master_weights:
            updated = []
            for name, p in adam:
                if p.dtype == torch.bfloat16:
                    master = torch.nn.Parameter(p.detach().float().clone())
                    self.adam_masters.append((name, p, master))
                    updated.append((name, master))
                else:
                    updated.append((name, p))
            adam = updated
        groups = [
            ("adam_decay", [(n, p) for n, p in adam if p.ndim >= 2 and n != "value_embed.weight"], weight_decay, 1.0),
            ("adam_no_decay", [(n, p) for n, p in adam if p.ndim < 2 and "_lambda_" not in n], 0.0, 1.0),
        ]
        value_embed = [(n, p) for n, p in adam if n == "value_embed.weight"]
        if value_embed:
            groups.append(("value_embed", value_embed, self.ve_weight_decay, self.ve_lr_mult))
        for group_name, pattern, multiplier in (("residual_multipliers", ".resid_lambda_", residual_lr_mult),
                                                ("branch_multipliers", ".post_lambda_", branch_lr_mult)):
            parameters = [(n, p) for n, p in adam if pattern in n]
            if parameters:
                if any(p.dtype != torch.float32 or p.ndim != 0 for _, p in parameters):
                    raise ValueError("Learned residual multipliers must be scalar FP32 parameters")
                groups.append((group_name, parameters, 0.0, float(multiplier)))
        self.adam_group_names = {name: [n for n, _ in parameters] for name, parameters, _, _ in groups}
        self.adam = torch.optim.AdamW([
            {"params": [p for _, p in parameters], "name": name, "weight_decay": decay,
             "lr_multiplier": multiplier, "lr": adam_lr * multiplier}
            for name, parameters, decay, multiplier in groups
        ], lr=adam_lr, betas=(0.9, 0.95),
            fused=next(model.parameters()).device.type == "cuda")
        self.param_groups = (self.muon.param_groups if self.muon else []) + self.adam.param_groups
        self.base_lrs = (muon_lr, adam_lr)

    def zero_grad(self, set_to_none=True):
        if self.muon:
            self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)
        for _, p, _ in self.adam_masters:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        if self.muon:
            self.muon.step()
        for _, p, master in self.adam_masters:
            master.grad = p.grad.float() if p.grad is not None else None
        self.adam.step()
        for _, p, master in self.adam_masters:
            p.copy_(master)

    def scale_lr(self, factor):
        if not math.isfinite(factor) or factor < 0:
            raise ValueError("LR schedule factor must be finite and nonnegative")
        if self.muon:
            for group in self.muon.param_groups:
                group["lr"] = self.base_lrs[0] * factor
        for group in self.adam.param_groups:
            group["lr"] = self.base_lrs[1] * group["lr_multiplier"] * factor

    def group_receipt(self):
        return [{"name": group["name"], "parameter_names": self.adam_group_names[group["name"]],
                 "parameters": sum(p.numel() for p in group["params"]),
                 "peak_lr": self.base_lrs[1] * group["lr_multiplier"],
                 "lr_multiplier": group["lr_multiplier"], "weight_decay": group["weight_decay"],
                 "betas": group["betas"]} for group in self.adam.param_groups]

    def state_dict(self):
        return {"schema": 5, "kind": self.kind, "names": self.names, "base_lrs": self.base_lrs,
                "ve_lr_mult": self.ve_lr_mult, "ve_weight_decay": self.ve_weight_decay,
                "adam_group_names": self.adam_group_names,
                "fp32_master_weights": self.fp32_master_weights,
                "normuon_stochastic_rounding": self.normuon_stochastic_rounding,
                "adam_master_weights": {name: master.detach() for name, _, master in self.adam_masters},
                "muon": self.muon.state_dict() if self.muon else None, "adam": self.adam.state_dict()}

    def load_state_dict(self, state):
        # Resume restores numerical state; current launcher settings select the next updates.
        # Only incompatible state layouts/types are rejected.
        if state["schema"] != 5 or state.get("fp32_master_weights") != self.fp32_master_weights or state.get("normuon_stochastic_rounding") != self.normuon_stochastic_rounding or state["kind"] != self.kind or state["names"] != self.names or state.get("adam_group_names") != self.adam_group_names:
            raise ValueError("Optimizer schema, parameter ordering or precision mode differs")
        adam_options = [{k: v for k, v in g.items() if k != "params"} for g in self.adam.param_groups]
        muon_options = [{k: v for k, v in g.items() if k != "params"} for g in self.muon.param_groups] if self.muon else []
        saved_groups = state["adam"]["param_groups"]
        if len(saved_groups) != len(self.adam.param_groups):
            raise ValueError("AdamW group count differs")
        for saved, live in zip(saved_groups, self.adam.param_groups):
            if saved.get("name") != live.get("name"):
                raise ValueError("AdamW group identity differs")
            if len(saved["params"]) != len(live["params"]):
                raise ValueError("AdamW group parameter count differs")
        if self.muon:
            if state["muon"] is None:
                raise ValueError("NorMuon state is missing")
            if len(state["muon"]["param_groups"]) != len(self.muon.param_groups):
                raise ValueError("NorMuon group count differs")
            for group, live in zip(state["muon"]["param_groups"], self.muon.param_groups):
                if len(group["params"]) != len(live["params"]):
                    raise ValueError("NorMuon parameter count differs")
                if group.get("fp32_master_weights", False) != (self.fp32_master_weights and not self.normuon_stochastic_rounding) or group.get("stochastic_rounding", False) != self.normuon_stochastic_rounding:
                    raise ValueError("NorMuon weight-update mode differs")
            self.muon.load_state_dict(state["muon"])
            for group, options in zip(self.muon.param_groups, muon_options):
                group.update(options)
            # Optimizer.load_state_dict otherwise casts this FP32 accumulator to
            # a BF16 parameter's dtype, silently changing the resumed trajectory.
            for saved_group, live_group in zip(state["muon"]["param_groups"], self.muon.param_groups):
                for saved_id, p in zip(saved_group["params"], live_group["params"]):
                    saved = state["muon"]["state"].get(saved_id, {})
                    for key in ("second_momentum_buffer", "master_weight"):
                        if key in saved:
                            self.muon.state[p][key] = saved[key].to(device=p.device, dtype=torch.float32).clone()
                    if self.fp32_master_weights and not self.normuon_stochastic_rounding and saved:
                        if "master_weight" not in saved or not torch.equal(p, self.muon.state[p]["master_weight"].to(p.dtype)):
                            raise ValueError("NorMuon master weight is missing or disagrees with visible checkpoint")
        self.adam.load_state_dict(state["adam"])
        for group, options in zip(self.adam.param_groups, adam_options):
            group.update(options)
        self.param_groups = (self.muon.param_groups if self.muon else []) + self.adam.param_groups
        masters = state.get("adam_master_weights", {})
        if set(masters) != {name for name, _, _ in self.adam_masters}:
            raise ValueError("Adam master parameter identity differs")
        with torch.no_grad():
            for name, p, master in self.adam_masters:
                master.copy_(masters[name].to(device=master.device, dtype=torch.float32))
                if not torch.equal(p, master.to(p.dtype)):
                    raise ValueError("Adam master weight disagrees with visible checkpoint")


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("wb") as f:
            torch.save(value, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, *, step, tokens_seen, cursor, contract):
    if model.config.tie_embeddings and model.embed.weight is not model.lm_head.weight:
        raise RuntimeError("Tied embedding identity was broken")
    state = {"schema": "rockso_hybrid_checkpoint_1", "config": model.config.to_dict(),
             "contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "step": step, "tokens_seen": tokens_seen, "cursor": cursor,
             "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                     "torch": torch.get_rng_state(),
                     "cuda": torch.cuda.get_rng_state_all() if model.embed.weight.is_cuda else None}}
    atomic_save(state, path)


def validate_resume_contract(saved, current):
    """Check token/data identity, not user-editable training hyperparameters or source hashes."""
    old_tok, new_tok = saved.get("tokenizer"), current.get("tokenizer")
    if (old_tok is None) != (new_tok is None):
        raise ValueError("Checkpoint tokenizer contract is missing")
    if old_tok is not None:
        for key in ("vocab_size", "pad_id", "eos_id"):
            if old_tok.get(key) != new_tok.get(key):
                raise ValueError(f"Checkpoint tokenizer {key} differs")
        if old_tok.get("files") != new_tok.get("files"):
            raise ValueError("Checkpoint tokenizer files differ")
    old_data, new_data = saved.get("data", {}), current.get("data", {})
    for key in ("kind", "dataset_fingerprint", "split", "rows", "sequence_length", "vocab_size",
                "order_length", "order", "ledger_sha256"):
        if old_data.get(key) != new_data.get(key):
            raise ValueError(f"Checkpoint data/ledger {key} differs; saved cursor cannot address another order")


def load_checkpoint(path, model, optimizer, contract):
    # Only trusted, locally produced checkpoints. Legacy weights use a separate importer.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if state.get("schema") != "rockso_hybrid_checkpoint_1":
        raise ValueError("Not a hybrid resume checkpoint; legacy import is weights-only")
    if state["config"] != model.config.to_dict():
        raise ValueError("Checkpoint architecture differs")
    validate_resume_contract(state["contract"], contract)
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    if model.embed.weight.is_cuda:
        if state["rng"]["cuda"] is None:
            raise ValueError("CUDA replay state is missing")
        torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    return {k: state[k] for k in ("step", "tokens_seen", "cursor")}


def import_rockso_weights(model, state):
    """Explicit neutral bridge import. No optimizer/scheduler/ledger continuation."""
    c = model.config
    if c.local_window or c.global_nope or c.rope_fraction != 1 or c.rope_base != 10000 or c.rope_math != "rockso" or c.sigmoid_logit_cap or c.attention_scale is not None:
        raise ValueError("Weights import requires Rockso attention and output mathematics")
    target = model.state_dict()
    new_names = {n for n in target if any(s in n for s in
                 ("attention_gate", "ve_gate", "value_embed", "xsa_alpha", "mudd", "_lambda_"))}
    expected = set(target) - new_names
    if set(state) != expected:
        raise ValueError(f"Legacy key mismatch: missing={sorted(expected-set(state))}, extra={sorted(set(state)-expected)}")
    for name in expected:
        if target[name].shape != state[name].shape:
            raise ValueError(f"Legacy shape mismatch: {name}")
    if c.tie_embeddings and not torch.equal(state["embed.weight"], state["lm_head.weight"]):
        raise ValueError("Legacy tied weights disagree")
    # Only permit the freshly initialized neutral modules; caller cannot accidentally
    # label an already trained hybrid as a neutral warm start.
    for name in new_names:
        tensor = target[name]
        if name.endswith("mudd.reduce.weight"):
            continue
        if "_lambda_" in name:
            if not torch.equal(tensor, torch.ones_like(tensor)):
                raise ValueError(f"Residual multiplier is not neutral: {name}")
            continue
        if name.endswith("mudd.expand.bias"):
            expected_bias = torch.tensor([0, 0, 1, 0, 0, 1], device=tensor.device, dtype=tensor.dtype)
            if not torch.equal(tensor, expected_bias):
                raise ValueError("MUDD is not neutral")
        elif torch.count_nonzero(tensor):
            raise ValueError(f"New parameter is not neutral: {name}")
    merged = dict(target)
    merged.update(state)
    model.load_state_dict(merged, strict=True)
    return {"imported_tensors": len(expected), "new_tensors": len(new_names), "optimizer_imported": False}


@contextlib.contextmanager
def gpu_ownership():
    import fcntl
    lock_path = Path(__file__).resolve().parents[1] / ".gpu.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                 "--format=csv,noheader"], text=True, capture_output=True, check=True)
        owners = [line for line in result.stdout.splitlines() if line.strip()]
        if owners:
            raise RuntimeError("GPU already has a compute owner: " + "; ".join(owners))
        yield


def raw_time():
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def source_receipt():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "rockso_hybrid").rglob("*.py")) + [root / name for name in
        ("train.py", "telemetry.py", "monitoring_io.py", "wandb_publisher.py")]
    return {str(p.relative_to(root)): sha256(p) for p in paths if p.exists()}
