#!/usr/bin/env python3
"""Load native hybrid saves into Rockso's unchanged frozen PPL evaluation routine.

This packaged entrypoint supports native Rockso V2 saves. Hybrid evaluation
keeps its native mixed precision, full architecture, and shifted response-only
cross entropy. No optimizer, FP8 conversion, or training state is restored.
"""
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ppl_reference as reference
from rockso_hybrid import ModelConfig, RocksoHybrid
from rockso_hybrid.data import tokenizer_receipt
from rockso_hybrid.loss import causal_loss
from rockso_hybrid.masking import build_masks
from rockso_hybrid.runtime import sha256

_original_load = reference.load_prefixlm_model
_original_masks = reference._resolve_masks


class HybridEvalModel(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    def forward(self, input_ids, position_ids, labels=None, attn_mask=None,
                block_mask=None):
        if labels is None:
            raise ValueError("This PPL adapter requires response labels")
        masks = block_mask if block_mask is not None else attn_mask
        backend = "flex" if block_mask is not None else "sdpa"
        self.native.backend = backend
        if masks is None and self.native.config.local_window:
            # The frozen scorer skips mask construction in causal mode.
            ends = torch.full((input_ids.shape[0], 1), input_ids.shape[1],
                              dtype=torch.int32, device=input_ids.device)
            masks = build_masks(ends, torch.zeros_like(input_ids), mode="causal",
                                backend=backend,
                                local_window=self.native.config.local_window)
        # The frozen scorer wraps its legacy model in autocast. The hybrid
        # trainer uses explicit dtypes: autocast would downcast its FP32 routers.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            hidden = self.native(input_ids, position_ids, masks, return_hidden=True)
            loss = causal_loss(hidden, self.native.lm_head.weight, labels,
                               backend="reference",
                               sigmoid_cap=self.native.config.sigmoid_logit_cap)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite hybrid PPL loss; no documents were silently dropped")
        return {"loss": loss}


class HybridMasks:
    def __init__(self, config):
        self.config = config


def resolve_masks(trainer, args, span_ends, metadata_mask, seq_len):
    if not isinstance(trainer, HybridMasks):
        return _original_masks(trainer, args, span_ends, metadata_mask, seq_len)
    backend = "flex" if args.prefixlm_impl == "flex" else "sdpa"
    masks = build_masks(span_ends, metadata_mask, mode=args.attn_mask_mode,
                        backend=backend, local_window=trainer.config.local_window)
    return (None, masks) if backend == "flex" else (masks, None)


def load_model(model_path, tokenizer_path, device, bf16, trainer_module):
    path = Path(reference._resolve_raw_checkpoint(model_path)).resolve()
    # Locally produced trusted training checkpoints include NumPy RNG state.
    # mmap leaves unused optimizer tensors on disk.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if state.get("schema") != "rockso_hybrid_checkpoint_1":
        del state
        raise ValueError("This repository evaluates native rockso_hybrid_checkpoint_1 saves only")

    contract = state["contract"]
    config = ModelConfig.from_dict(state["config"])
    implementation = {}
    for name in ("rockso_hybrid/model.py", "rockso_hybrid/config.py",
                 "rockso_hybrid/masking.py", "rockso_hybrid/loss.py", "rockso_hybrid/residual.py"):
        digest = sha256(ROOT / name)
        if contract["source_sha256"].get(name) != digest:
            raise ValueError(f"Checkpoint model implementation differs: {name}")
        implementation[name] = digest
    _, tokenizer = tokenizer_receipt(tokenizer_path, config.vocab_size)
    # Location may change; tokenizer content and all numerical metadata must match.
    actual_tokenizer = {k: v for k, v in tokenizer.items() if k != "path"}
    saved_tokenizer = {k: v for k, v in contract["tokenizer"].items() if k != "path"}
    if actual_tokenizer != saved_tokenizer:
        raise ValueError("Tokenizer differs from the hybrid checkpoint")

    dtype = torch.bfloat16 if bf16 else torch.float32
    model = RocksoHybrid(config).to(device=device, dtype=dtype).preserve_precision()
    model.load_state_dict(state["model"], strict=True)
    model.configure_runtime(backend="sdpa", checkpointing=False,
                            native_gqa=contract["arguments"].get("native_gqa", False))
    model.eval()
    info = {
        "kind": "rockso_hybrid_checkpoint_1",
        "path": str(path),
        "sha256": sha256(path),
        "step": int(state["step"]),
        "tokens_seen": int(state["tokens_seen"]),
        "cursor": int(state["cursor"]) if state.get("cursor") is not None else None,
        "trainer_module": "rockso_hybrid",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "config": config.to_dict(),
        "checkpoint_attn_mask_mode": contract["arguments"]["attention_mode"],
        "checkpoint_prefixlm_impl": contract["arguments"]["backend"],
        "checkpoint_fp8": bool(contract["arguments"]["fp8"]),
        "eval_fp8": False,
        "eval_dtype": str(dtype),
        "eval_fp32_components": "norms, gates, XSA coefficients, MUDD routers",
        "eval_autocast": False,
        "model_source_sha256": implementation,
        "evaluator_sha256": sha256(Path(reference.__file__)),
        "adapter_sha256": sha256(Path(__file__)),
    }
    print(f"[model] Hybrid step={info['step']:,} tokens={info['tokens_seen']:,} "
          f"parameters={info['parameter_count']:,}; strict load, {dtype}, FP8 off", flush=True)
    return HybridEvalModel(model).eval(), HybridMasks(config), info


def main():
    reference.load_prefixlm_model = load_model
    reference._resolve_masks = resolve_masks
    reference.main()


if __name__ == "__main__":
    main()
