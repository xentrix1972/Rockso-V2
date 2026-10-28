#!/usr/bin/env python3
"""
PrefixLM-aware plain LM perplexity eval for TRAINER_MAIN_dense_prefixlm_toy.py.

This mirrors eval_lm_ppl.py's external-corpus workflow, but loads the raw model
with TRAINER_MAIN_dense_prefixlm_toy.py so PrefixLM checkpoints use the right
architecture contract.

For the usual HRM LM-PPL command with no --meco-tag, the external corpus has no
instruction prefix. metadata_mask is therefore all response tokens and PrefixLM
is exactly equivalent to causal attention. The script takes that faithful fast
path automatically.

If --meco-tag is supplied, the tag tokens are treated as a bidirectional prefix
and are not scored.
"""

import argparse
import importlib
import inspect
import json
import math
import os
import sys
import time
import types
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


from eval_lm_ppl import (  # noqa: E402
    MECO_VARIANTS,
    PRESETS,
    build_sequences,
    load_records,
    print_result,
)


OLD_HRM_DATA_DIR = "eval_data/legacy_hrm"
OLD_HRM_LEDGER = f"{OLD_HRM_DATA_DIR}/ledger.json"


def _looks_like_raw_checkpoint(path: str) -> bool:
    p = Path(path)
    if p.suffix == ".pt":
        return True
    return (
        (p / "latest.pt").exists()
        or (p / "checkpoints" / "latest.pt").exists()
        or (p / ".latest").exists()
    )


def _resolve_raw_checkpoint(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    if p.suffix == ".pt":
        siblings = []
        if p.parent.exists():
            siblings = sorted(x.name for x in p.parent.glob("*.pt"))[:12]
        hint = f" Nearby .pt files: {siblings}" if siblings else ""
        raise FileNotFoundError(f"Checkpoint file not found: {path}.{hint}")
    candidates = [
        p / "latest.pt",
        p / "checkpoints" / "latest.pt",
    ]
    latest_pointer = p / ".latest"
    if latest_pointer.exists():
        pointed = Path(latest_pointer.read_text(encoding="utf-8").strip())
        candidates.insert(0, pointed)
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(f"No raw checkpoint found under {path}")


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in sd.items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod.") :]
        out[key] = value
    return out


def _install_eval_torchao_stubs() -> None:
    """Avoid importing TorchAO during eval-only trainer import.

    The PrefixLM trainer imports torchao.float8 at module import time, but this
    diagnostic always constructs the model with use_fp8=False and loads plain
    checkpoint tensors. Some local envs can segfault while importing TorchAO, so
    provide the tiny surface the trainer needs to finish importing.
    """
    if "torchao.float8" in sys.modules:
        return

    torchao_mod = sys.modules.get("torchao")
    if torchao_mod is None:
        torchao_mod = types.ModuleType("torchao")
        torchao_mod.__path__ = []
        torchao_mod.__version__ = "eval-stub"
        sys.modules["torchao"] = torchao_mod

    float8_mod = types.ModuleType("torchao.float8")

    def _unavailable(*_args, **_kwargs):
        raise RuntimeError("TorchAO FP8 is disabled in eval_prefixlm_ppl.py")

    float8_mod.convert_to_float8_training = _unavailable
    sys.modules["torchao.float8"] = float8_mod
    setattr(torchao_mod, "float8", float8_mod)


def load_prefixlm_model(model_path, tokenizer_path, device, bf16, trainer_module):
    raise RuntimeError("Use eval_prefixlm_ppl.py, which installs the native Rockso V2 loader")


def _prefix_metadata_for_window(
    length: int,
    begin_loc: int,
    global_prefix_len: int,
    device: str,
) -> torch.Tensor:
    if global_prefix_len <= 0:
        return torch.zeros((1, length), dtype=torch.int8, device=device)
    positions = torch.arange(begin_loc, begin_loc + length, device=device)
    return (positions < global_prefix_len).to(torch.int8).unsqueeze(0)


def _resolve_masks(trainer, args, span_ends, metadata_mask, seq_len: int):
    if args.attn_mask_mode == "causal":
        return None, None
    if args.prefixlm_impl == "flex":
        return None, trainer.build_flex_block_mask(
            span_ends, metadata_mask, seq_len, args.attn_mask_mode
        )
    allow = trainer.build_attn_allow_mask(
        span_ends, metadata_mask, seq_len, args.attn_mask_mode
    )
    return allow.unsqueeze(1), None


def _prefixlm_collapses_to_causal(metadata_mask: torch.Tensor) -> bool:
    return bool(metadata_mask is None or not metadata_mask.bool().any().item())


def install_eval_flex_dtype_guard(trainer) -> None:
    """Flex requires q/k/v to share dtype; VR scalars can promote v in eval.

    This patches only the imported trainer module used by this diagnostic. The
    training file on disk is untouched.
    """
    flex_fn = getattr(trainer, "flex_attention", None)
    if flex_fn is None or getattr(flex_fn, "_prefixlm_eval_dtype_guard", False):
        return

    def guarded_flex_attention(q, k, v, *args, **kwargs):
        target = q.dtype
        if k.dtype != target:
            k = k.to(target)
        if v.dtype != target:
            v = v.to(target)
        return flex_fn(q, k, v, *args, **kwargs)

    guarded_flex_attention._prefixlm_eval_dtype_guard = True
    trainer.flex_attention = guarded_flex_attention


def compute_prefixlm_ppl(
    model,
    trainer,
    tokenizer,
    records: List[Dict],
    meco_tag: Optional[str],
    mode: str,
    max_length: int,
    stride: int,
    device: str,
    bf16: bool,
    max_eval_tokens: Optional[int],
    args,
) -> Dict:
    model.eval()
    if args.prefixlm_impl == "flex":
        install_eval_flex_dtype_guard(trainer)
    dtype = torch.bfloat16 if bf16 else torch.float32
    autocast_enabled = device.startswith("cuda") and bf16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype, enabled=autocast_enabled)

    prefix_len = 0
    if meco_tag:
        prefix_len = len(tokenizer.encode(f"{meco_tag}\n", add_special_tokens=False))

    seqs = build_sequences(tokenizer, records, meco_tag=meco_tag, mode=mode)
    total_loss = 0.0
    total_tokens = 0
    windows = 0
    windows_masked = 0
    windows_collapsed = 0

    desc = f"{mode} PrefixLM={args.attn_mask_mode}/{args.prefixlm_impl} MeCo={meco_tag or 'None'}"
    with torch.no_grad(), autocast_ctx:
        for ids in tqdm(seqs, desc=desc):
            if len(ids) <= 1:
                continue
            prev_end_loc = 0
            for begin_loc in range(0, len(ids), stride):
                end_loc = min(begin_loc + max_length, len(ids))
                trg_len = end_loc - prev_end_loc
                if trg_len <= 0:
                    break

                window_ids = ids[begin_loc:end_loc]
                seq_len = len(window_ids)
                x = torch.tensor([window_ids], dtype=torch.long, device=device)
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                labels = x.clone()

                if begin_loc > 0:
                    labels[:, :-trg_len] = -100
                elif prefix_len > 0:
                    labels[:, : min(prefix_len, labels.size(1))] = -100

                if getattr(args, "context_as_prefix", False) and begin_loc > 0:
                    # Unscored overlap region acts as a bidirectional prefix:
                    # the training-shaped mask for a PrefixLM model on raw text.
                    ctx = seq_len - trg_len
                    metadata_mask = (
                        (torch.arange(seq_len, device=device) < ctx)
                        .to(torch.int8).unsqueeze(0)
                    )
                else:
                    metadata_mask = _prefix_metadata_for_window(
                        seq_len, begin_loc, prefix_len, device
                    )
                span_ends = torch.tensor([[seq_len]], dtype=torch.int32, device=device)
                if args.attn_mask_mode == "causal" or _prefixlm_collapses_to_causal(metadata_mask):
                    attn_mask, block_mask = None, None
                    windows_collapsed += 1
                else:
                    attn_mask, block_mask = _resolve_masks(
                        trainer, args, span_ends, metadata_mask, seq_len
                    )
                    windows_masked += 1

                # Match eval_lm_ppl.py's weighting for apples-to-apples trend
                # comparison with the old HRM run.
                valid = int((labels != -100).sum().item())
                if valid > 0:
                    out = model(
                        input_ids=x,
                        position_ids=position_ids,
                        labels=labels,
                        attn_mask=attn_mask,
                        block_mask=block_mask,
                    )
                    loss_t = out["loss"] if isinstance(out, dict) else out.loss
                    loss = float(loss_t.detach().float().cpu().item())
                    if math.isfinite(loss):
                        total_loss += loss * valid
                        total_tokens += valid
                        windows += 1

                prev_end_loc = end_loc
                if max_eval_tokens is not None and total_tokens >= max_eval_tokens:
                    break
                if end_loc >= len(ids):
                    break

            if max_eval_tokens is not None and total_tokens >= max_eval_tokens:
                break

    if args.attn_mask_mode != "causal":
        if windows_masked == 0:
            effective_attn = "causal (ALL windows collapsed)"
            print(
                f"[WARN] requested attn_mask_mode={args.attn_mask_mode} but every "
                f"window collapsed to causal (no prefix tokens). This eval measured "
                f"CAUSAL attention. Use --meco-tag, --context-as-prefix, or "
                f"--preset hrm for a real {args.attn_mask_mode} measurement.",
                flush=True,
            )
        elif windows_collapsed > 0:
            effective_attn = f"{args.attn_mask_mode} ({windows_masked} masked / {windows_collapsed} causal)"
        else:
            effective_attn = args.attn_mask_mode
    else:
        effective_attn = "causal"

    avg_loss = total_loss / max(1, total_tokens)
    total_words = sum(r["words"] for r in records)
    total_bytes = sum(r["bytes"] for r in records)
    if max_eval_tokens is None:
        word_ppl = math.exp(min(total_loss / max(1, total_words), 20.0))
        byte_ppl = math.exp(min(total_loss / max(1, total_bytes), 20.0))
        bits_per_byte = total_loss / max(1, total_bytes) / math.log(2)
    else:
        word_ppl = float("nan")
        byte_ppl = float("nan")
        bits_per_byte = float("nan")

    return {
        "token_perplexity": float(math.exp(min(avg_loss, 20.0))),
        "ppl": float(math.exp(min(avg_loss, 20.0))),
        "word_perplexity": float(word_ppl),
        "byte_perplexity": float(byte_ppl),
        "bits_per_byte": float(bits_per_byte),
        "avg_loss": avg_loss,
        "tokens": total_tokens,
        "words": total_words,
        "bytes": total_bytes,
        "windows": windows,
        "mode": mode,
        "meco_tag": meco_tag,
        "attn_mask_mode": args.attn_mask_mode,
        "effective_attn_mask": effective_attn,
        "windows_masked": windows_masked,
        "windows_collapsed_to_causal": windows_collapsed,
        "prefixlm_impl": args.prefixlm_impl,
        "partial_eval": max_eval_tokens is not None,
    }


# --------------------------------------------------------------------------
# --preset hrm: held-out in-distribution single-doc scoring (the arbiter).
# Pulls blocks from the TAIL of the ledger order (unseen until the very end
# of the 175K-step schedule), splits each into documents via span_ends +
# metadata_mask (same boundary definition as hrm_batch_adapter), and scores
# response-only NLL one doc per sequence: identical information set for both
# models, each under its training-consistent mask. Per-condition breakdown
# from span_meta.
# --------------------------------------------------------------------------

def _split_block_docs(row, include_legacy_fragments: bool = False, *, boq_id=6, eoa_id=11):
    ids = list(row["input_ids"])
    mask = list(row["metadata_mask"])
    ends = [int(e) for e in row.get("span_ends", []) if int(e) > 0]
    metas = list(row.get("span_meta", []))
    docs = []
    start = 0
    for i, end in enumerate(ends):
        if end <= start:
            continue
        d_ids, d_mask = ids[start:end], mask[start:end]
        try:
            prefix = next(j for j, m in enumerate(d_mask) if m == 0)
        except StopIteration:
            start = end
            continue  # no response tokens -> nothing to score
        if prefix == 0 and not include_legacy_fragments:
            start = end
            continue
        if len(d_ids) - prefix <= 1:
            start = end
            continue
        starts_boq = bool(d_ids and d_ids[0] == boq_id)
        ends_eoa = bool(d_ids and d_ids[-1] == eoa_id)
        if starts_boq and ends_eoa and prefix > 0:
            fragment_kind = "complete"
        elif not starts_boq and prefix == 0:
            fragment_kind = "prefixless_response_continuation"
        elif not starts_boq:
            fragment_kind = "mid_prompt_or_wrapper_continuation"
        elif not ends_eoa:
            fragment_kind = "left_truncated_response"
        else:
            fragment_kind = "other_incomplete"
        docs.append({
            "ids": d_ids,
            "mask": d_mask,
            "condition": str(metas[i]) if i < len(metas) else "unknown",
            "fragment_kind": fragment_kind,
        })
        start = end
    return docs


def load_hrm_docs(args):
    import numpy as np
    from datasets import load_from_disk
    with open(args.hrm_ledger, "r", encoding="utf-8") as f:
        meta = json.load(f)
    import hashlib
    token_file = Path(args.tokenizer) / "tokenizer.json"
    token_sha = hashlib.sha256(token_file.read_bytes()).hexdigest()
    if meta.get("tokenizer_sha256") and token_sha != meta["tokenizer_sha256"]:
        raise ValueError("PPL corpus tokenizer does not match --tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    boq_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eoa_id = tokenizer.convert_tokens_to_ids("<|box_end|>")
    args.hrm_reference_meta = meta
    if args.old and meta.get("kind") != "rockso_legacy_damage_ppl_v1":
        raise RuntimeError(
            f"--old requires a rockso_legacy_damage_ppl_v1 ledger, got {meta.get('kind')!r}"
        )
    order_path = meta.get("order_path", "order.npy")
    if not os.path.isabs(order_path):
        order_path = os.path.join(os.path.dirname(args.hrm_ledger), order_path)
    order = np.load(order_path, mmap_mode="r")
    if args.hrm_num_blocks > len(order):
        raise ValueError(
            f"Requested {args.hrm_num_blocks:,} HRM probe blocks, but this reference "
            f"contains only {len(order):,}; supply a larger --hrm-data-dir/--hrm-ledger "
            "pair instead of silently scoring fewer blocks."
        )
    block_idx = [int(b) for b in order[-args.hrm_num_blocks:]]
    ds = load_from_disk(args.hrm_data_dir)["train"]
    docs = []
    for b in block_idx:
        docs.extend(_split_block_docs(
            ds[int(b)], include_legacy_fragments=bool(args.old), boq_id=boq_id, eoa_id=eoa_id
        ))
    fragment_counts = Counter(doc["fragment_kind"] for doc in docs)
    args.hrm_fragment_counts = dict(sorted(fragment_counts.items()))
    lane = "legacy_damage" if args.old else "atomic_clean"
    print(
        f"[hrm] blocks={len(block_idx)} (ledger tail) docs={len(docs)} "
        f"lane={lane} data={args.hrm_data_dir}",
        flush=True,
    )
    print(f"[hrm] fragment_mix={args.hrm_fragment_counts}", flush=True)
    return docs


def compute_hrm_packed_ppl(model, trainer, docs, device, bf16, args):
    model.eval()
    if args.prefixlm_impl == "flex":
        install_eval_flex_dtype_guard(trainer)
    dtype = torch.bfloat16 if bf16 else torch.float32
    autocast_ctx = torch.amp.autocast(
        "cuda", dtype=dtype, enabled=device.startswith("cuda") and bf16
    )
    agg = {}
    with torch.no_grad(), autocast_ctx:
        for doc in tqdm(docs, desc=f"hrm single-doc attn={args.attn_mask_mode}"):
            T = len(doc["ids"])
            x = torch.tensor([doc["ids"]], dtype=torch.long, device=device)
            meta_t = torch.tensor([doc["mask"]], dtype=torch.int8, device=device)
            pos = torch.arange(T, device=device).unsqueeze(0)
            labels = x.clone()
            labels[meta_t == 1] = -100
            span_ends = torch.tensor([[T]], dtype=torch.int32, device=device)
            if args.attn_mask_mode == "causal":
                attn_mask, block_mask = None, None
            else:
                attn_mask, block_mask = _resolve_masks(
                    trainer, args, span_ends, meta_t, T
                )
            valid = int((labels[:, 1:] != -100).sum().item())
            if valid <= 0:
                continue
            out = model(input_ids=x, position_ids=pos, labels=labels,
                        attn_mask=attn_mask, block_mask=block_mask)
            loss_t = out["loss"] if isinstance(out, dict) else out.loss
            loss = float(loss_t.detach().float().cpu().item())
            if not math.isfinite(loss):
                continue
            cond = doc["condition"]
            a = agg.setdefault(cond, {"loss": 0.0, "tokens": 0, "docs": 0})
            a["loss"] += loss * valid
            a["tokens"] += valid
            a["docs"] += 1

    result = {"attn_mask_mode": args.attn_mask_mode,
              "prefixlm_impl": args.prefixlm_impl,
              "eval_lane": "legacy_damage" if args.old else "atomic_clean",
              "old": bool(args.old),
              "hrm_data_dir": args.hrm_data_dir,
              "hrm_ledger": args.hrm_ledger,
              "hrm_num_blocks": args.hrm_num_blocks,
              "fragment_counts": getattr(args, "hrm_fragment_counts", {}),
              "per_condition": {}}
    tot_loss, tot_tokens, tot_docs = 0.0, 0, 0
    for cond, a in sorted(agg.items()):
        al = a["loss"] / max(1, a["tokens"])
        result["per_condition"][cond] = {
            "avg_loss": al, "ppl": float(math.exp(min(al, 20.0))),
            "tokens": a["tokens"], "docs": a["docs"],
        }
        tot_loss += a["loss"]; tot_tokens += a["tokens"]; tot_docs += a["docs"]
    al = tot_loss / max(1, tot_tokens)
    result.update({"avg_loss": al, "ppl": float(math.exp(min(al, 20.0))),
                   "tokens": tot_tokens, "docs": tot_docs})
    print(f"  OVERALL  resp_ppl={result['ppl']:9.3f} loss={al:.5f} "
          f"tokens={tot_tokens:,} docs={tot_docs}", flush=True)
    for cond, r in result["per_condition"].items():
        print(f"  {cond:<22} resp_ppl={r['ppl']:9.3f} loss={r['avg_loss']:.5f} "
              f"tokens={r['tokens']:,} docs={r['docs']}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(
        description="PrefixLM-aware WikiText/PTB perplexity eval for Rockso checkpoints"
    )
    ap.add_argument("model_path", help="Raw .pt checkpoint, run dir, or checkpoints dir")
    ap.add_argument("--preset", choices=sorted(PRESETS) + ["hrm"], default="hrm")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--preprocess", choices=["none", "wikitext"], default="none")
    ap.add_argument("--cache-dir", default=".cache/huggingface/datasets")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--trainer-module", default="auto",
                    help="Trainer module for raw checkpoint loading; auto chooses fixed rotate-half trainer for fixed/marked runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--mode", choices=["concat", "documents"], default="concat")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-eval-tokens", type=int, default=None)
    ap.add_argument("--meco-tag", default=None)
    ap.add_argument("--no-meco", action="store_true")
    ap.add_argument("--compare-all", action="store_true")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--attn-mask-mode", choices=["causal", "doc_causal", "prefixlm"], default="prefixlm")
    ap.add_argument("--prefixlm-impl", choices=["flex", "sdpa_bool"], default="sdpa_bool")
    ap.add_argument("--context-as-prefix", action="store_true",
                    help="treat each window's unscored overlap as bidirectional prefix")
    ap.add_argument("--old", action="store_true",
                    help="with --preset hrm, use the fixed legacy broken-packer probe and retain its prefixless continuations")
    ap.add_argument("--hrm-data-dir",
                    default="eval_data/rsi_tail1024_v1")
    ap.add_argument("--hrm-ledger",
                    default="eval_data/rsi_tail1024_v1/ledger.json")
    ap.add_argument("--hrm-num-blocks", type=int, default=1024)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--wandb_project", default=None,
                    help="opt-in W&B project; HRM PPL logs to the fixed or legacy 174500+ lane")
    ap.add_argument("--wandb_entity", default=None,
                    help="optional W&B entity; defaults to the logged-in account entity")
    args = ap.parse_args()
    if args.wandb_project or args.wandb_entity:
        raise SystemExit("Historical evaluation W&B adapters are not packaged; use --out-json")
    if args.old:
        raise SystemExit("The legacy broken-packer probe is not part of this V2 repository")
    if args.wandb_project and (args.preset != "hrm" or args.suite):
        raise SystemExit("--wandb_project is currently defined only for --preset hrm")


    if args.old:
        if args.preset != "hrm" or args.suite:
            raise SystemExit("--old is defined only for --preset hrm")
        args.hrm_data_dir = OLD_HRM_DATA_DIR
        args.hrm_ledger = OLD_HRM_LEDGER
        print(
            f"[old] legacy broken-packer lane data={args.hrm_data_dir} "
            f"ledger={args.hrm_ledger}",
            flush=True,
        )

    if args.stride <= 0 or args.stride > args.max_length:
        raise SystemExit("--stride must be >0 and <= --max-length")
    t0 = time.time()
    model, trainer, model_info = load_prefixlm_model(
        args.model_path,
        tokenizer_path=args.tokenizer,
        device=args.device,
        bf16=args.bf16,
        trainer_module=args.trainer_module,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(
        f"[model] Loaded kind={model_info.get('kind')} step={model_info.get('step')} "
        f"vocab={len(tokenizer)} in {time.time() - t0:.1f}s",
        flush=True,
    )

    presets = ["wikitext2", "wikitext103", "ptb"] if args.suite else [args.preset]
    all_results = {
        "model_path": args.model_path,
        "model_info": model_info,
        "tokenizer": args.tokenizer,
        "config": {
            "mode": args.mode,
            "max_length": args.max_length,
            "stride": args.stride,
            "max_samples": args.max_samples,
            "max_eval_tokens": args.max_eval_tokens,
            "attn_mask_mode": args.attn_mask_mode,
            "prefixlm_impl": args.prefixlm_impl,
            "old": bool(args.old),
            "hrm_data_dir": args.hrm_data_dir,
            "hrm_ledger": args.hrm_ledger,
            "hrm_num_blocks": args.hrm_num_blocks,
        },
        "results": {},
    }

    for preset in presets:
        args.preset = preset
        if preset == "hrm":
            docs = load_hrm_docs(args)
            probe_meta = getattr(args, "hrm_reference_meta", {})
            source_tail = probe_meta.get("source_tail", {})
            start = source_tail.get("ledger_start_inclusive")
            stop = source_tail.get("ledger_stop_exclusive")
            if probe_meta.get("kind") == "rockso_rsi_tail_ppl_v1" and stop is not None:
                start = int(stop) - args.hrm_num_blocks
            cursor = model_info.get("cursor")
            reached = None if cursor is None or start is None else max(0, min(int(cursor), int(stop)) - int(start))
            all_results["probe_provenance"] = {
                "reference_kind": probe_meta.get("kind"),
                "source_tail": source_tail,
                "selected_ledger_start_inclusive": start,
                "selected_ledger_stop_exclusive": stop,
                "tokenizer_sha256": probe_meta.get("tokenizer_sha256"),
                "content_sha256": probe_meta.get("content_sha256"),
                "checkpoint_cursor": cursor,
                "source_tail_positions_consumed": reached,
                "document_overlap_with_earlier_training": "not_audited; RSI source contains repeated views",
                "scope": "Frozen training-corpus-tail probe; not a permanent document-disjoint holdout",
            }
            print("\n" + "=" * 78)
            print(f"[eval] preset=hrm mode=single-doc docs={len(docs):,} "
                  f"attn={args.attn_mask_mode}/{args.prefixlm_impl}", flush=True)
            print("=" * 78, flush=True)
            all_results["results"]["hrm"] = {"single_doc": compute_hrm_packed_ppl(
                model, trainer, docs, args.device, args.bf16, args)}
            continue
        records = load_records(args)
        tags = MECO_VARIANTS if args.compare_all else [None if args.no_meco else args.meco_tag]
        print("\n" + "=" * 78)
        print(
            f"[eval] preset={preset} mode={args.mode} records={len(records):,} "
            f"attn={args.attn_mask_mode}/{args.prefixlm_impl}",
            flush=True,
        )
        print("=" * 78, flush=True)

        preset_results = {}
        for tag in tags:
            result = compute_prefixlm_ppl(
                model,
                trainer,
                tokenizer,
                records,
                meco_tag=tag,
                mode=args.mode,
                max_length=args.max_length,
                stride=args.stride,
                device=args.device,
                bf16=args.bf16,
                max_eval_tokens=args.max_eval_tokens,
                args=args,
            )
            key = tag or "none"
            preset_results[key] = result
            print_result(key, result)

        if args.compare_all:
            best_key, best = min(preset_results.items(), key=lambda kv: kv[1]["ppl"])
            base = preset_results.get("none")
            if base:
                imp = (base["ppl"] - best["ppl"]) / max(base["ppl"], 1e-9) * 100.0
                print(
                    f"[best] {best_key} PPL={best['ppl']:.3f} "
                    f"vs baseline={base['ppl']:.3f} improvement={imp:.2f}%"
                )

        all_results["results"][preset] = preset_results

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"[json] wrote {args.out_json}", flush=True)



if __name__ == "__main__":
    main()
