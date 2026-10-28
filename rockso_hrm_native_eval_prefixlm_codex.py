#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PrefixLM adapter for rockso_hrm_native_eval.py.

Thin layer only: benchmark construction, HRM prompt wrapping, scoring summaries,
and JSON reports stay in rockso_hrm_native_eval. This file swaps the model-call
helpers so PrefixLM checkpoints use bidirectional prompt prefill followed by
causal KV-cache decode, matching the HRM-Text inference shape.

BBH uses the same generation adapter; its pinned tasks, fixed exemplars,
context accounting and exact-answer scoring live in rockso_bbh_eval.py.
"""

from __future__ import annotations
import json
import time
import hashlib

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

import rockso_hrm_native_eval as base


DEFAULT_PREFIXLM_IMPLICIT = str(Path(__file__).resolve().with_name("Implicit_curriculum_math_dense_v8_prefixlm_codex.py"))
_PREFIXLM_PREFILL = True
_ROPE_IMPL = "auto"
_ACTIVE_ROPE_IMPL = "legacy"
_DECODE_MODE = "per_row"
_STATIC_DECODE_PARITY = True
_STATIC_DECODE_PARITY_DONE = False
_FAST_UNSAFE_GSM8K = False
_WANDB_PROJECT = None
_WANDB_ENTITY = None
_ACTIVE_IMPLICIT_MODULE = None
_GRAPH_DECODE_MODEL = None
_GRAPH_DECODE_INFO = None
_GENERATION_STATS = []
_CODE_PROMPT_FORMAT = "original"
_CODE_OUTPUT_MODE = "legacy"
_ENCODE_ATTEMPTS = []
_GENERATION_STREAM = None
_GENERATION_STREAM_PATH = None
_DECODE_REQUESTED_MODE = None


def _normalize_rope_impl(value: str) -> str:
    v = (value or "auto").lower().strip()
    if v in {"rotate_half", "rotate-half", "rotatehalf", "hrm", "olmo"}:
        return "rotate_half"
    if v in {"legacy", "interleaved", "pair"}:
        return "legacy"
    if v == "auto":
        return "auto"
    raise ValueError(f"Unknown rope_impl: {value}")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    cos = cos.unsqueeze(1).to(dtype)
    sin = sin.unsqueeze(1).to(dtype)
    if _ACTIVE_ROPE_IMPL == "rotate_half":
        return (x * cos + _rotate_half(x) * sin).to(dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + xr * sin


def _extract_logits_and_cache(out):
    if isinstance(out, dict):
        logits = out.get("logits")
        kv = out.get("kv_caches", out.get("past_key_values"))
        if logits is None:
            raise RuntimeError(f"model returned dict without logits: {sorted(out.keys())}")
        return logits, kv
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


def _call_model(model, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches=None):
    try:
        out = model(input_ids, position_ids=position_ids, kv_caches=kv_caches)
    except TypeError:
        if kv_caches is not None:
            out = model(input_ids, position_ids, kv_caches)
        else:
            try:
                out = model(input_ids, position_ids=position_ids)
            except TypeError:
                out = model(input_ids, position_ids)
    return _extract_logits_and_cache(out)


def _prefill(model, prompt_ids: List[int], device: str):
    if not prompt_ids:
        raise ValueError("PrefixLM prefill requires a non-empty prompt")
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    pos = torch.arange(len(prompt_ids), device=device).unsqueeze(0)
    return _call_model(model, x, pos, kv_caches=None)


def _decode_one(model, token_id: int, position: int, kv_caches, device: str):
    x = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
    pos = torch.tensor([[int(position)]], dtype=torch.long, device=device)
    return _call_model(model, x, pos, kv_caches=kv_caches)


def _make_static_caches(row_kv_caches: List[Any], prompt_lengths: List[int], max_context: int, model=None):
    if getattr(model, "is_rockso_hybrid", False):
        return model.pack_static_caches(row_kv_caches, prompt_lengths, max_context)
    if not row_kv_caches:
        raise ValueError("no KV caches to pack")
    batch_size = len(row_kv_caches)
    static_caches = []
    for layer_idx, (k0, v0) in enumerate(row_kv_caches[0]):
        if max(prompt_lengths) > max_context:
            raise ValueError(f"prompt exceeds max_context={max_context}")
        k_static = torch.zeros(
            (batch_size, k0.size(1), max_context, k0.size(3)),
            dtype=k0.dtype,
            device=k0.device,
        )
        v_static = torch.zeros(
            (batch_size, v0.size(1), max_context, v0.size(3)),
            dtype=v0.dtype,
            device=v0.device,
        )
        for row_idx, row_cache in enumerate(row_kv_caches):
            k, v = row_cache[layer_idx]
            length = int(prompt_lengths[row_idx])
            k_static[row_idx, :, :length, :].copy_(k[0, :, :length, :])
            v_static[row_idx, :, :length, :].copy_(v[0, :, :length, :])
        static_caches.append((k_static, v_static))
    return static_caches


def _static_decode_block_forward(block, x, position_ids, static_cache, cache_lengths, v1=None):
    B, T, _ = x.shape
    if T != 1:
        raise ValueError("static decode only supports one token per row")
    h = block.norm1(x)

    q = block.q_proj(h).view(B, T, block.n_head, block.head_dim).transpose(1, 2)
    k = block.k_proj(h).view(B, T, block.n_kv_heads, block.head_dim).transpose(1, 2)
    v = block.v_proj(h).view(B, T, block.n_kv_heads, block.head_dim).transpose(1, 2)
    v_out = v

    q = block.q_norm(q)
    k = block.k_norm(k)

    if block.value_residual and v1 is not None:
        denom = torch.rsqrt(block.vr_alpha1 ** 2 + block.vr_alpha2 ** 2 + 1e-8)
        v = block.vr_scale * (block.vr_alpha1 * v + block.vr_alpha2 * v1) * denom

    cos, sin = block.rope(position_ids)
    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    k_cache, v_cache = static_cache
    rows = torch.arange(B, device=x.device)
    pos = cache_lengths.long()
    k_cache[rows, :, pos, :] = k.squeeze(2)
    v_cache[rows, :, pos, :] = v.squeeze(2)

    max_len = int(pos.max().item()) + 1
    k_all = k_cache[:, :, :max_len, :]
    v_all = v_cache[:, :, :max_len, :]

    if block.n_kv_heads < block.n_head:
        rep = block.n_head // block.n_kv_heads
        k_all = k_all.repeat_interleave(rep, dim=1)
        v_all = v_all.repeat_interleave(rep, dim=1)

    # SDPA has no cache_lengths argument. This tiny decode-only mask is the
    # local equivalent of HRM-Text's flash_attn_with_kvcache(cache_seqlens=...).
    valid = torch.arange(max_len, device=x.device).view(1, 1, 1, max_len) <= pos.view(B, 1, 1, 1)
    attn = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=valid, is_causal=False)
    attn = attn.transpose(1, 2).reshape(B, T, block.n_embd)
    x = x + block.out_proj(attn)

    h2 = block.norm2(x)
    x = x + block.mlp(h2)
    return x, v_out


def _static_decode_step(model, token_ids: torch.Tensor, static_caches, cache_lengths: torch.Tensor,
                        active_mask=None) -> torch.Tensor:
    global _ACTIVE_ROPE_IMPL
    if getattr(model, "is_rockso_hybrid", False):
        return model.static_decode(token_ids, static_caches, cache_lengths, active_mask)
    model_rope = getattr(model, "rope_impl", None)
    if model_rope:
        _ACTIVE_ROPE_IMPL = _normalize_rope_impl(str(model_rope))
    input_ids = token_ids.long().view(-1, 1)
    position_ids = cache_lengths.long().view(-1, 1)
    x = model.embed(input_ids)
    v1 = None
    for layer_idx, block in enumerate(model.blocks):
        x, v_out = _static_decode_block_forward(
            block,
            x,
            position_ids,
            static_caches[layer_idx],
            cache_lengths,
            v1=v1,
        )
        if v1 is None:
            v1 = v_out
    x = model.norm(x)
    return model.lm_head(x)[:, -1, :]


@torch.no_grad()
def model_logits(model, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return last-prompt logits without letting bidirectional prefill see pads."""
    outs = []
    device = str(input_ids.device)
    for i, length in enumerate(lengths.tolist()):
        ids = input_ids[i, :int(length)].tolist()
        logits, _kv = _prefill(model, [int(x) for x in ids], device)
        outs.append(logits[0, int(length) - 1, :])
    return torch.stack(outs, dim=0)


@torch.no_grad()
def model_full_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    rows = []
    device = str(input_ids.device)
    for i in range(input_ids.size(0)):
        ids = [int(x) for x in input_ids[i].tolist()]
        logits, _kv = _prefill(model, ids, device)
        rows.append(logits)
    return torch.cat(rows, dim=0)


@torch.no_grad()
def continuation_logprob(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> float:
    if not cont_ids:
        return 0.0
    logits, kv_caches = _prefill(model, [int(x) for x in prompt_ids], device)
    total = 0.0
    position = len(prompt_ids)
    for tid in [int(x) for x in cont_ids]:
        lp = F.log_softmax(logits[0, -1, :].float(), dim=-1)
        total += float(lp[tid].item())
        logits, kv_caches = _decode_one(model, tid, position, kv_caches, device)
        position += 1
    return total


def continuation_logprob_stats(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> Tuple[float, int]:
    return continuation_logprob(model, prompt_ids, cont_ids, device), max(1, len(cont_ids))


@torch.no_grad()
def continuation_logprob_next_only(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> float:
    return continuation_logprob(model, prompt_ids, cont_ids, device)


@torch.no_grad()
def _continuation_logprob_from_prefill(
    model,
    prefill_logits: torch.Tensor,
    prefill_kv,
    prompt_length: int,
    cont_ids: List[int],
    device: str,
) -> float:
    """Score one causal continuation while sharing an immutable prompt prefill."""
    if not cont_ids:
        return 0.0
    logits = prefill_logits
    kv_caches = prefill_kv
    total = 0.0
    position = int(prompt_length)
    ids = [int(x) for x in cont_ids]
    for token_index, tid in enumerate(ids):
        lp = F.log_softmax(logits[0, -1, :].float(), dim=-1)
        total += float(lp[tid].item())
        if token_index + 1 < len(ids):
            logits, kv_caches = _decode_one(model, tid, position, kv_caches, device)
            position += 1
    return total


@torch.no_grad()
def _continuation_logprobs_batched_from_prefill(
    model,
    prefill_logits: torch.Tensor,
    prefill_kv,
    prompt_length: int,
    continuations: List[List[int]],
) -> List[float]:
    """Score answer branches together after one shared PrefixLM prefill.

    Every branch has the same prompt length, so the static-cache decode path is
    exact here (the variable-prompt-length case that made it unsafe for general
    batched generation does not apply).
    """
    if not continuations:
        return []
    ids = [[int(x) for x in seq] for seq in continuations]
    batch_size = len(ids)
    max_tokens = max((len(seq) for seq in ids), default=0)
    if max_tokens == 0:
        return [0.0] * batch_size

    # Allocate only as far as these answer branches can reach, rather than the
    # evaluator's full max_context. This keeps the 8-way GPQA cache modest.
    cache_capacity = int(prompt_length) + max_tokens
    static_caches = _make_static_caches(
        [prefill_kv] * batch_size,
        [int(prompt_length)] * batch_size,
        cache_capacity,
        model=model,
    )
    cache_lengths = torch.full(
        (batch_size,), int(prompt_length), dtype=torch.long, device=prefill_logits.device
    )
    logits = prefill_logits[:, -1, :].expand(batch_size, -1)
    totals = torch.zeros(batch_size, dtype=torch.float32, device=prefill_logits.device)

    for token_index in range(max_tokens):
        active = torch.tensor(
            [token_index < len(seq) for seq in ids],
            dtype=torch.bool,
            device=prefill_logits.device,
        )
        token_ids = torch.tensor(
            [seq[token_index] if token_index < len(seq) else (seq[-1] if seq else 0) for seq in ids],
            dtype=torch.long,
            device=prefill_logits.device,
        )
        lp = F.log_softmax(logits.float(), dim=-1)
        selected = lp.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        totals += torch.where(active, selected, torch.zeros_like(selected))
        if token_index + 1 < max_tokens:
            logits = _static_decode_step(model, token_ids, static_caches, cache_lengths)
            cache_lengths.add_(1)
    return [float(x) for x in totals.tolist()]


@torch.no_grad()
def score_rc_rows(
    model,
    tokenizer,
    items,
    rows,
    device: str,
    pad_id: int,
    max_context: int,
) -> Dict[int, Dict[str, Any]]:
    """PrefixLM RC scoring with one shared bidirectional prefill per prompt."""
    del pad_id
    by_local: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    text_by_local: Dict[int, Dict[str, str]] = defaultdict(dict)
    for local_i, (item_idx, prompt_ids) in enumerate(rows):
        item = items[item_idx]
        prefill_logits, prefill_kv = _prefill(model, [int(x) for x in prompt_ids], device)
        branches = []
        for lab, answer_text in zip(item.choices, item.choice_texts):
            for variant, cont_ids in base.text_continuation_variants(tokenizer, answer_text).items():
                if len(prompt_ids) + len(cont_ids) >= max_context:
                    continue
                branches.append((lab, answer_text, variant, cont_ids))
        totals = _continuation_logprobs_batched_from_prefill(
            model,
            prefill_logits,
            prefill_kv,
            len(prompt_ids),
            [cont_ids for _lab, _text, _variant, cont_ids in branches],
        )
        for (lab, answer_text, _variant, cont_ids), total in zip(branches, totals):
                n_tok = max(1, len(cont_ids))
                n_char = max(1, len(str(answer_text).strip()))
                stats = {
                    "raw": total,
                    "tok_norm": total / n_tok,
                    "char_norm": total / n_char,
                    "tokens": float(n_tok),
                    "chars": float(n_char),
                }
                previous = by_local[local_i].get(lab)
                if previous is None or stats["tok_norm"] > previous["tok_norm"]:
                    by_local[local_i][lab] = stats
                    text_by_local[local_i][lab] = str(answer_text).strip()

    out: Dict[int, Dict[str, Any]] = {}
    for local_i, scores in by_local.items():
        if not scores:
            continue
        raw_scores = {lab: values["raw"] for lab, values in scores.items()}
        tok_scores = {lab: values["tok_norm"] for lab, values in scores.items()}
        char_scores = {lab: values["char_norm"] for lab, values in scores.items()}
        out[local_i] = {
            "rc_raw_pred": base.argmax_label(raw_scores),
            "rc_tok_norm_pred": base.argmax_label(tok_scores),
            "rc_char_norm_pred": base.argmax_label(char_scores),
            "rc_raw_scores": raw_scores,
            "rc_tok_norm_scores": tok_scores,
            "rc_char_norm_scores": char_scores,
            "choice_texts": text_by_local[local_i],
        }
    return out


@torch.no_grad()
def generate_greedy_batch_per_row(
    model,
    tokenizer,
    prompt_ids: List[List[int]],
    device: str,
    max_new_tokens: int,
    max_context: int,
    suppress_ids: Sequence[int],
    stop_strings: Sequence[str],
    pad_id: int,
    preserve_whitespace: bool = False,
) -> List[str]:
    del pad_id
    stops = base.stop_token_ids(tokenizer)
    suppress = list(suppress_ids) if suppress_ids else []
    generations: List[str] = []

    # Correctness anchor: HRM-Text prefills each prompt independently and then
    # decodes against that row's cache length. The attempted static batched SDPA
    # replacement proved unsafe for variable-length DROP prompts, so generation
    # stays per-row here.
    for prompt in prompt_ids:
        prompt = [int(x) for x in prompt]
        started = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
        logits, kv_caches = _prefill(model, prompt, device)
        generated: List[int] = []
        position = len(prompt)
        stop_reason, stop_id = "token_limit", None
        code_end = None

        for _ in range(max_new_tokens):
            next_logits = logits[0, -1, :]
            if suppress:
                next_logits = next_logits.clone()
                next_logits[suppress] = -torch.inf
            tid = int(torch.argmax(next_logits, dim=-1).item())
            if tid in stops:
                stop_reason, stop_id = "stop_token", tid
                break
            if len(prompt) + len(generated) + 1 >= max_context:
                stop_reason = "context_limit"
                break

            generated.append(tid)
            text_so_far = tokenizer.decode(generated, skip_special_tokens=True)
            if _CODE_OUTPUT_MODE == "fenced":
                from code_output_policy import closing_fence_end
                code_end = closing_fence_end(text_so_far)
                if code_end is not None:
                    stop_reason = "closing_code_fence"
                    break
            if any(ss and ss in text_so_far for ss in stop_strings):
                stop_reason = "stop_string"
                break

            logits, kv_caches = _decode_one(model, tid, position, kv_caches, device)
            position += 1

        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        if code_end is not None:
            decoded = decoded[:code_end]
        generations.append(base.trim_generation(decoded, stop_strings, preserve_whitespace))
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        stats = {"prompt_tokens": len(prompt), "generated_tokens": len(generated),
                 "stop_reason": stop_reason, "stop_id": stop_id,
                 "seconds": time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - started,
                 "prompt_ids_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest()}
        _GENERATION_STATS.append(stats)
        if _GENERATION_STREAM is not None:
            _GENERATION_STREAM.write(json.dumps({"generation_index": len(_GENERATION_STATS) - 1,
                                                "generation": generations[-1], "decode_stats": stats}) + "\n")
            _GENERATION_STREAM.flush()

    return generations


@torch.no_grad()
def generate_greedy_batch_static_decode(
    model,
    tokenizer,
    prompt_ids: List[List[int]],
    device: str,
    max_new_tokens: int,
    max_context: int,
    suppress_ids: Sequence[int],
    stop_strings: Sequence[str],
    pad_id: int,
    preserve_whitespace: bool = False,
) -> List[str]:
    del pad_id
    prompts = [[int(x) for x in prompt] for prompt in prompt_ids]
    if not prompts:
        return []

    stops = base.stop_token_ids(tokenizer)
    suppress = list(suppress_ids) if suppress_ids else []
    generated: List[List[int]] = [[] for _ in prompts]
    active = [True for _ in prompts]
    code_end_offsets = [None for _ in prompts]

    row_kv_caches = []
    prompt_lengths = []
    last_logits = []
    for prompt in prompts:
        logits, kv_caches = _prefill(model, prompt, device)
        row_kv_caches.append(kv_caches)
        prompt_lengths.append(len(prompt))
        # A view would keep the entire prefill's sequence-by-vocabulary storage.
        last_logits.append(logits[0, -1, :].clone())

    capacity = min(max_context, max(prompt_lengths) + max(1, max_new_tokens))
    static_caches = _make_static_caches(row_kv_caches, prompt_lengths, capacity, model=model)
    cache_lengths = torch.tensor(prompt_lengths, dtype=torch.long, device=device)
    logits_batch = torch.stack(last_logits, dim=0)
    del row_kv_caches, last_logits, logits, kv_caches

    for _ in range(max_new_tokens):
        if not any(active):
            break

        step_logits = logits_batch
        if suppress:
            step_logits = step_logits.clone()
            step_logits[:, suppress] = -torch.inf
        next_ids = step_logits.argmax(dim=-1).tolist()

        decode_tokens = torch.empty(len(prompts), dtype=torch.long, device=device)
        decode_mask = torch.zeros(len(prompts), dtype=torch.bool, device=device)

        for row_i, tid in enumerate(next_ids):
            tid = int(tid)
            decode_tokens[row_i] = tid
            if not active[row_i]:
                continue
            if tid in stops or len(prompts[row_i]) + len(generated[row_i]) + 1 >= max_context:
                active[row_i] = False
                continue

            generated[row_i].append(tid)
            text_so_far = tokenizer.decode(generated[row_i], skip_special_tokens=True)
            if _CODE_OUTPUT_MODE == "fenced":
                from code_output_policy import closing_fence_end
                code_end_offsets[row_i] = closing_fence_end(text_so_far)
                if code_end_offsets[row_i] is not None:
                    active[row_i] = False
                    continue
            if any(ss and ss in text_so_far for ss in stop_strings):
                active[row_i] = False
                continue

            decode_mask[row_i] = True

        if not bool(decode_mask.any().item()):
            break

        # Decode as a dense batch. Hybrid caches skip writes for finished rows;
        # the legacy path rewrites their unused slot. Neither advances them.
        logits_batch = _static_decode_step(model, decode_tokens, static_caches, cache_lengths,
                                           active_mask=decode_mask)
        cache_lengths[decode_mask] += 1

    return [
        base.trim_generation(tokenizer.decode(ids, skip_special_tokens=True)[:code_end_offsets[i]], stop_strings, preserve_whitespace)
        for i, ids in enumerate(generated)
    ]


@torch.no_grad()
def generate_greedy_batch(
    model,
    tokenizer,
    prompt_ids: List[List[int]],
    device: str,
    max_new_tokens: int,
    max_context: int,
    suppress_ids: Sequence[int],
    stop_strings: Sequence[str],
    pad_id: int,
    preserve_whitespace: bool = False,
) -> List[str]:
    global _DECODE_MODE, _STATIC_DECODE_PARITY_DONE, _GRAPH_DECODE_MODEL, _GRAPH_DECODE_INFO
    if _DECODE_MODE == "graph" and prompt_ids:
        # Keep the original model for likelihood scoring and branched caches.
        # The graph wrapper owns one sequential generation cache only.
        stats_start = len(_GENERATION_STATS)
        try:
            if _GRAPH_DECODE_MODEL is None or _GRAPH_DECODE_MODEL.reference is not model:
                from hybrid_graph_decode import enable_cuda_graph_decode
                _GRAPH_DECODE_MODEL = enable_cuda_graph_decode(model, prompt_ids[0])
                _GRAPH_DECODE_INFO = _GRAPH_DECODE_MODEL.inference_provenance["cuda_graph_decode"]
            return generate_greedy_batch_per_row(
                _GRAPH_DECODE_MODEL, tokenizer, prompt_ids, device, max_new_tokens, max_context,
                suppress_ids, stop_strings, pad_id, preserve_whitespace,
            )
        except (AssertionError, ValueError, TypeError, RuntimeError) as exc:
            del _GENERATION_STATS[stats_start:]
            if _GENERATION_STREAM is not None:
                _GENERATION_STREAM.write(json.dumps({"event": "discard_batch_before_reference_retry", "from_generation_index": stats_start}) + "\n")
                _GENERATION_STREAM.flush()
            print(f"[prefixlm] CUDA graph decode unavailable ({type(exc).__name__}: {exc}); using reference per-row decode")
            _GRAPH_DECODE_INFO = {"backend": "reference_fallback", "reason": f"{type(exc).__name__}: {exc}"}
            _GRAPH_DECODE_MODEL = None
            _DECODE_MODE = "per_row"
    if _DECODE_MODE != "static":
        return generate_greedy_batch_per_row(
            model, tokenizer, prompt_ids, device, max_new_tokens, max_context,
            suppress_ids, stop_strings, pad_id, preserve_whitespace,
        )

    if _STATIC_DECODE_PARITY and not _STATIC_DECODE_PARITY_DONE and len(prompt_ids) >= 2:
        probe_prompts = prompt_ids[: min(3, len(prompt_ids))]
        probe_tokens = min(64, max_new_tokens)
        try:
            static_probe = generate_greedy_batch_static_decode(
                model, tokenizer, probe_prompts, device, probe_tokens, max_context,
                suppress_ids, stop_strings, pad_id, preserve_whitespace,
            )
        except Exception as exc:
            print(f"[prefixlm] static decode parity errored ({type(exc).__name__}: {exc}); falling back to per-row decode")
            _STATIC_DECODE_PARITY_DONE = True
            _DECODE_MODE = "per_row"
            return generate_greedy_batch_per_row(
                model, tokenizer, prompt_ids, device, max_new_tokens, max_context,
                suppress_ids, stop_strings, pad_id, preserve_whitespace,
            )
        stats_start = len(_GENERATION_STATS)
        per_row_probe = generate_greedy_batch_per_row(
            model, tokenizer, probe_prompts, device, probe_tokens, max_context,
            suppress_ids, stop_strings, pad_id, preserve_whitespace,
        )
        del _GENERATION_STATS[stats_start:]
        _STATIC_DECODE_PARITY_DONE = True
        if static_probe != per_row_probe:
            print("[prefixlm] static decode parity FAILED; falling back to per-row decode")
            for i, (a, b) in enumerate(zip(static_probe, per_row_probe)):
                if a != b:
                    print(f"[prefixlm] parity row {i} static={a[:160]!r} per_row={b[:160]!r}")
                    break
            _DECODE_MODE = "per_row"
            return generate_greedy_batch_per_row(
                model, tokenizer, prompt_ids, device, max_new_tokens, max_context,
                suppress_ids, stop_strings, pad_id, preserve_whitespace,
            )
        print(f"[prefixlm] static decode parity OK on {len(probe_prompts)} rows / {probe_tokens} tokens")

    try:
        return generate_greedy_batch_static_decode(
            model, tokenizer, prompt_ids, device, max_new_tokens, max_context,
            suppress_ids, stop_strings, pad_id, preserve_whitespace,
        )
    except Exception as exc:
        print(f"[prefixlm] static decode errored ({type(exc).__name__}: {exc}); falling back to per-row decode")
        _DECODE_MODE = "per_row"
        return generate_greedy_batch_per_row(
            model, tokenizer, prompt_ids, device, max_new_tokens, max_context,
            suppress_ids, stop_strings, pad_id, preserve_whitespace,
        )


def _consume_prefixlm_args(argv: List[str]) -> List[str]:
    global _PREFIXLM_PREFILL, _ROPE_IMPL, _DECODE_MODE, _STATIC_DECODE_PARITY, _FAST_UNSAFE_GSM8K
    global _WANDB_PROJECT, _WANDB_ENTITY, _CODE_PROMPT_FORMAT, _CODE_OUTPUT_MODE
    cleaned = [argv[0]]
    it = iter(argv[1:])
    for arg in it:
        if arg == "--causal-prefill":
            _PREFIXLM_PREFILL = False
            continue
        if arg == "--rope-impl":
            _ROPE_IMPL = _normalize_rope_impl(next(it))
            continue
        if arg.startswith("--rope-impl="):
            _ROPE_IMPL = _normalize_rope_impl(arg.split("=", 1)[1])
            continue
        if arg in {"--static-decode", "--gsm8k-static-decode"}:
            _DECODE_MODE = "static"
            continue
        if arg == "--gsm8k-fast-unsafe":
            _DECODE_MODE = "static"
            _STATIC_DECODE_PARITY = False
            _FAST_UNSAFE_GSM8K = True
            continue
        if arg == "--per-row-decode":
            _DECODE_MODE = "per_row"
            continue
        if arg == "--cuda-graph-decode":
            _DECODE_MODE = "graph"
            continue
        if arg == "--code-output-mode":
            _CODE_OUTPUT_MODE = next(it)
            continue
        if arg.startswith("--code-output-mode="):
            _CODE_OUTPUT_MODE = arg.split("=", 1)[1]
            continue
        if arg == "--code-prompt-format":
            _CODE_PROMPT_FORMAT = next(it)
            continue
        if arg.startswith("--code-prompt-format="):
            _CODE_PROMPT_FORMAT = arg.split("=", 1)[1]
            continue
        if arg == "--no-static-decode-parity":
            _STATIC_DECODE_PARITY = False
            continue
        if arg == "--wandb_project":
            _WANDB_PROJECT = next(it)
            continue
        if arg.startswith("--wandb_project="):
            _WANDB_PROJECT = arg.split("=", 1)[1]
            continue
        if arg == "--wandb_entity":
            _WANDB_ENTITY = next(it)
            continue
        if arg.startswith("--wandb_entity="):
            _WANDB_ENTITY = arg.split("=", 1)[1]
            continue
        cleaned.append(arg)
    return cleaned


_orig_import_implicit_loader = base.import_implicit_loader
_orig_run_mcq_task = base.run_mcq_task
_orig_run_generation_task = base.run_generation_task
_orig_encode_gen_prompt = base.encode_gen_prompt


def encode_gen_prompt(tokenizer, item, condition, max_context, max_new_tokens):
    ids = _orig_encode_gen_prompt(tokenizer, item, condition, max_context, max_new_tokens)
    _ENCODE_ATTEMPTS.append({"index": len(_ENCODE_ATTEMPTS), "encoded": ids is not None,
                             "group": item.group,
                             "prompt_sha256": hashlib.sha256(item.prompt.encode()).hexdigest()})
    return ids


def _run_observed(runner, args):
    global _GENERATION_STREAM, _GENERATION_STREAM_PATH, _DECODE_REQUESTED_MODE
    _GENERATION_STATS.clear()
    _ENCODE_ATTEMPTS.clear()
    _DECODE_REQUESTED_MODE = _DECODE_MODE
    _GENERATION_STREAM_PATH = None
    if args.json and (_DECODE_MODE != "static"):
        # Persist each completed response even if a long benchmark is interrupted.
        path = Path(args.json).with_suffix(".generations.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        _GENERATION_STREAM = path.open("x", encoding="utf-8")
        _GENERATION_STREAM_PATH = str(path)
    started = time.clock_gettime(time.CLOCK_MONOTONIC_RAW)
    try:
        report = _annotate_report(runner(args), args)
        report["evaluation_seconds_raw"] = time.clock_gettime(time.CLOCK_MONOTONIC_RAW) - started
        return report
    finally:
        if _GENERATION_STREAM is not None:
            _GENERATION_STREAM.close()
            _GENERATION_STREAM = None


def import_implicit_loader(implicit_path: str):
    global _ACTIVE_IMPLICIT_MODULE
    module = _orig_import_implicit_loader(implicit_path)
    _ACTIVE_IMPLICIT_MODULE = module
    if hasattr(module, "_PREFIXLM_PREFILL"):
        module._PREFIXLM_PREFILL = bool(_PREFIXLM_PREFILL)
    if hasattr(module, "_ROPE_IMPL"):
        module._ROPE_IMPL = _ROPE_IMPL
    return module


def _effective_split_for_args(args) -> str:
    if args.split:
        return str(args.split)
    if args.task == "gpqa":
        return "train"
    if args.task in {
        "drop",
        "coqa",
        "naturalqs",
        "triviaqa",
        "squad",
        "hellaswag",
        "winogrande",
        "boolq",
        "sciq",
        "piqa",
        "siqa",
    }:
        return "validation"
    return "test"


def _annotate_report(report: Dict[str, Any], args) -> Dict[str, Any]:
    report["model_inference"] = getattr(_ACTIVE_IMPLICIT_MODULE, "_MODEL_INFO", None)
    report["base_diagnostic"] = report.get("diagnostic")
    report["diagnostic"] = "rockso_hrm_native_eval_prefixlm"
    report["prefixlm_inference"] = {
        "prompt_prefill": "bidirectional" if _PREFIXLM_PREFILL else "causal",
        "rope_impl": _ROPE_IMPL,
        "decode": ("not_used_likelihood_scoring" if args.task in {"mmlu", "mmlu_pro", "gpqa", "arc", "arc_easy", "hellaswag", "winogrande", "boolq", "sciq", "piqa", "siqa"} and args.mcq_mode != "cot"
                   else {"static": "causal_static_batched_kv_cache", "graph": "causal_per_row_cuda_graph"}.get(_DECODE_MODE, "causal_per_row_kv_cache")),
        "decode_requested": _DECODE_REQUESTED_MODE,
        "cuda_graph_decode": _GRAPH_DECODE_INFO,
        "pad_handling": "unpadded_per_row_prefill",
        "static_decode_parity": bool(_STATIC_DECODE_PARITY),
        "fast_unsafe_gsm8k": bool(_FAST_UNSAFE_GSM8K),
        "rc_scoring": "shared_bidirectional_prefill_batched_causal_kv_branches",
        "implicit_loader": str(getattr(args, "implicit", "")),
    }
    report["evaluation_config"] = {
        "effective_split": _effective_split_for_args(args),
        "split": args.split,
        "subjects": args.subjects,
        "limit": int(args.limit),
        "batch_size": int(args.batch_size),
        "dtype": str(args.dtype),
        "device": str(args.device),
        "no_suppress_special": bool(args.no_suppress_special),
    }
    if args.task == "bbh":
        report["evaluation_config"]["bbh_protocol_signature"] = report["bbh_protocol"]["signature"]
    if "code_protocol" in report:
        report["code_protocol"]["v2_output_mode"] = _CODE_OUTPUT_MODE
        if _CODE_OUTPUT_MODE == "fenced":
            from code_output_policy import POLICY_VERSION
            source = Path(__file__).with_name("code_output_policy.py")
            report["code_protocol"]["name"] += "_fenced_output_v1"
            report["code_protocol"]["extraction"] = "Whole plain Python, or one complete Python fence with surrounding prose allowed; no code edits or repair"
            report["code_protocol"]["output_policy"] = {
                "version": POLICY_VERSION,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "stopping": "first complete Python fence closes, or original stop token/context/token budget",
                "prompt_changed": False, "condition_changed": False, "answer_prefix_forced": False,
                "code_edited": False, "original_tests_unchanged": True,
            }
        report["code_protocol"]["v2_prompt_format"] = _CODE_PROMPT_FORMAT
        if _CODE_PROMPT_FORMAT in {"task", "approach_first"}:
            source = Path(__file__).with_name("code_prompt_formats.py")
            report["code_protocol"]["name"] += "_task_description"
            report["code_protocol"]["v2_prompt_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            report["code_protocol"]["prompt_note"] = ("Public description/examples outside code; exact signature/imports preserved. Hidden tests unchanged; output extraction/stopping recorded separately."
                                                        if args.task == "humaneval" else "Zero-shot target description and public test_list only. No canonical solution or challenge tests in prompt. Original executable tests unchanged; output extraction/stopping recorded separately.")
            if _CODE_PROMPT_FORMAT == "approach_first":
                from code_prompt_formats import APPROACH_FIRST_INSTRUCTION, APPROACH_FIRST_VERSION
                report["code_protocol"]["name"] += "_approach_first_3sentences_v1"
                report["code_protocol"]["prompt_variant_version"] = APPROACH_FIRST_VERSION
                report["code_protocol"]["response_instruction"] = APPROACH_FIRST_INSTRUCTION
                report["code_protocol"]["prompt_note"] += " Final response instruction requests a brief approach before one complete Python block; public task content unchanged."
                if "output_policy" in report["code_protocol"]:
                    report["code_protocol"]["output_policy"]["prompt_changed"] = True
    items = [item for item in report.get("items", [])
             if "generation" in item and item.get("status") not in {"context_failure", "dropped_context"}]
    if _GENERATION_STATS:
        if items and len(_GENERATION_STATS) == len(items):
            for item, stats in zip(items, _GENERATION_STATS):
                item["decode_stats"] = stats
        report["summary"]["generation_stop_reasons"] = {
            reason: sum(x["stop_reason"] == reason for x in _GENERATION_STATS)
            for reason in sorted({x["stop_reason"] for x in _GENERATION_STATS})}
        report["summary"]["generated_tokens"] = sum(x["generated_tokens"] for x in _GENERATION_STATS)
        report["summary"]["generation_seconds_raw"] = sum(x["seconds"] for x in _GENERATION_STATS)
        report["summary"]["generated_n"] = len(_GENERATION_STATS)
        report["generation_progress_jsonl"] = _GENERATION_STREAM_PATH
    if _ENCODE_ATTEMPTS:
        requested = len(_ENCODE_ATTEMPTS)
        dropped = sum(not row["encoded"] for row in _ENCODE_ATTEMPTS)
        report["coverage"] = {"requested_n": requested, "scored_n": report["summary"]["n"],
                              "dropped_context_n": dropped,
                              "score_denominator": "scored items; context drops excluded",
                              "context_drops": [row for row in _ENCODE_ATTEMPTS if not row["encoded"]]}
        # The shared DROP/OpenQA scorer receives only encoded items. Correct its
        # dropped count here without altering its scores or their denominator.
        report["summary"]["dropped"] = dropped
    return report


def run_mcq_task(args) -> Dict[str, Any]:
    return _run_observed(_orig_run_mcq_task, args)


def run_generation_task(args) -> Dict[str, Any]:
    return _run_observed(_orig_run_generation_task, args)


def _ensure_prefixlm_default_implicit(argv: List[str]) -> List[str]:
    if "--implicit" in argv or any(arg.startswith("--implicit=") for arg in argv):
        return argv
    if not Path(DEFAULT_PREFIXLM_IMPLICIT).exists():
        print(f"[prefixlm] warning: default implicit loader not found: {DEFAULT_PREFIXLM_IMPLICIT}")
    return argv[:1] + ["--implicit", DEFAULT_PREFIXLM_IMPLICIT] + argv[1:]


def _cli_arg_value(argv: List[str], flag: str):
    value = None
    for index, arg in enumerate(argv):
        if arg == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value")
            value = argv[index + 1]
        if arg.startswith(flag + "="):
            value = arg.split("=", 1)[1]
    return value

def main() -> int:
    task = _cli_arg_value(sys.argv, "--task")
    code_instruct = task in {"humaneval", "mbpp"} and _cli_arg_value(sys.argv, "--code-prompt-mode") in {None, "instruct"}
    if _cli_arg_value(sys.argv, "--code-output-mode") is None:
        sys.argv.extend(["--code-output-mode", "fenced" if code_instruct else "legacy"])
    if task == "mbpp" and _cli_arg_value(sys.argv, "--num-shots") is None:
        sys.argv.extend(["--num-shots", "0"])
    if _cli_arg_value(sys.argv, "--json") is None:
        checkpoint = _cli_arg_value(sys.argv, "--checkpoint") or "checkpoint"
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        sys.argv.extend(["--json", str(Path(__file__).parent / "evals" / f"{Path(checkpoint).stem}_{task}_{stamp}.json")])
    if (task in {"humaneval", "mbpp"} and _cli_arg_value(sys.argv, "--code-prompt-format") is None
            and (task == "humaneval" or _cli_arg_value(sys.argv, "--num-shots") == "0")
            and _cli_arg_value(sys.argv, "--code-prompt-mode") in {None, "instruct"}):
        sys.argv.extend(["--code-prompt-format", "approach_first"])
    if _cli_arg_value(sys.argv, "--tokenizer") is None:
        raise SystemExit("Pass --tokenizer with the exact Rockso V2 tokenizer directory")
    if _cli_arg_value(sys.argv, "--task") in {"mbpp", "humaneval"} and _cli_arg_value(sys.argv, "--condition") is None:
        sys.argv.extend(["--condition", "synth,cot"])
    # Keep the code benchmark's RoPE choice explicit, including short CLI calls.
    if _cli_arg_value(sys.argv, "--task") in {"mbpp", "humaneval"} and _cli_arg_value(sys.argv, "--rope-impl") is None:
        sys.argv.extend(["--rope-impl", "rotate_half"])
    sys.argv = _consume_prefixlm_args(sys.argv)
    if _WANDB_PROJECT or _WANDB_ENTITY:
        raise SystemExit("Historical evaluation W&B adapters are not packaged; use --json")
    sys.argv = _ensure_prefixlm_default_implicit(sys.argv)
    if _CODE_OUTPUT_MODE not in {"legacy", "fenced"}:
        raise ValueError("--code-output-mode must be legacy or fenced")
    if _CODE_OUTPUT_MODE == "fenced":
        if not code_instruct:
            raise ValueError("--code-output-mode fenced requires HumanEval/MBPP instruction mode")
        import rockso_code_eval
        from code_output_policy import install_scorer
        install_scorer(rockso_code_eval)
    if _CODE_PROMPT_FORMAT not in {"original", "task", "approach_first"}:
        raise ValueError("--code-prompt-format must be original, task, or approach_first")
    if _CODE_PROMPT_FORMAT in {"task", "approach_first"}:
        if (_cli_arg_value(sys.argv, "--task") not in {"humaneval", "mbpp"}
                or _cli_arg_value(sys.argv, "--code-prompt-mode") not in {None, "instruct"}):
            raise ValueError("The task-description format requires --task humaneval/mbpp --code-prompt-mode instruct")
        if task == "mbpp" and _cli_arg_value(sys.argv, "--num-shots") != "0":
            raise ValueError("MBPP task-description format requires explicit --num-shots 0")
        import rockso_code_eval
        from code_prompt_formats import humaneval_task_prompt, mbpp_task_prompt, approach_first_task_prompt
        original_builder = rockso_code_eval.build_prompt
        def task_prompt_builder(task, row, shots, mode="benchmark", num_shots=None):
            prompt, stops, nshots = original_builder(task, row, shots, mode, num_shots)
            if task == "humaneval":
                prompt = humaneval_task_prompt(row["prompt"], row["entry_point"])
            else:
                prompt = mbpp_task_prompt(row["text"], row["test_list"])
            if _CODE_PROMPT_FORMAT == "approach_first":
                prompt = approach_first_task_prompt(prompt)
            return prompt, stops, nshots
        rockso_code_eval.build_prompt = task_prompt_builder

    base.model_logits = model_logits
    base.model_full_logits = model_full_logits
    base.continuation_logprob = continuation_logprob
    base.continuation_logprob_stats = continuation_logprob_stats
    base.continuation_logprob_next_only = continuation_logprob_next_only
    base.score_rc_rows = score_rc_rows
    base.generate_greedy_batch = generate_greedy_batch
    base.encode_gen_prompt = encode_gen_prompt
    base.import_implicit_loader = import_implicit_loader
    base.run_mcq_task = run_mcq_task
    base.run_generation_task = run_generation_task

    decode_mode = {"static": "static batched KV decode", "graph": "per-row CUDA graph decode (startup parity checked)"}.get(_DECODE_MODE, "per-row KV decode")
    mode = f"bidirectional PrefixLM prefill + causal {decode_mode}" if _PREFIXLM_PREFILL else f"causal prefill ablation + causal {decode_mode}"
    print(f"[prefixlm] Native HRM eval using {mode}")
    print(f"[prefixlm] RoPE impl: {_ROPE_IMPL}")
    if _FAST_UNSAFE_GSM8K:
        print("[prefixlm] FAST UNSAFE GSM8K mode: static decode parity is disabled; use for triage, not final benchmark provenance")
    exit_code = base.main()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
