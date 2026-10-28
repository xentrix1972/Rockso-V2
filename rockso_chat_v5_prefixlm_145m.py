#!/usr/bin/env python3
"""
Rockso Chat CLI v5 — PrefixLM prefill edition
=============================================
Key improvements over v2:
  - KV cache: ~10x faster generation (no full recompute per token)
  - PrefixLM prefill: bidirectional prompt attention, causal KV decode
  - Proper sampling: min_p only, no stacked penalty chaos
  - Fixed repetition penalty (correct logit handling)
  - Better prompt formats that match pretraining data patterns
  - Seed control for reproducible outputs
  - Multi-prompt templates (encyclopedia, QA, essay, etc.)

Usage:
    python rockso_chat_v5_prefixlm_145m.py --checkpoint /path/to/latest.pt --tokenizer /path/to/tokenizer
    python rockso_chat_v5_prefixlm_145m.py --checkpoint /path/to/latest.pt --tokenizer /path/to/tokenizer --style qa --temp 0.4
    python rockso_chat_v5_prefixlm_145m.py --checkpoint /path/to/latest.pt --tokenizer /path/to/tokenizer --seed 42
"""

import os
import sys
import argparse
import readline
import time
import json
from pathlib import Path
from itertools import permutations
from typing import Optional, List, Dict, Tuple
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


def _cfg_get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _install_eval_runtime_guards():
    # Keep TorchAO/allocator behavior aligned with the working PrefixLM eval path.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "backend:native")
    try:
        from eval_prefixlm_ppl import _install_eval_torchao_stubs
        _install_eval_torchao_stubs()
    except Exception:
        pass


def _rockso_resolve_dtype(dtype: str, device: str = "cuda"):
    """Resolve user dtype string to a torch dtype or None for native/checkpoint dtype."""
    d = (dtype or "bf16").lower().strip()
    if d in ("native", "none", "checkpoint"):
        return None
    if d in ("fp32", "float32", "f32"):
        return torch.float32
    if d in ("bf16", "bfloat16"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}. Use bf16, float32/fp32, fp16, or native.")


# ============================================================================
# MODEL DEFINITION (matches trainer checkpoint structure)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(dtype) * self.weight.to(dtype)


class QK_RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        x_f32 = x.float()
        norm = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


class SquaredReLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, linear_cls=nn.Linear):
        super().__init__()
        scaled_d_ff = int(d_ff * 1.5)
        self.w1 = linear_cls(d_model, scaled_d_ff, bias=False)
        self.w2 = linear_cls(scaled_d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)).pow(2))

SwiGLU = SquaredReLU


class LatentSquaredReLU(nn.Module):
    def __init__(self, latent_dim: int, expert_ff: int):
        super().__init__()
        scaled_ff = int(expert_ff * 1.5)
        self.w1 = nn.Linear(latent_dim, scaled_ff, bias=False)
        self.w2 = nn.Linear(scaled_ff, latent_dim, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)).pow(2))


class TopKRouter(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2,
                 route_scale: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = float(route_scale)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))
        self.register_buffer('ema_rel_dev', torch.zeros(num_experts))
        self.register_buffer('bias_update_counter', torch.zeros((), dtype=torch.int64))
        self.register_buffer('router_entropy_sum', torch.tensor(0.0))
        self.register_buffer('router_entropy_count', torch.tensor(0.0))
        self.register_buffer('ema_maxvio', torch.tensor(0.0))

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        raw_logits = self.gate(x_flat)
        biased_logits = raw_logits + self.expert_bias.to(raw_logits.dtype)
        top_logits, top_indices = torch.topk(biased_logits, self.top_k, dim=-1)
        selected_raw = torch.gather(raw_logits, dim=-1, index=top_indices)
        top_weights = F.sigmoid(selected_raw)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)
        top_weights = top_weights * self.route_scale
        return top_indices, top_weights, raw_logits


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, use_shared_expert=True,
                 route_scale=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopKRouter(d_model, num_experts, top_k, route_scale=route_scale)
        expert_ff = d_ff // 4
        self.experts = nn.ModuleList([SwiGLU(d_model, expert_ff) for _ in range(num_experts)])
        self.shared_expert = SwiGLU(d_model, d_ff // 2) if use_shared_expert else None

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        expert_indices, expert_weights, _ = self.router(x)
        final_out = torch.zeros_like(x_flat)
        if self.shared_expert is not None:
            final_out = final_out + self.shared_expert(x_flat)
        flat_indices = expert_indices.view(-1)
        flat_weights = expert_weights.view(-1)
        x_repeated = x_flat.repeat_interleave(self.top_k, dim=0)
        results_buffer = torch.zeros_like(x_repeated)
        for i, expert in enumerate(self.experts):
            mask = (flat_indices == i)
            if mask.any():
                results_buffer[mask] = expert(x_repeated[mask]).to(results_buffer.dtype)
        weighted = results_buffer * flat_weights.unsqueeze(-1)
        weighted = weighted.view(B * T, self.top_k, D).sum(dim=1)
        final_out = final_out + weighted
        return final_out.view(B, T, D)


class LatentMoELayer(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2,
                 use_shared_expert=True, latent_ratio=4, route_scale=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        self.latent_dim = d_model // latent_ratio
        self.down_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.up_proj = nn.Linear(self.latent_dim, d_model, bias=False)
        self.router = TopKRouter(d_model, num_experts, top_k, route_scale=route_scale)
        self.shared_expert = SwiGLU(d_model, d_ff // 4) if use_shared_expert else None
        expert_ff = 8192
        self.experts = nn.ModuleList([
            LatentSquaredReLU(self.latent_dim, expert_ff)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        expert_indices, expert_weights, _ = self.router(x)
        final_out = torch.zeros_like(x_flat)
        if self.shared_expert is not None:
            final_out = final_out + self.shared_expert(x_flat)
        x_latent = self.down_proj(x_flat)
        flat_indices = expert_indices.view(-1)
        flat_weights = expert_weights.view(-1)
        x_latent_repeated = x_latent.repeat_interleave(self.top_k, dim=0)
        results_buffer = torch.zeros_like(x_latent_repeated)
        for i in range(self.num_experts):
            mask = (flat_indices == i)
            if mask.any():
                results_buffer[mask] = self.experts[i](x_latent_repeated[mask]).to(results_buffer.dtype)
        weighted = results_buffer * flat_weights.unsqueeze(-1)
        weighted = weighted.view(B * T, self.top_k, self.latent_dim).sum(dim=1)
        final_out = final_out + self.up_proj(weighted)
        return final_out.view(B, T, D)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids):
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


_ROPE_IMPL = "legacy"


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _resolve_rope_impl(cfg=None, path: str = "", requested: str = "auto") -> str:
    req = (requested or "auto").lower().strip()
    if req in {"rotate_half", "rotate-half", "rotatehalf", "hrm", "olmo"}:
        return "rotate_half"
    if req in {"legacy", "interleaved", "pair"}:
        return "legacy"
    if req != "auto":
        raise ValueError(f"Unknown rope_impl: {requested}")

    trained = None
    if isinstance(cfg, dict):
        trained = cfg.get("rope_impl") or cfg.get("rope_convention")
    else:
        trained = getattr(cfg, "rope_impl", None) or getattr(cfg, "rope_convention", None)
    if trained:
        return _resolve_rope_impl(None, "", str(trained))

    p = str(path).lower()
    if (
        "fixed" in p
        or "rotate" in p
        or "prod_fixed" in p
        or "hrm_145m" in p
        or "main_code_20b" in p
    ):
        return "rotate_half"
    return "legacy"


def apply_rope(x, cos, sin):
    dtype = x.dtype
    cos = cos.unsqueeze(1).to(dtype)
    sin = sin.unsqueeze(1).to(dtype)
    if _ROPE_IMPL == "rotate_half":
        return (x * cos + _rotate_half(x) * sin).to(dtype)
    x1 = x[..., ::2]; x2 = x[..., 1::2]
    xr = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    return x * cos + xr * sin


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, d_ff, n_kv_heads=None,
                 use_moe=False, num_experts=8, moe_top_k=2,
                 use_shared_expert=True, value_residual=False,
                 use_latent_moe=False, latent_ratio=4, route_scale=1.0):
        super().__init__()
        self.n_head = n_head
        self.n_kv_heads = n_kv_heads or n_head
        self.head_dim = n_embd // n_head
        self.n_embd = n_embd

        self.norm1 = RMSNorm(n_embd)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(n_embd, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.q_norm = QK_RMSNorm(self.head_dim)
        self.k_norm = QK_RMSNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm2 = RMSNorm(n_embd)

        if value_residual:
            self.vr_alpha1 = nn.Parameter(torch.tensor([1.0]))
            self.vr_alpha2 = nn.Parameter(torch.tensor([0.0]))
            self.vr_scale = nn.Parameter(torch.tensor([1.0]))
        else:
            self.register_buffer("vr_alpha1", torch.tensor([1.0]), persistent=False)
            self.register_buffer("vr_alpha2", torch.tensor([0.0]), persistent=False)
            self.register_buffer("vr_scale", torch.tensor([1.0]), persistent=False)
        self.value_residual = value_residual

        if use_moe:
            if use_latent_moe:
                self.mlp = LatentMoELayer(n_embd, d_ff, num_experts, moe_top_k,
                                          use_shared_expert, latent_ratio, route_scale)
            else:
                self.mlp = MoELayer(n_embd, d_ff, num_experts, moe_top_k, use_shared_expert,
                                    route_scale)
        else:
            self.mlp = SwiGLU(n_embd, d_ff)

    def forward(self, x, position_ids, kv_cache=None, v1=None,
                prefill_causal: bool = True):
        """
        Forward with optional KV cache and value-residual threading.

        Args:
            x: [B, T, D] input
            position_ids: [B, T] positions
            kv_cache: tuple (cached_k, cached_v) or None
            v1: layer-0's pre-GQA V projection [B, Hkv, T, D] for value-residual
                mixing. Pass None for layer 0 (skips mixing); pass layer-0's v_out
                for layers 1+. Ignored if self.value_residual=False.

        Returns:
            output: [B, T, D]
            new_kv_cache: tuple (k, v)  — V is post-mixing if value_residual is on
            v_out: pre-mixing V from this layer's projection (for v1 threading)
        """
        B, T, _ = x.shape
        h = self.norm1(x)

        q = self.q_proj(h).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Save fresh V (pre-RoPE, pre-GQA) for v1 threading to subsequent layers
        v_out = v

        q = self.q_norm(q)
        k = self.k_norm(k)

        # Value-residual mixing on fresh pre-GQA V (math is equivalent to post-GQA
        # mixing because alpha1/alpha2/scale are scalars and v1 is expanded the
        # same way as v during GQA — the broadcast factors out).
        if self.value_residual and v1 is not None:
            denom = torch.rsqrt(self.vr_alpha1 ** 2 + self.vr_alpha2 ** 2 + 1e-8)
            v = self.vr_scale * (self.vr_alpha1 * v + self.vr_alpha2 * v1) * denom

        cos, sin = self.rope(position_ids)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # KV cache stores post-mix V, post-RoPE K
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)
        new_kv_cache = (k, v)

        # GQA expansion (after cache concat for memory efficiency)
        if self.n_kv_heads < self.n_head:
            rep = self.n_head // self.n_kv_heads
            k_exp = k.repeat_interleave(rep, dim=1)
            v_exp = v.repeat_interleave(rep, dim=1)
        else:
            k_exp, v_exp = k, v

        # PrefixLM prefill is the fast dense path: full bidirectional prompt,
        # no materialized mask object. Cached decode must be non-causal here
        # because the single query should see the entire cached prefix.
        is_causal = bool(prefill_causal) if kv_cache is None else False
        attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        attn = attn.transpose(1, 2).reshape(B, T, self.n_embd)
        x = x + self.out_proj(attn)

        h2 = self.norm2(x)
        if isinstance(self.mlp, (MoELayer, LatentMoELayer)):
            x = x + self.mlp(h2)
        else:
            x = x + self.mlp(h2)

        return x, new_kv_cache, v_out


class RocksoLM(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_embd, block_size,
                 num_kv_heads=None, use_moe=True, num_experts=8, moe_top_k=2,
                 use_shared_expert=True, dense_first_layers=0, value_residual=False,
                 dense_d_ff=None, moe_d_ff=None, use_latent_moe=False, latent_ratio=4,
                 route_scale=1.0):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.embed = nn.Embedding(vocab_size, n_embd)

        default_d_ff = 4 * n_embd
        _dense_d_ff = dense_d_ff if dense_d_ff is not None else default_d_ff
        _moe_d_ff = moe_d_ff if moe_d_ff is not None else default_d_ff

        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            layer_use_moe = use_moe and (i >= dense_first_layers)
            layer_d_ff = _moe_d_ff if layer_use_moe else _dense_d_ff
            self.blocks.append(TransformerBlock(
                n_embd, n_head, layer_d_ff, n_kv_heads=num_kv_heads,
                use_moe=layer_use_moe, num_experts=num_experts,
                moe_top_k=moe_top_k, use_shared_expert=use_shared_expert,
                value_residual=value_residual,
                use_latent_moe=(use_latent_moe and layer_use_moe),
                latent_ratio=latent_ratio,
                route_scale=route_scale,
            ))

        self.norm = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids, position_ids=None, kv_caches=None,
                prefill_causal: bool = None):
        """
        Forward pass with optional KV caches.

        Args:
            input_ids: [B, T]
            position_ids: [B, T]
            kv_caches: list of (k, v) tuples per layer, or None

        Returns:
            logits: [B, T, vocab]
            new_kv_caches: list of (k, v) tuples
        """
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        if prefill_causal is None:
            prefill_causal = not bool(getattr(self, "prefixlm_prefill", True))

        x = self.embed(input_ids)

        new_kv_caches = []
        v1 = None  # captured from layer 0 for value-residual threading
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache, v_out = block(
                x, position_ids, kv_cache=cache, v1=v1,
                prefill_causal=prefill_causal,
            )
            new_kv_caches.append(new_cache)
            if v1 is None:
                v1 = v_out  # capture layer 0's V; never overwrite

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches


# ============================================================================
# SAMPLING
# ============================================================================

def apply_repetition_penalty(logits: torch.Tensor, token_ids: torch.Tensor,
                             penalty: float = 1.1) -> torch.Tensor:
    """Correct repetition penalty implementation.

    For tokens that have appeared in the context:
    - Positive logits are divided by penalty (made less likely)
    - Negative logits are multiplied by penalty (made even less likely)
    """
    if penalty == 1.0:
        return logits

    unique_ids = token_ids.unique()
    gathered = logits[0, unique_ids]
    # Correct: divide positive, multiply negative
    gathered = torch.where(gathered > 0, gathered / penalty, gathered * penalty)
    logits[0, unique_ids] = gathered
    return logits


def sample_token(logits_1d: torch.Tensor, temperature: float = 0.5,
                 min_p: float = 0.05, rep_penalty_logits: torch.Tensor = None) -> int:
    """Clean sampling with min_p only. No stacked filters.

    For early-training base models, simplicity beats complexity.
    min_p naturally adapts to the model's per-token confidence.
    """
    if temperature == 0:
        return logits_1d.argmax().item()

    logits_1d = logits_1d / temperature
    probs = F.softmax(logits_1d, dim=-1)

    # min_p: keep tokens with prob >= min_p * max_prob
    if min_p > 0:
        max_prob = probs.max()
        mask = probs < (min_p * max_prob)
        probs[mask] = 0.0
        # Renormalize
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            # Fallback: just take argmax
            return logits_1d.argmax().item()

    return torch.multinomial(probs, num_samples=1).item()


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

HRM_SPECIAL_TOKENS = {
    "boq": "<|im_start|>",
    "eoq": "<|im_end|>",
    "eoa": "<|box_end|>",
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}

HRM_CONDITION_PARTS = ("direct", "cot", "noisy", "synth")


def parse_hrm_condition(condition: str = "direct") -> Tuple[str, ...]:
    parts = tuple(p.strip().lower() for p in str(condition or "direct").split(",") if p.strip())
    if not parts:
        parts = ("direct",)
    unknown = [p for p in parts if p not in HRM_CONDITION_PARTS]
    if unknown:
        raise ValueError(f"Unknown HRM condition(s): {unknown}. Known: direct,cot,noisy,synth")
    return parts


def normalize_hrm_condition(condition: str = "direct") -> str:
    return ",".join(parse_hrm_condition(condition))


def build_hrm_condition_tokens(condition: str = "direct") -> str:
    return "".join(HRM_SPECIAL_TOKENS[p] for p in parse_hrm_condition(condition))


def hrm_condition_style_name(condition: str) -> str:
    return "hrm_cond_" + "_".join(parse_hrm_condition(condition))


def build_hrm_prompt(user_input: str, system_prompt: str = None, condition: str = "direct") -> str:
    cond_tok = build_hrm_condition_tokens(condition)
    instruction = user_input.strip() if not system_prompt else f"{system_prompt.strip()}\n\n{user_input.strip()}"
    return (
        f"{HRM_SPECIAL_TOKENS['boq']}{cond_tok}"
        f"{instruction}"
        f"{HRM_SPECIAL_TOKENS['eoq']}"
    )


def build_hrm_casual_prompt(user_input: str, system_prompt: str = None) -> str:
    instruction = system_prompt or "Continue this casual conversation."
    instruction = (
        f"{instruction.strip()}\n\n"
        f"User: Hello\n"
        f"Rockso: Hi there!\n"
        f"User: {user_input.strip()}\n"
        f"Rockso:"
    )
    return build_hrm_prompt(instruction, None, "cot")


def build_python_code_prompt(user_input: str, system_prompt: str = None,
                             condition: str = "synth,cot") -> str:
    task = user_input.strip()
    if system_prompt:
        task = f"{system_prompt.strip()}\n\n{task}"
    instruction = (
        "Language: Python 3.11\n\n"
        "Implement the following Python solution:\n\n"
        f"{task}\n\n"
        "Return exactly one Python code block and no explanation."
    )
    return build_hrm_prompt(instruction, None, condition)


def build_python_complete_prompt(user_input: str, system_prompt: str = None,
                                 condition: str = "synth,cot") -> str:
    code = user_input.strip()
    low = code.lower()
    if low.startswith("complete def "):
        code = code[len("complete "):].lstrip()
    elif low.startswith("complete class "):
        code = code[len("complete "):].lstrip()
    elif low.startswith("complete ```"):
        code = code[len("complete "):].lstrip()
    if system_prompt:
        code = f"{system_prompt.strip()}\n\n{code}"
    instruction = (
        "Language: Python 3.11\n\n"
        "Complete the following Python code. Preserve the requested signature and behavior:\n\n"
        f"{code}\n\n"
        "Return exactly one Python code block and no explanation."
    )
    return build_hrm_prompt(instruction, None, condition)


def build_code_explain_prompt(user_input: str, system_prompt: str = None,
                              condition: str = "synth,cot") -> str:
    task = user_input.strip()
    if system_prompt:
        task = f"{system_prompt.strip()}\n\n{task}"
    instruction = (
        "Language: Python 3.11\n\n"
        "Solve the following programming task. Include concise reasoning only inside code comments when helpful:\n\n"
        f"{task}\n\n"
        "Return exactly one Python code block and no explanation."
    )
    return build_hrm_prompt(instruction, None, condition)


def build_python_task_prompt(user_input: str, system_prompt: str = None,
                             condition: str = "synth,cot") -> str:
    """Description-first code request, following the V2 task evaluation format."""
    task = user_input.strip()
    if system_prompt:
        task = f"{system_prompt.strip()}\n\n{task}"
    instruction = (
        f"In Python, solve the following task:\n\n{task}\n\n"
        "Preserve any requested function names and signatures.\n"
        "Return the complete implementation, including any needed imports, in one Python code block."
    )
    return build_hrm_prompt(instruction, None, condition)


def clean_response_for_style(response: str, style: str) -> str:
    if style != "casual":
        return response
    text = response.strip()
    for prefix in ("Rockso:", "Assistant:", "Bot:"):
        if text.lower().startswith(prefix.lower()):
            return text[len(prefix):].strip()
    return text


def special_token_ids(tokenizer, tokens: List[str]) -> List[int]:
    ids = []
    for tok in tokens:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            ids.append(int(tid))
    return ids


def suppressed_special_token_ids(tokenizer) -> List[int]:
    """Mask actual control/sentinel tokens, except the HRM answer-stop token.

    Never infer "special" from a leading ``<`` or trailing ``>``. The Rockso
    tokenizer contains ordinary Python/code tokens such as ``<``, ``<=``,
    ``>>`` and `` ->``; the old heuristic silently made those impossible to
    generate.
    """
    cached = getattr(tokenizer, "_rockso_suppressed_control_ids", None)
    if cached is not None:
        return list(cached)

    eoa_id = tokenizer.convert_tokens_to_ids(HRM_SPECIAL_TOKENS["eoa"])
    out = set()
    vocab = tokenizer.get_vocab()

    # Hugging Face-declared special IDs (BOS/EOS/PAD, depending on tokenizer).
    for idx in getattr(tokenizer, "all_special_ids", []) or []:
        if idx is not None and int(idx) != eoa_id:
            out.add(int(idx))

    # The HRM/Qwen control namespace is exact: <|name|>. This deliberately
    # excludes ordinary operators and HTML/code fragments beginning with "<".
    for tok, idx in vocab.items():
        if int(idx) == eoa_id:
            continue
        if tok.startswith("<|") and tok.endswith("|>"):
            out.add(int(idx))

    # Tool/reasoning delimiters are control tags but do not use <|...|>.
    for tok in (
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>",
    ):
        idx = vocab.get(tok)
        if idx is not None and int(idx) != eoa_id:
            out.add(int(idx))

    # Fail closed if a future tokenizer/config change ever masks code syntax.
    leaked_syntax = [
        tok for tok in ("<", ">", "<=", "<<", ">>", "Ġ->")
        if tok in vocab and int(vocab[tok]) in out
    ]
    if leaked_syntax:
        raise RuntimeError(
            f"Control-token suppression captured ordinary syntax: {leaked_syntax}"
        )

    result = sorted(out)
    setattr(tokenizer, "_rockso_suppressed_control_ids", tuple(result))
    return result


PROMPT_STYLES = {
    "hrm": {
        "desc": "HRM I/O format — direct task-completion prompt",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "direct"),
        "stops": [],
    },
    "hrm_cot": {
        "desc": "HRM I/O format — synth,cot math/reasoning condition",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "synth,cot"),
        "stops": [],
    },
    "hrm_direct": {
        "desc": "HRM I/O format — direct condition",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "direct"),
        "stops": [],
    },
    "hrm_synth_direct": {
        "desc": "HRM I/O format — synth,direct condition",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "synth,direct"),
        "stops": [],
    },
    "hrm_synth_cot": {
        "desc": "HRM I/O format — synth,cot condition",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "synth,cot"),
        "stops": [],
    },
    "casual": {
        "desc": "HRM-wrapped casual chatbot lane — cot condition with stateless tiny example scaffold",
        "build": build_hrm_casual_prompt,
        "stops": [],
    },
    "qa": {
        "desc": "HRM-wrapped Q&A — direct condition, short factual answers",
        "build": lambda q, sys: build_hrm_prompt(f"Question: {q}\nAnswer:", sys, "direct"),
        "stops": [],
    },
    "qa_synth": {
        "desc": "HRM-wrapped Q&A — synth,direct condition",
        "build": lambda q, sys: build_hrm_prompt(f"Question: {q}\nAnswer:", sys, "synth,direct"),
        "stops": [],
    },
    "code": {
        "desc": "Python code generation — synth,cot code-corpus prompt",
        "build": build_python_code_prompt,
        "stops": [],
    },
    "code_complete": {
        "desc": "Python code completion — synth,cot code-corpus prompt",
        "build": build_python_complete_prompt,
        "stops": [],
    },
    "code_task": {
        "desc": "Python task description — synth,cot, complete implementation",
        "build": build_python_task_prompt,
        "stops": [],
    },
    "code_task_legacy": {
        "desc": "Previous Language: Python 3.11 task prompt — synth,cot",
        "build": build_code_explain_prompt,
        "stops": [],
    },
    "tools": {
        "desc": "Native tool-schema question — direct; calls displayed, never executed",
        "build": lambda q, sys: build_hrm_prompt(q, sys, "direct"),
        "stops": [],
    },
    "raw_qa": {
        "desc": "Legacy raw Q&A format — no HRM wrapper",
        "build": lambda q, sys: f"Question: {q}\nAnswer:",
        "stops": ["\nQuestion:", "\n\n", "\nQ:"],
    },
    "encyclopedia": {
        "desc": "Encyclopedia entry — informative paragraph",
        "build": lambda q, sys: f"{q}\n\n",
        "stops": ["\n\n\n", "\nReferences", "\nSee also", "\nExternal links"],
    },
    "explain": {
        "desc": "Explanation format — clear, educational",
        "build": lambda q, sys: f"Explain {q.lower().lstrip('explain ').lstrip('what is ').lstrip('how ')}.\n\n",
        "stops": ["\n\n\n"],
    },
    "chat": {
        "desc": "User/Assistant format — conversational",
        "build": lambda q, sys: (
            f"{sys}\n\nUser: {q}\nAssistant:" if sys
            else f"User: {q}\nAssistant:"
        ),
        "stops": ["\nUser:", "\n\nUser", "\n\n\n"],
    },
    "complete": {
        "desc": "Raw completion — just continue the text",
        "build": lambda q, sys: q,
        "stops": [],
    },
    "instruct": {
        "desc": "Instruction format with ### markers",
        "build": lambda q, sys: (
            f"### System:\n{sys}\n\n### User:\n{q}\n\n### Assistant:\n" if sys
            else f"### User:\n{q}\n\n### Assistant:\n"
        ),
        "stops": ["### User:", "\n###", "\n\n\n"],
    },
}

HRM_STYLE_CONDITIONS = {
    "hrm": "direct",
    "hrm_direct": "direct",
    "hrm_cot": "synth,cot",
    "hrm_synth_cot": "synth,cot",
    "hrm_synth_direct": "synth,direct",
    "qa": "direct",
    "qa_synth": "synth,direct",
    "code": "synth,cot",
    "code_complete": "synth,cot",
    "code_task": "synth,cot",
    "code_task_legacy": "synth,cot",
    "tools": "direct",
    "casual": "cot",
}


def _make_hrm_condition_style(condition: str) -> Dict:
    condition = normalize_hrm_condition(condition)
    return {
        "desc": f"HRM I/O condition — {condition}",
        "build": lambda q, sys, c=condition: build_hrm_prompt(q, sys, c),
        "stops": [],
        "condition": condition,
    }


def _all_hrm_condition_styles() -> Dict[str, Dict]:
    styles = {}
    for n in range(1, len(HRM_CONDITION_PARTS) + 1):
        for combo in permutations(HRM_CONDITION_PARTS, n):
            condition = ",".join(combo)
            styles[hrm_condition_style_name(condition)] = _make_hrm_condition_style(condition)
    return styles


for _name, _cfg in _all_hrm_condition_styles().items():
    PROMPT_STYLES.setdefault(_name, _cfg)


def style_condition(style: str) -> Optional[str]:
    cfg = PROMPT_STYLES.get(style)
    if cfg and "condition" in cfg:
        return cfg["condition"]
    return HRM_STYLE_CONDITIONS.get(style)


def build_prompt(user_input: str, style: str, system_prompt: str = None,
                 history: List[Dict] = None, condition_override: str = None,
                 tools: Optional[List[Dict]] = None) -> str:
    """Build prompt from template with optional history."""
    cfg = PROMPT_STYLES.get(style, PROMPT_STYLES["qa"])
    forced_condition = normalize_hrm_condition(condition_override) if condition_override else None
    condition = forced_condition or style_condition(style)

    # A condition override changes only the HRM routing tokens. It must not
    # discard the selected code style's training-matched instruction scaffold.
    if condition and style == "code":
        return build_python_code_prompt(user_input, system_prompt, condition)
    if condition and style == "code_complete":
        return build_python_complete_prompt(user_input, system_prompt, condition)
    if condition and style == "code_task":
        return build_python_task_prompt(user_input, system_prompt, condition)
    if condition and style == "code_task_legacy":
        return build_code_explain_prompt(user_input, system_prompt, condition)
    if style == "tools":
        if not tools:
            raise ValueError('Load tool schemas with /tools /path/to/tools.json first.')
        from tool_prompt_formats import render_tool_question
        return build_hrm_prompt(render_tool_question(tools, user_input, system_prompt), None, condition)

    if condition and style in ("qa", "qa_synth"):
        parts = []
        if history:
            for turn in history:
                parts.append(f"Question: {turn['user']}\nAnswer: {turn['assistant']}")
        parts.append(f"Question: {user_input}\nAnswer:")
        return build_hrm_prompt("\n\n".join(parts), system_prompt, condition)

    if condition and style == "casual":
        instruction = system_prompt or "Continue this casual conversation."
        parts = [instruction.strip(), "", "User: Hello", "Rockso: Hi there!"]
        if history:
            for turn in history:
                parts.append(f"User: {turn['user']}")
                parts.append(f"Rockso: {turn['assistant']}")
        parts.append(f"User: {user_input}")
        parts.append("Rockso:")
        return build_hrm_prompt("\n".join(parts), None, condition)

    if condition and (history or forced_condition or style.startswith("hrm")):
        parts = []
        if history:
            for turn in history:
                parts.append(f"Question: {turn['user']}\nAnswer: {turn['assistant']}")
            parts.append(f"Question: {user_input}\nAnswer:")
            return build_hrm_prompt("\n\n".join(parts), system_prompt, condition)
        return build_hrm_prompt(user_input, system_prompt, condition)

    # For chat/instruct modes, prepend history
    if style in ("chat", "instruct") and history:
        parts = []
        if system_prompt and style == "chat":
            parts.append(f"{system_prompt}\n\n")
        elif system_prompt and style == "instruct":
            parts.append(f"### System:\n{system_prompt}\n\n")

        for turn in history:
            if style == "chat":
                parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n")
            else:
                parts.append(f"### User:\n{turn['user']}\n\n### Assistant:\n{turn['assistant']}\n\n")

        if style == "chat":
            parts.append(f"User: {user_input}\nAssistant:")
        else:
            parts.append(f"### User:\n{user_input}\n\n### Assistant:\n")
        return ''.join(parts)

    return cfg["build"](user_input, system_prompt)


# ============================================================================
# GENERATION WITH KV CACHE
# ============================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200,
             temperature: float = 0.5, min_p: float = 0.05,
             repetition_penalty: float = 1.1,
             stop_strings: List[str] = None,
             stop_tokens: List[int] = None,
             suppress_special: bool = True,
             device: str = 'cuda', stream: bool = True,
             rep_detect: bool = True,
             seed: int = None,
             allow_tool_calls: bool = False) -> Tuple[str, dict]:
    """Generate with KV cache for fast autoregressive decoding."""

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    model.eval()

    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    prompt_len = input_ids.size(1)
    if prompt_len >= model.block_size:
        raise ValueError(
            f"Prompt is {prompt_len} tokens but model context is {model.block_size}. "
            "Refusing to remove the HRM prefix or task contract. Shorten the prompt."
        )
    requested_max_new_tokens = max(0, int(max_new_tokens))
    effective_max_new_tokens = min(
        requested_max_new_tokens,
        model.block_size - prompt_len,
    )
    if effective_max_new_tokens < requested_max_new_tokens:
        print(
            f"  [context] prompt={prompt_len}; generation budget reduced "
            f"{requested_max_new_tokens}->{effective_max_new_tokens} tokens "
            f"to preserve the complete prompt."
        )

    # Special stop tokens
    special_stops = set()
    if tokenizer.bos_token_id is not None:
        special_stops.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_stops.add(tokenizer.eos_token_id)
    if stop_tokens:
        special_stops.update(stop_tokens)
    special_stops.update(special_token_ids(
        tokenizer,
        [
            HRM_SPECIAL_TOKENS["eoa"],
            HRM_SPECIAL_TOKENS["boq"],
            HRM_SPECIAL_TOKENS["eoq"],
        ],
    ))
    suppressed_ids = suppressed_special_token_ids(tokenizer) if suppress_special else []
    if allow_tool_calls:
        tool_ids = set(special_token_ids(tokenizer, ['<tool_call>', '</tool_call>']))
        suppressed_ids = [idx for idx in suppressed_ids if idx not in tool_ids]

    # Prefill: process entire prompt at once
    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)
    logits, kv_caches = model(input_ids, position_ids)

    generated_tokens = []
    generated_text = ""           # init so EOS-on-first-token doesn't UnboundLocalError
    prev_text = ""
    stop_reason = "max_tokens"
    timer = ((lambda: time.clock_gettime(time.CLOCK_MONOTONIC_RAW))
             if hasattr(time, 'CLOCK_MONOTONIC_RAW') else time.perf_counter)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    t0 = timer()

    if stream:
        sys.stdout.write("\033[94mRockso:\033[0m ")
        sys.stdout.flush()

    for step in range(effective_max_new_tokens):
        # Get logits for last position
        # Graph outputs are inference tensors. Apply chat-only token filtering
        # to a normal copy, keeping the cached graph output immutable.
        next_logits = logits[0, -1, :].unsqueeze(0).clone()
        if suppressed_ids:
            next_logits[:, suppressed_ids] = -torch.inf

        # Repetition penalty over all context
        if repetition_penalty and abs(repetition_penalty - 1.0) > 1e-6:
            all_ids = torch.cat([input_ids[0], torch.tensor(generated_tokens, dtype=torch.long, device=device)])
            next_logits = apply_repetition_penalty(next_logits, all_ids, repetition_penalty)

        # Sample
        token = sample_token(next_logits[0], temperature, min_p)

        # Check stop tokens
        if token in special_stops:
            stop_reason = "eos"
            break

        generated_tokens.append(token)

        # Decode and check stop strings
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=not allow_tool_calls)

        if stop_strings:
            stopped = False
            for ss in stop_strings:
                if ss in generated_text:
                    generated_text = generated_text.split(ss)[0]
                    stop_reason = f"stop_string: {repr(ss)}"
                    stopped = True
                    break
            if stopped:
                if stream:
                    new_chunk = generated_text[len(prev_text):]
                    if new_chunk:
                        sys.stdout.write(new_chunk)
                    sys.stdout.flush()
                break

        # Detect a repeated 4-gram before streaming its completing token. Keep
        # the already-streamed prefix, discard only the new token, and keep
        # token counts consistent with the returned text.
        if rep_detect and len(generated_tokens) > 40:
            recent = generated_tokens[-60:]
            ngrams = [tuple(recent[i:i+4]) for i in range(len(recent) - 3)]
            counts = Counter(ngrams)
            if any(c >= 3 for c in counts.values()):
                generated_tokens = generated_tokens[:-1]
                generated_text = tokenizer.decode(
                    generated_tokens, skip_special_tokens=not allow_tool_calls
                )
                stop_reason = "repetition_detected"
                break

        # Stream output incrementally
        if stream:
            new_chunk = generated_text[len(prev_text):]
            if new_chunk:
                sys.stdout.write(new_chunk)
                sys.stdout.flush()
            prev_text = generated_text

        # Check total length
        if prompt_len + len(generated_tokens) >= model.block_size:
            stop_reason = "context_limit"
            break

        # Next token: single-token forward with KV cache
        new_input = torch.tensor([[token]], device=device)
        new_pos = torch.tensor([[prompt_len + step]], device=device)
        logits, kv_caches = model(new_input, new_pos, kv_caches)

    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = timer() - t0
    tok_per_sec = len(generated_tokens) / elapsed if elapsed > 0 else 0

    if stream:
        print()  # newline after streaming

    stats = {
        "tokens": len(generated_tokens),
        "elapsed": elapsed,
        "tok_per_sec": tok_per_sec,
        "stop_reason": stop_reason,
        "prompt_tokens": prompt_len,
        "requested_max_new_tokens": requested_max_new_tokens,
        "effective_max_new_tokens": effective_max_new_tokens,
    }
    return generated_text.strip(), stats


# ============================================================================
# CHECKPOINT LOADING
# ============================================================================

def _resolve_prefill_mode(cfg, requested):
    """requested: None=auto from checkpoint provenance, True/False=forced."""
    trained = str(_cfg_get(cfg, "attn_mask_mode", None) or "causal")
    auto = trained == "prefixlm"
    if requested is None:
        return auto, trained, False
    return bool(requested), trained, bool(requested) != auto


def _enforce_fp32_norms(model):
    qk_count = 0
    norm_count = 0
    for module in model.modules():
        if isinstance(module, QK_RMSNorm):
            if module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
            qk_count += 1
        elif isinstance(module, RMSNorm):
            if module.weight.dtype != torch.float32:
                module.weight.data = module.weight.data.float()
            norm_count += 1
    return qk_count, norm_count


def _load_json_if_exists(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_export_or_checkpoint(path: str):
    """Load either a raw trainer checkpoint or an exported HF-style snapshot."""
    src = Path(path).expanduser()
    if src.is_dir():
        export_dir = src
        cfg = _load_json_if_exists(export_dir / "config.json")
        meta = _load_json_if_exists(export_dir / "export_meta.json")
        cfg = {**cfg, **(meta.get("cfg") or {})}
        weights_path = export_dir / (meta.get("weights") or "")
        if not weights_path.exists():
            for name in ("model.safetensors", "pytorch_model.bin"):
                candidate = export_dir / name
                if candidate.exists():
                    weights_path = candidate
                    break
        if not weights_path.exists():
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {export_dir}")
        print(f"Loading export: {export_dir}")
        print(f"  Weights: {weights_path.name}")
        if weights_path.suffix == ".safetensors":
            try:
                from safetensors.torch import load_file
            except Exception as exc:
                raise RuntimeError("safetensors is required to load model.safetensors") from exc
            raw_sd = load_file(str(weights_path), device="cpu")
        else:
            raw_sd = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        step = meta.get("step", "?")
        tokens = meta.get("tokens_seen", 0)
        return cfg, raw_sd, step, tokens, str(export_dir), "export"

    print(f"Loading checkpoint: {src}")
    ck = torch.load(str(src), map_location='cpu', mmap=True, weights_only=False)
    if isinstance(ck, dict) and ck.get("schema") in (
            "rockso_hybrid_checkpoint_1", "rockso_hybrid_export_1"):
        return ck["config"], ck, ck.get("step", "?"), ck.get("tokens_seen", 0), str(src), "hybrid"
    if isinstance(ck, dict) and "cfg" in ck and "model" in ck:
        return ck["cfg"], ck["model"], ck.get("step", "?"), ck.get("tokens_seen", 0), str(src), "checkpoint"

    # Allow direct .bin/.safetensors with sibling config/export_meta.
    export_dir = src.parent
    cfg = _load_json_if_exists(export_dir / "config.json")
    meta = _load_json_if_exists(export_dir / "export_meta.json")
    cfg = {**cfg, **(meta.get("cfg") or {})}
    step = meta.get("step", "?")
    tokens = meta.get("tokens_seen", 0)
    return cfg, ck, step, tokens, str(src), "export"


def _normalize_cfg(cfg: dict) -> dict:
    if cfg is None:
        cfg = {}
    elif not isinstance(cfg, dict):
        if hasattr(cfg, "__dict__"):
            cfg = vars(cfg)
        else:
            cfg = dict(cfg)
    else:
        cfg = dict(cfg)
    if "num_kv_heads" not in cfg and "num_key_value_heads" in cfg:
        cfg["num_kv_heads"] = cfg["num_key_value_heads"]
    if "block_size" not in cfg and "n_positions" in cfg:
        cfg["block_size"] = cfg["n_positions"]
    if "tie_embeddings" not in cfg and "tie_word_embeddings" in cfg:
        cfg["tie_embeddings"] = cfg["tie_word_embeddings"]
    if "attn_mask_mode" not in cfg and cfg.get("prefixlm_impl"):
        cfg["attn_mask_mode"] = "prefixlm"
    return cfg


def _strip_and_map_state_dict(raw_sd: dict) -> dict:
    """Map trainer/export naming onto this chat model's module names."""
    sd = {}
    name_map = {
        ".q.weight": ".q_proj.weight",
        ".k.weight": ".k_proj.weight",
        ".v.weight": ".v_proj.weight",
        ".o.weight": ".out_proj.weight",
    }
    for k, v in raw_sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        for old, new in name_map.items():
            if k.endswith(old):
                k = k[: -len(old)] + new
                break
        sd[k] = v
    return sd


def _infer_d_ff_from_state_dict(sd: dict, fallback: Optional[int], n_embd: int) -> int:
    w = sd.get("blocks.0.mlp.w1.weight")
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        inferred = int(round(w.shape[0] / 1.5))
        if fallback is not None and int(fallback) != inferred:
            print(
                f"  [WARN] cfg d_ff={int(fallback)} but checkpoint implies d_ff={inferred}; "
                "using checkpoint tensor shape"
            )
        return inferred
    if fallback is not None:
        return int(fallback)
    return 4 * int(n_embd)


def load_checkpoint(path: str, device: str = 'cuda', dtype: str = 'bf16',
                    prefixlm_prefill=None, rope_impl: str = "auto", tokenizer_path=None):
    """prefixlm_prefill: None=auto from cfg.attn_mask_mode, True/False=forced."""
    global _ROPE_IMPL
    import re
    _install_eval_runtime_guards()

    cfg, raw_sd, step, tokens, provenance_path, source_kind = _load_export_or_checkpoint(path)
    if source_kind == "hybrid":
        from hybrid_chat_adapter import load_chat_model
        return load_chat_model(raw_sd, device=device,
                               target_dtype=_rockso_resolve_dtype(dtype, device),
                               prefixlm_prefill=prefixlm_prefill, rope_impl=rope_impl,
                               tokenizer_path=tokenizer_path)
    cfg = _normalize_cfg(cfg)
    _ROPE_IMPL = _resolve_rope_impl(cfg, path, rope_impl)
    print(f"  Step: {step}, Tokens: {tokens:,}")
    print(f"  Source kind: {source_kind}")
    print(f"  RoPE impl: {_ROPE_IMPL} ({'auto' if rope_impl == 'auto' else 'FORCED by flag'})")

    # Filter FP8/MTP/scale artifacts FIRST so we autodetect from real weight keys.
    # NOTE: 'vr_alpha1', 'vr_alpha2', 'vr_scale' are REAL value-residual parameters —
    # the substring '_scale' in 'vr_scale' must NOT match the FP8 filter.
    sd = {}
    for k, v in _strip_and_map_state_dict(raw_sd).items():
        if '._extra_state' in k:
            continue
        # Exempt value-residual params from the FP8 filter
        if 'vr_' not in k:
            if '_scale' in k or '_amax' in k:
                continue
        if 'mtp_modules' in k:
            continue  # MTP not needed for inference
        if isinstance(v, torch.Tensor) and 'float8' in str(v.dtype):
            v = v.to(torch.bfloat16)
        sd[k] = v

    # ---- Auto-detect arch from state_dict (cfg may be missing keys; never trust defaults) ----
    sd_keys = set(sd.keys())
    has_experts   = any('.experts.' in k for k in sd_keys)
    has_router    = any('.router.gate' in k for k in sd_keys)
    has_lm_head_w = 'lm_head.weight' in sd_keys
    has_vr_param  = any(k.endswith('vr_alpha1') or k.endswith('vr_alpha2') or k.endswith('vr_scale')
                        for k in sd_keys)
    # Latent MoE projections live inside each block's ``mlp`` module.
    has_latent_moe = has_experts and any(
        re.search(r'^blocks\.\d+\.mlp\.(down_proj|up_proj)\.weight$', k) for k in sd_keys
    )

    detected_use_moe = has_experts and has_router

    # Detect dense-first-layers count: dense layers have exactly blocks.{i}.mlp.w1.weight
    # (MoE shared_expert lives at blocks.{i}.mlp.shared_expert.w1.weight — does NOT match)
    dense_pat = re.compile(r'^blocks\.(\d+)\.mlp\.w1\.weight$')
    dense_idxs = sorted({int(m.group(1)) for k in sd_keys for m in [dense_pat.match(k)] if m})
    if detected_use_moe and dense_idxs:
        # Sanity: dense layers should be a contiguous prefix
        detected_dense_first_layers = len(dense_idxs)
        if dense_idxs != list(range(detected_dense_first_layers)):
            print(f"  [WARN] non-contiguous dense layers detected at {dense_idxs} — using count")
    else:
        detected_dense_first_layers = 0

    # Cross-check vs cfg (informational; detection wins)
    for fld, det in [('use_moe', detected_use_moe),
                     ('use_latent_moe', has_latent_moe),
                     ('dense_first_layers', detected_dense_first_layers),
                     ('value_residual', has_vr_param)]:
        if fld in cfg and cfg[fld] != det:
            print(f"  [WARN] cfg says {fld}={cfg[fld]} but state_dict implies {det} — using state_dict")

    print(f"  Detected: use_moe={detected_use_moe} latent_moe={has_latent_moe} "
          f"dense_first_layers={detected_dense_first_layers} value_residual={has_vr_param}")

    dense_d_ff = _infer_d_ff_from_state_dict(
        sd,
        cfg.get('dense_d_ff', cfg.get('d_ff', None)),
        cfg.get('n_embd', 2048),
    )
    moe_d_ff = cfg.get('moe_d_ff', dense_d_ff)
    print(f"  FFN d_ff: dense={dense_d_ff} moe={moe_d_ff}")

    model = RocksoLM(
        vocab_size=cfg.get('vocab_size', 65536),
        n_layer=cfg.get('n_layer', 30),
        n_head=cfg.get('n_head', 32),
        n_embd=cfg.get('n_embd', 2048),
        block_size=cfg.get('block_size', 2048),
        num_kv_heads=cfg.get('num_kv_heads', cfg.get('n_head', 32)),
        use_moe=detected_use_moe,
        num_experts=cfg.get('num_experts', 8),
        moe_top_k=cfg.get('moe_top_k', 2),
        use_shared_expert=cfg.get('use_shared_expert', True),
        dense_first_layers=detected_dense_first_layers,
        value_residual=has_vr_param,
        dense_d_ff=dense_d_ff,
        moe_d_ff=moe_d_ff,
        use_latent_moe=has_latent_moe,
        latent_ratio=cfg.get('latent_ratio', 4),
        route_scale=cfg.get('route_scale', 1.0),
    )

    # Tie lm_head ↔ embed if the trainer tied them, OR if the saved state_dict
    # only contains embed.weight (PyTorch state_dict dedupes tied params).
    if cfg.get('tie_embeddings', False) or not has_lm_head_w:
        model.lm_head.weight = model.embed.weight
        print(f"  Tied lm_head ↔ embed")

    # Loud load: never silent-swallow missing/unexpected keys
    result = model.load_state_dict(sd, strict=False)
    missing    = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)

    # vr_* params have safe init defaults: alpha1=1.0, alpha2=0.0, scale=1.0
    # → math collapses to v_mixed = v (no-op). Tolerable if checkpoint omits them
    # (e.g. older trainer version). Warn but don't fail.
    def _is_benign_vr(k):
        return any(k.endswith(s) for s in ('vr_alpha1', 'vr_alpha2', 'vr_scale'))
    benign_missing   = [k for k in missing if _is_benign_vr(k)]
    critical_missing = [k for k in missing if not _is_benign_vr(k)]

    if benign_missing:
        print(f"  [WARN] {len(benign_missing)} value-residual keys missing — using no-op defaults (alpha1=1, alpha2=0, scale=1):")
        for k in benign_missing[:5]:
            print(f"    - {k}")
        if len(benign_missing) > 5:
            print(f"    ... and {len(benign_missing) - 5} more")

    if critical_missing:
        print(f"  [ERROR] {len(critical_missing)} critical keys missing (would be RANDOM INIT — refusing):")
        for k in critical_missing[:10]:
            print(f"    - {k}")
        if len(critical_missing) > 10:
            print(f"    ... and {len(critical_missing) - 10} more")
        raise RuntimeError(f"Checkpoint load incomplete: {len(critical_missing)} critical keys missing. "
                           f"Architecture detection failed — fix before inference.")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys (ignored):")
        for k in unexpected[:5]:
            print(f"    - {k}")

    target_dtype = _rockso_resolve_dtype(dtype, device)
    model = model.to(device)
    if target_dtype is not None:
        model = model.to(target_dtype)
    qk_count, norm_count = _enforce_fp32_norms(model)
    model.eval()
    print(f"  Runtime dtype: {target_dtype if target_dtype is not None else 'native/checkpoint'}")

    print(f"  Config: {cfg.get('n_layer',30)}L {cfg.get('n_head',32)}H {cfg.get('n_embd',2048)}D")
    print(f"  LatentMoE: {has_latent_moe}")
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.0f}M")
    resolved, trained, mismatch = _resolve_prefill_mode(cfg, prefixlm_prefill)
    model.rope_impl = _ROPE_IMPL
    model.prefixlm_prefill = resolved
    mode = "PrefixLM bidirectional prefill" if resolved else "causal prefill"
    src_note = "auto from checkpoint" if prefixlm_prefill is None else "FORCED by flag"
    print(f"  Checkpoint trained mask: {trained}")
    print(f"  Prefill mode: {mode} ({src_note})")
    if mismatch:
        print("  " + "!" * 64)
        print("  ! WARNING: prefill mode contradicts the checkpoint's trained mask")
        print(f"  ! ({trained}-trained weights under {'prefixlm' if resolved else 'causal'} prefill).")
        print("  ! This is the off-diagonal 2x2 cell -- valid ONLY as an ablation.")
        print("  " + "!" * 64)
    print(f"  FP32 norms: QK={qk_count}, RMS={norm_count}")
    print(f"  Ready on {device}")
    return model, cfg


def _resolve_selftest_trainer_module(checkpoint: str, requested: str) -> str:
    if requested != "auto":
        return requested
    p = str(checkpoint).lower()
    if "fixed" in p or "rotate" in p or "prod_fixed" in p:
        return "TRAINER_MAIN_dense_prefixlm_prod_fixed_fast"
    return "TRAINER_MAIN_dense_prefixlm_toy"


def selftest(checkpoint: str, tokenizer_path: str, device: str, dtype: str,
             prompt: str, atol: float = None, rope_impl: str = "auto",
             trainer_module: str = "auto"):
    """Golden parity gate against the trainer PrefixLM forward path."""
    from eval_prefixlm_ppl import load_model as load_prefixlm_model

    print("[selftest] forcing fp32 for both stacks (correctness gate, not a perf path)")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    chat_model, _ = load_checkpoint(checkpoint, device=device, dtype="float32",
                                    prefixlm_prefill=True, rope_impl=rope_impl,
                                    tokenizer_path=tokenizer_path)
    if getattr(chat_model, "is_rockso_hybrid", False):
        from hybrid_chat_adapter import selftest_model
        selftest_model(chat_model, tokenizer, prompt, atol)
        return
    trainer_module = _resolve_selftest_trainer_module(checkpoint, trainer_module)
    print(f"[selftest] trainer module: {trainer_module}")
    trainer_model, trainer, _ = load_prefixlm_model(
        checkpoint, tokenizer_path, device, False,  # bf16=False -> fp32
        trainer_module,
    )
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    T = ids.size(1)
    pos = torch.arange(T, device=device).unsqueeze(0)
    meta = torch.ones_like(ids, dtype=torch.int8)
    span_ends = torch.tensor([[T]], dtype=torch.int32, device=device)
    allow = trainer.build_attn_allow_mask(span_ends, meta, T, "prefixlm").unsqueeze(1)

    def _last(chat_logits, trainer_logits):
        a = chat_logits[:, -1].float()
        b = trainer_logits[:, -1].float()
        err = (a - b).abs().max().item()
        top1 = bool((a.argmax(-1) == b.argmax(-1)).all().item())
        return err, top1

    with torch.no_grad():
        c_chat, _ = chat_model(ids, pos, prefill_causal=True)
        c_tr = trainer_model(input_ids=ids, position_ids=pos, labels=None,
                             attn_mask=None, block_mask=None, return_logits=True)
        ctrl_err, ctrl_top1 = _last(c_chat, c_tr["logits"])

        p_chat, caches = chat_model(ids, pos, prefill_causal=False)
        p_tr = trainer_model(input_ids=ids, position_ids=pos, labels=None,
                             attn_mask=allow, block_mask=None, return_logits=True)
        test_err, test_top1 = _last(p_chat, p_tr["logits"])

    floor = atol if atol is not None else 2e-2
    gate = max(floor, 2.5 * ctrl_err)
    print(f"[selftest] control (causal/causal)    max_err={ctrl_err:.6f} top1_match={ctrl_top1}")
    print(f"[selftest] test    (prefixlm/prefixlm) max_err={test_err:.6f} top1_match={test_top1}")
    print(f"[selftest] gate: test_err <= max(floor={floor}, 2.5x control={2.5*ctrl_err:.6f}) = {gate:.6f}")
    if not ctrl_top1:
        print("[selftest][warn] control top-1 mismatch: stacks disagree even on causal -- "
              "any failure below is stack plumbing, not the prefixlm path")
    if test_err > gate or not test_top1:
        raise SystemExit(
            f"PrefixLM prefill parity FAILED: err={test_err:.6f} > gate={gate:.6f} "
            f"or top1_match={test_top1}")
    print("[selftest] prefill parity OK")

    next_id = p_chat[:, -1].argmax(dim=-1, keepdim=True)
    ext_ids = torch.cat([ids, next_id], dim=1)
    ext_pos = torch.arange(T + 1, device=device).unsqueeze(0)
    ext_meta = torch.cat([meta, torch.zeros((1, 1), dtype=torch.int8, device=device)], dim=1)
    ext_span = torch.tensor([[T + 1]], dtype=torch.int32, device=device)
    ext_allow = trainer.build_attn_allow_mask(ext_span, ext_meta, T + 1, "prefixlm").unsqueeze(1)
    with torch.no_grad():
        dec_logits, _ = chat_model(next_id, torch.tensor([[T]], device=device),
                                   kv_caches=caches)
        ext_out = trainer_model(input_ids=ext_ids, position_ids=ext_pos, labels=None,
                                attn_mask=ext_allow, block_mask=None, return_logits=True)
    dec_err, dec_top1 = _last(dec_logits, ext_out["logits"])
    print(f"[selftest] decode  (first response tok) max_err={dec_err:.6f} top1_match={dec_top1}")
    if dec_err > gate or not dec_top1:
        raise SystemExit(
            f"PrefixLM decode parity FAILED: err={dec_err:.6f} > gate={gate:.6f} "
            f"or top1_match={dec_top1}")
    print("[selftest] prefill + decode parity OK")


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def print_help(styles):
    print("\n  Commands:")
    print("    /help                 - This help")
    print("    /params               - Show current settings")
    print("    /set <param> <value>  - Change a setting")
    print("    /style <name>         - Change prompt style")
    print("    /styles               - List available styles")
    print("    /condition <cond>     - Override HRM condition tokens, e.g. synth,cot")
    print("    /condition auto       - Use the style's default condition again")
    print("    /conditions           - List HRM condition primitives and combo styles")
    print("    /style code           - Python code generation")
    print("    /style code_complete  - Python function/body completion")
    print("    /style code_task      - Description-first Python task, synth,cot")
    print("    /style hrm_synth_cot  - Unmodified question under synth,cot (math/code)")
    print("    /tools <file.json>    - Load schemas and select Direct tools mode; no execution")
    print("    /paste                - Multiline question/code; finish with /end")
    print("    /seed <n>             - Set random seed (0 = random)")
    print("    /clear                - Clear chat history")
    print("    /debug                - Show the raw prompt being sent")
    print("    /warn                 - Show PrefixLM multi-turn caveat")
    print("    /quit                 - Exit")
    print()
    print("  Settable params: temp, min_p, rep, max_tokens, suppress_special, rep_detect")
    print("  Example: /set temp 0.4")
    print("           /set rep_detect off")
    print()


def main():
    parser = argparse.ArgumentParser(description="Rockso Chat v5 PrefixLM prefill")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Original/hybrid trainer checkpoint, native hybrid export, or HF-style export',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'bfloat16', 'float32', 'fp32', 'fp16', 'float16', 'native'], help='Model runtime dtype. Use float32/fp32 for parity/official eval anchors; bf16 for fast chat.')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to the matching tokenizer directory')
    parser.add_argument('--style', type=str, default='qa', choices=list(PROMPT_STYLES.keys()))
    parser.add_argument('--condition', type=str, default=None,
                        help='Override HRM condition tokens, e.g. direct, synth,cot, noisy,direct')
    parser.add_argument('--system', type=str, default=None)
    parser.add_argument('--tools-json', type=str, default=None,
                        help='Tool schema JSON file; selects tools style. Calls are displayed, never executed.')
    parser.add_argument('--temp', type=float, default=0.0)
    parser.add_argument('--min_p', type=float, default=0.05)
    parser.add_argument('--rep', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0 = random each time)')
    parser.add_argument('--no_suppress_special', action='store_true',
                        help='Do not mask special/control tokens during generation.')
    parser.add_argument('--no_rep_detect', '--no-rep-detect', dest='rep_detect',
                        action='store_false', default=True,
                        help='Disable 4-gram repetition early stop')
    parser.add_argument('--causal-prefill', action='store_true',
                        help='2x2 ablation: force causal prefill regardless of checkpoint')
    parser.add_argument('--prefixlm-prefill', action='store_true',
                        help='2x2 ablation: force PrefixLM prefill regardless of checkpoint')
    parser.add_argument('--cuda-graph-decode', action='store_true',
                        help='Replay native single-row V2 decode with CUDA graphs after a bitwise startup check')
    parser.add_argument('--rope-impl', choices=['auto', 'legacy', 'interleaved', 'rotate_half', 'rotate-half', 'rotatehalf', 'hrm', 'olmo'],
                        default='auto', help='RoPE convention for raw checkpoints; auto detects fixed/rotate-half runs by metadata/path')
    parser.add_argument('--selftest', action='store_true',
                        help='Run PrefixLM chat-vs-trainer parity gate and exit')
    parser.add_argument('--selftest-prompt', default='<|im_start|><|object_ref_start|>What is 2+2?<|im_end|>')
    parser.add_argument('--selftest-atol', type=float, default=None)
    parser.add_argument('--selftest-trainer-module', default='auto',
                        help='Trainer module for selftest parity; auto chooses fixed trainer for fixed/marked runs')
    args = parser.parse_args()

    if args.causal_prefill and args.prefixlm_prefill:
        raise SystemExit("--causal-prefill and --prefixlm-prefill are mutually exclusive")
    forced_prefill = True if args.prefixlm_prefill else (False if args.causal_prefill else None)

    if args.selftest:
        selftest(args.checkpoint, args.tokenizer, args.device, args.dtype,
                 args.selftest_prompt, args.selftest_atol,
                 rope_impl=args.rope_impl,
                 trainer_module=args.selftest_trainer_module)
        return

    print("""
╔══════════════════════════════════════════════════════╗
║        ROCKSO CHAT v5 — PrefixLM KV Edition          ║
║     Full v5 chat shell, bidirectional prefill        ║
║            /help for commands, /quit to exit         ║
╚══════════════════════════════════════════════════════╝
""")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    suppressed_control_ids = suppressed_special_token_ids(tokenizer)
    model, cfg = load_checkpoint(
        args.checkpoint, args.device, args.dtype,
        prefixlm_prefill=forced_prefill,
        rope_impl=args.rope_impl,
        tokenizer_path=args.tokenizer,
    )
    if args.cuda_graph_decode:
        from hybrid_graph_decode import enable_cuda_graph_decode
        probe = build_hrm_prompt('Write a Python function that returns the factorial of n.',
                                 condition='synth,cot')
        try:
            model = enable_cuda_graph_decode(model, tokenizer.encode(probe, add_special_tokens=False))
        except (AssertionError, ValueError, TypeError, RuntimeError) as exc:
            print(f'[decode] CUDA graph unavailable ({type(exc).__name__}: {exc}); using reference decoder')

    # State
    style = args.style
    system_prompt = args.system
    temperature = args.temp
    min_p = args.min_p
    rep_penalty = args.rep
    max_tokens = args.max_tokens
    suppress_special = not args.no_suppress_special
    condition_override = normalize_hrm_condition(args.condition) if args.condition else None
    seed = args.seed if args.seed != 0 else None
    rep_detect = args.rep_detect
    history = []
    tool_schemas = None
    if args.tools_json:
        from tool_prompt_formats import load_tools
        tool_schemas = load_tools(args.tools_json)
        style = 'tools'

    print(f"  Style: {style} ({PROMPT_STYLES[style]['desc']})")
    if condition_override:
        print(f"  Condition override: {condition_override}")
    print(f"  Active condition: {condition_override or style_condition(style) or 'none'}")
    print(f"  temp={temperature} min_p={min_p} rep={rep_penalty} max_tokens={max_tokens} "
          f"suppress_special={suppress_special} rep_detect={rep_detect}")
    if suppress_special:
        print(
            f"  Suppressed control tokens: {len(suppressed_control_ids)} "
            "(ordinary code syntax preserved)"
        )
    if seed:
        print(f"  Seed: {seed} (deterministic)")
    print("  HRM/code/tools styles are single-turn: each question starts fresh.")
    print("  Code: /style code_task | Math: /style hrm_synth_cot, then /set max_tokens 1280")
    print("  Tools: /tools /path/to/tools.json | Multiline input: /paste, then /end")
    print()

    while True:
        try:
            user_input = input("\033[92mYou:\033[0m ").strip()
            if not user_input:
                continue
            if user_input == '/paste':
                print('  Paste question/code; finish with /end on its own line.')
                lines = []
                while True:
                    line = input()
                    if line.strip() == '/end':
                        break
                    lines.append(line)
                user_input = '\n'.join(lines).strip()
                if not user_input:
                    continue

            # Commands
            if user_input.startswith('/'):
                parts = user_input[1:].split(maxsplit=2)
                cmd = parts[0].lower()

                if cmd in ('quit', 'exit', 'q'):
                    print("Goodbye!")
                    break
                elif cmd == 'help':
                    print_help(PROMPT_STYLES)
                elif cmd == 'styles':
                    print("\n  Available styles:")
                    for name, cfg_s in PROMPT_STYLES.items():
                        marker = " ◀" if name == style else ""
                        print(f"    {name:12s} — {cfg_s['desc']}{marker}")
                    print()
                elif cmd == 'tools' and len(parts) >= 2:
                    from tool_prompt_formats import load_tools
                    path = user_input.split(maxsplit=1)[1].strip().strip('"').strip("'")
                    try:
                        loaded_tools = load_tools(path)
                    except (OSError, ValueError) as exc:
                        print(f'  Cannot load tools: {exc}')
                        continue
                    tool_schemas = loaded_tools
                    style, condition_override, history = 'tools', None, []
                    print(f'  Loaded {len(tool_schemas)} schemas. Style: tools; condition: direct; calls are not executed.')
                elif cmd == 'conditions':
                    print("\n  HRM condition primitives:")
                    print("    " + ", ".join(HRM_CONDITION_PARTS))
                    print("\n  Official/common:")
                    print("    direct      -> <|object_ref_start|>")
                    print("    cot         -> <|object_ref_end|>")
                    print("    synth,cot   -> <|quad_end|><|object_ref_end|>  (HRM GSM8K/MATH)")
                    print("    synth,direct-> <|quad_end|><|object_ref_start|>")
                    print("    noisy,*     -> <|quad_start|> + the following condition token(s)")
                    print("\n  Generated style names:")
                    by_len = {}
                    for name, cfg_s in PROMPT_STYLES.items():
                        if name.startswith("hrm_cond_"):
                            cond = cfg_s["condition"]
                            by_len.setdefault(len(parse_hrm_condition(cond)), []).append(name)
                    for n in sorted(by_len):
                        print(f"    {n}-token: " + ", ".join(sorted(by_len[n])))
                    print()
                elif cmd == 'style' and len(parts) >= 2:
                    new_style = parts[1].lower()
                    if new_style in PROMPT_STYLES:
                        style = new_style
                        history = []
                        print(f"  Style: {style} ({PROMPT_STYLES[style]['desc']})")
                        print(f"  Active condition: {condition_override or style_condition(style) or 'none'}")
                    else:
                        print(f"  Unknown style. Use /styles to see options.")
                elif cmd == 'params':
                    print(f"\n  style={style} temp={temperature} min_p={min_p} "
                          f"rep={rep_penalty} max_tokens={max_tokens} "
                          f"condition={condition_override or style_condition(style) or 'none'} "
                          f"suppress_special={suppress_special} rep_detect={rep_detect} "
                          f"seed={seed or 'random'}\n")
                elif cmd in ('condition', 'cond'):
                    if len(parts) < 2:
                        print(f"  Condition override: {condition_override or 'style-default'}")
                        continue
                    raw = "".join(parts[1:]).replace(" ", "").lower()
                    if raw in ("auto", "default", "off", "clear", "none"):
                        condition_override = None
                        history = []
                        print("  Condition override cleared; using style default.")
                    else:
                        try:
                            condition_override = normalize_hrm_condition(raw)
                        except ValueError as e:
                            print(f"  {e}")
                            continue
                        history = []
                        print(f"  Condition override: {condition_override}")
                elif cmd == 'set' and len(parts) >= 3:
                    p, v = parts[1].lower(), parts[2]
                    try:
                        if p == 'temp': temperature = float(v)
                        elif p == 'min_p': min_p = float(v)
                        elif p == 'rep': rep_penalty = float(v)
                        elif p == 'max_tokens': max_tokens = int(v)
                        elif p == 'suppress_special':
                            suppress_special = v.lower() in ('1', 'true', 'yes', 'on')
                        elif p in ('rep_detect', 'repetition_detect', 'repetition_detection'):
                            rep_detect = v.lower() in ('1', 'true', 'yes', 'on')
                        else: print(f"  Unknown param: {p}"); continue
                        print(f"  {p} = {v}")
                    except ValueError:
                        print(f"  Invalid value: {v}")
                elif cmd == 'seed' and len(parts) >= 2:
                    s = int(parts[1])
                    seed = s if s != 0 else None
                    print(f"  Seed: {seed or 'random'}")
                elif cmd == 'clear':
                    had_history = bool(history)
                    history = []
                    if had_history:
                        print("  History cleared.")
                    elif style not in ('chat', 'instruct'):
                        print(f"  History already empty; {style} is single-turn/stateless.")
                    else:
                        print("  History already empty.")
                elif cmd == 'debug':
                    prompt = build_prompt(user_input.replace('/debug ', '').strip() or "test",
                                         style, system_prompt, history, condition_override, tool_schemas)
                    print(f"\n--- Style: {style} ---")
                    print(f"--- Condition: {condition_override or style_condition(style) or 'none'} ---")
                    print(repr(prompt))
                    print(f"--- Stops: {PROMPT_STYLES[style]['stops']} ---\n")
                elif cmd == 'warn':
                    print("  Multi-turn is approximate: full history is re-prefilled each turn.")
                    print("  Exact HRM-Text fidelity is single-turn prompt -> response.")
                else:
                    print("  Unknown command. /help for options.")
                continue

            # Build prompt and generate
            prompt = build_prompt(user_input, style, system_prompt, history, condition_override, tool_schemas)
            stop_strings = PROMPT_STYLES[style]["stops"]
            stream_response = style != "casual"

            response, stats = generate(
                model, tokenizer, prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                min_p=min_p,
                repetition_penalty=rep_penalty,
                stop_strings=stop_strings,
                suppress_special=suppress_special,
                device=args.device,
                stream=stream_response,
                rep_detect=rep_detect,
                seed=seed,
                allow_tool_calls=(style == 'tools'),
            )
            response = clean_response_for_style(response, style)
            if not stream_response:
                print(f"\033[94mRockso:\033[0m {response}")

            # Stats line
            print(f"  \033[90m[{stats['tokens']} tok, {stats['tok_per_sec']:.1f} tok/s, "
                  f"{stats['stop_reason']}]\033[0m")

            # Save to history for raw multi-turn modes. HRM casual stays stateless;
            # early checkpoints tend to amplify their own previous phrasing.
            if style in ('chat', 'instruct'):
                history.append({'user': user_input, 'assistant': response})

        except KeyboardInterrupt:
            print("\n  Use /quit to exit.")
        except ValueError as exc:
            print(f"  {exc}")
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
