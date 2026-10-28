"""Native hybrid inference with KV-cached SDPA for the existing v5 chat shell.

The cached Block.forward preserves all learned mechanisms and explicit mixed
precision. Training sources stay unchanged.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode
from rockso_hybrid import ModelConfig, RocksoHybrid
from rockso_hybrid.data import tokenizer_receipt
from rockso_hybrid.masking import build_masks
from rockso_hybrid.runtime import sha256

ROOT = Path(__file__).resolve().parent
# Cached forward delegates the MLP to the native block, including SwiGLU.
# Pinned to the implementation checked by check_inference_cpu.py and final GPU parity.
CACHED_MODEL_SHA256 = "612fc4182f128bac38066124d992ca636bfaf8ac56e841eb7546f336e361bbf9"


@dataclass(frozen=True)
class HybridKVCache:
    layers: tuple
    length: int
    prefix_length: int


@dataclass(frozen=True)
class HybridStaticKVCache:
    layers: tuple
    prefix_lengths: torch.Tensor
    capacity: int


class _StaticDecodeReductions(TorchFunctionMode):
    """Keep one-token reductions consistent with the per-row decoder.

    A normal BF16 batched GEMM selects different reductions on the 5090 and can
    flip greedy choices. RMS mean reductions can also change at BF16 rounding
    boundaries. Attention and pointwise work remain batched, while Linear and
    normalization reductions use the reference's row shape. This scoped mode
    also covers the native MLP and MUDD's functional linears without copying
    their implementation or modifying modules, global F.linear, or training.
    """
    _reductions = {torch.mean, torch.Tensor.mean, torch.sum, torch.Tensor.sum,
                   torch.norm, torch.Tensor.norm, torch.linalg.vector_norm,
                   F.rms_norm, F.normalize}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.linear:
            x = args[0] if args else kwargs['input']
            if x.ndim == 3 and x.shape[1] == 1 and x.shape[0] > 1:
                weight = args[1] if len(args) > 1 else kwargs['weight']
                bias = args[2] if len(args) > 2 else kwargs.get('bias')
                return torch.cat([F.linear(row[None], weight, bias) for row in x], dim=0)
        if func in self._reductions and args:
            x = args[0]
            if isinstance(x, torch.Tensor) and x.ndim >= 3 and x.shape[0] > 1:
                # All reductions in the pinned inference model are over feature
                # dimensions. Preserve each row's complete head/sequence shape.
                return torch.cat([func(row[None], *args[1:], **kwargs) for row in x], dim=0)
        return func(*args, **kwargs)


def _cached_block(block, x, positions, mask, first_v, value_embed, taps, cache,
                  native_gqa, static_write=None):
    c = block.cfg
    b, t, _ = x.shape
    if block.mudd is None:
        h, residual = block.norm1(x), x
    else:
        attn_input, residual = block.mudd(x, taps)
        h = block.norm1(attn_input)
    q = block.q_proj(h).view(b, t, c.n_head, c.head_dim).transpose(1, 2)
    k = block.k_proj(h).view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
    v = block.v_proj(h).view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
    raw_v = v
    q, k = block.q_norm(q), block.k_norm(k)
    if not (c.global_nope and not c.local(block.index)):
        q, k = block.rope(q, positions), block.rope(k, positions)
    group = c.n_head // c.n_kv_head
    expanded = not native_gqa and group > 1
    if expanded:
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    if c.value_residual and first_v is not None:
        # Preserve Rockso's reduction order for the learned VR scalars.
        # Mixing before replication has identical forward values but different
        # BF16 scalar gradients. Native GQA is an explicit separate ablation.
        first = first_v.repeat_interleave(group, 1) if expanded else first_v
        v = block.vr_scale * (block.vr_alpha1 * v + block.vr_alpha2 * first) * torch.rsqrt(
            block.vr_alpha1.square() + block.vr_alpha2.square() + 1e-8)
    if block.ve_gate is not None:
        gate = 2 * torch.sigmoid(F.linear(h[..., :c.ve_gate_channels].float(), block.ve_gate.weight))
        addition = value_embed.view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
        gate = gate.transpose(1, 2)[..., None]
        if expanded:
            addition, gate = addition.repeat_interleave(group, 1), gate.repeat_interleave(group, 1)
        v = (v.float() + gate * addition.float()).to(q.dtype)
    v = v.to(q.dtype)
    # Cache post-RoPE K and post-VR/embedding V. XSA below uses only
    # the current token's effective value direction, never cached history.
    if static_write is None:
        keys = torch.cat((cache[0], k), dim=2) if cache is not None else k
        values = torch.cat((cache[1], v), dim=2) if cache is not None else v
    else:
        # Fixed storage, one distinct write position per active row. The caller
        # slices to the longest live context and masks shorter rows explicitly.
        rows, write_positions = static_write
        keys, values = cache
        keys[rows, :, write_positions, :] = k[rows, :, 0, :]
        values[rows, :, write_positions, :] = v[rows, :, 0, :]
    out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask,
        is_causal=False, scale=c.attention_scale, enable_gqa=native_gqa)
    if block.xsa_alpha is not None:
        # Signed learned value-direction suppression, not removal of the self key.
        self_v = v.repeat_interleave(group, 1) if native_gqa and group > 1 else v
        vn = F.normalize(self_v.float(), dim=-1, eps=1e-4)
        yf = out.float()
        projection = (yf * vn).sum(-1, keepdim=True)
        out = (yf - block.xsa_alpha.tanh()[None, :, None, None] * projection * vn).to(out.dtype)
    if block.attention_gate is not None:
        gate = 2 * torch.sigmoid(F.linear(h[..., :c.attention_gate_channels].float(), block.attention_gate.weight))
        out = (out.float() * gate.transpose(1, 2)[..., None]).to(out.dtype)
    out = out.transpose(1, 2).contiguous().view(b, t, c.n_embd)
    x = block.attention_residual(residual, block.out_proj(out))
    x = block.mlp_residual(x, block.mlp(block.norm2(x)))
    return x, raw_v, (keys, values)


class HybridChatModel(nn.Module):
    is_rockso_hybrid = True

    def __init__(self, native, prefixlm_prefill):
        super().__init__()
        self.native = native
        self.config = native.config
        self.block_size = native.config.max_seq_len
        self.n_layer = native.config.n_layer
        self.prefixlm_prefill = prefixlm_prefill
        self.rope_impl = "rotate_half"
        self._retained = {j for site in self.config.mudd_layers
                          for j in self.config.taps(site)[:-1]}

    @torch.no_grad()
    def forward(self, input_ids, position_ids=None, kv_caches=None,
                prefill_causal=None):
        batch, length = input_ids.shape
        past = kv_caches.length if kv_caches is not None else 0
        total = past + length
        if not length or total > self.block_size:
            raise ValueError("Hybrid chat input must be nonempty and within its context")
        if kv_caches is None:
            causal = not self.prefixlm_prefill if prefill_causal is None else prefill_causal
            prefix_length = 0 if causal else length
        else:
            prefix_length = kv_caches.prefix_length
        if position_ids is None:
            position_ids = torch.arange(past, total, device=input_ids.device)[None].expand(batch, -1)
        # New queries use their absolute sequence indices. Prefix keys stay
        # visible even outside a configured local window.
        q = torch.arange(past, total, device=input_ids.device)[:, None]
        k = torch.arange(total, device=input_ids.device)[None, :]
        masks = {}
        for window in dict.fromkeys((0, self.config.local_window)):
            allowed = k <= q
            if window:
                allowed = allowed & (k >= q - window)
            masks[window] = (allowed | (k < prefix_length))[None, None].expand(batch, 1, -1, -1)
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            x = self.native.embed(input_ids)
            ve = self.native.value_embed(input_ids) if self.native.value_embed is not None else None
            history, first_v, new_caches = {}, None, []
            for i, block in enumerate(self.native.blocks):
                if i in self._retained:
                    history[i] = x
                taps = tuple(history[j] if j != i else x for j in self.config.taps(i)) if block.mudd is not None else ()
                window = self.config.local_window if self.config.local(i) else 0
                previous = kv_caches.layers[i] if kv_caches is not None else None
                x, raw_v, cached = _cached_block(
                    block, x, position_ids, masks[window], first_v,
                    ve if block.ve_gate is not None else None, taps, previous,
                    self.native.native_gqa)
                new_caches.append(cached)
                if i == 0 and self.config.value_residual:
                    first_v = raw_v
            hidden = self.native.norm(x)
            logits = F.linear(hidden, self.native.lm_head.weight).float()
            if self.config.sigmoid_logit_cap:
                logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        return logits, HybridKVCache(tuple(new_caches), total, prefix_length)


    @torch.no_grad()
    def pack_static_caches(self, row_caches, prompt_lengths, capacity):
        """Pack unpadded, independent prefills without changing their positions."""
        if not row_caches or len(row_caches) != len(prompt_lengths):
            raise ValueError("Static cache requires one length per nonempty row cache")
        if not 0 < capacity <= self.block_size:
            raise ValueError("Static cache capacity exceeds the hybrid context")
        for cache, length in zip(row_caches, prompt_lengths):
            if not isinstance(cache, HybridKVCache):
                raise TypeError("Expected native HybridKVCache prefills")
            if cache.length != length or not 0 < length <= capacity:
                raise ValueError("Static prefill length does not match its cache/capacity")
            if len(cache.layers) != self.n_layer or not 0 <= cache.prefix_length <= length:
                raise ValueError("Invalid hybrid prefill cache metadata")
        batch = len(row_caches)
        packed = []
        for layer_index in range(self.n_layer):
            k0, v0 = row_caches[0].layers[layer_index]
            keys = k0.new_zeros((batch, k0.shape[1], capacity, k0.shape[3]))
            values = v0.new_zeros((batch, v0.shape[1], capacity, v0.shape[3]))
            for row, (cache, length) in enumerate(zip(row_caches, prompt_lengths)):
                for src, dst in zip(cache.layers[layer_index], (keys, values)):
                    expected = (1, dst.shape[1], length, dst.shape[3])
                    if src.shape != expected or src.device != dst.device or src.dtype != dst.dtype:
                        raise ValueError("Incompatible shapes, devices or dtypes in hybrid prefills")
                    dst[row, :, :length].copy_(src[0])
            packed.append((keys, values))
        prefixes = torch.tensor([c.prefix_length for c in row_caches],
                                dtype=torch.long, device=packed[0][0].device)
        return HybridStaticKVCache(tuple(packed), prefixes, int(capacity))

    @torch.no_grad()
    def static_decode(self, token_ids, cache, cache_lengths, active_mask=None):
        """One batched token through the same blocks as the per-row decoder.

        cache_lengths are the next write positions; the generation loop advances
        them after this call. Finished rows do not write or advance their caches.
        """
        if not isinstance(cache, HybridStaticKVCache):
            raise TypeError("Hybrid static decode requires HybridStaticKVCache")
        batch = cache.prefix_lengths.numel()
        if token_ids.shape != (batch,) or cache_lengths.shape != (batch,):
            raise ValueError("Static decode expects one token and cache length per row")
        if cache_lengths.dtype != torch.long or cache_lengths.device != token_ids.device:
            raise ValueError("Static lengths must be int64 on the token device")
        if active_mask is None:
            active_mask = torch.ones_like(cache_lengths, dtype=torch.bool)
        if active_mask.shape != (batch,) or active_mask.dtype != torch.bool:
            raise ValueError("Static active mask must be one boolean per row")
        # One host transfer per token, not one per layer. An inactive row can
        # already fill the context; give its ignored computation a valid index.
        bounds = torch.stack((cache_lengths.min(), cache_lengths.max(),
                              (active_mask & (cache_lengths >= cache.capacity)).any(),
                              (cache_lengths < cache.prefix_lengths).any())).tolist()
        if bounds[0] < 0 or bounds[1] > cache.capacity or bounds[2] or bounds[3]:
            raise ValueError("Static decode cache position is outside its valid context")
        positions = cache_lengths.clamp(max=cache.capacity - 1)
        max_length = min(int(bounds[1]) + 1, cache.capacity)
        rows = torch.arange(batch, device=token_ids.device)[active_mask]
        write = (rows, positions[rows])
        key_positions = torch.arange(max_length, device=token_ids.device)[None, :]
        query_positions = positions[:, None]
        masks = {}
        for window in dict.fromkeys((0, self.config.local_window)):
            valid = key_positions <= query_positions
            if window:
                valid = valid & ((key_positions >= query_positions - window) |
                                 (key_positions < cache.prefix_lengths[:, None]))
            masks[window] = valid[:, None, None, :]
        with torch.autocast(device_type=token_ids.device.type, enabled=False), _StaticDecodeReductions():
            ids = token_ids[:, None]
            x = self.native.embed(ids)
            ve = self.native.value_embed(ids) if self.native.value_embed is not None else None
            history, first_v = {}, None
            for i, block in enumerate(self.native.blocks):
                if i in self._retained:
                    history[i] = x
                taps = tuple(history[j] if j != i else x for j in self.config.taps(i)) if block.mudd is not None else ()
                window = self.config.local_window if self.config.local(i) else 0
                storage = tuple(t[:, :, :max_length] for t in cache.layers[i])
                x, raw_v, _ = _cached_block(
                    block, x, positions[:, None], masks[window], first_v,
                    ve if block.ve_gate is not None else None, taps, storage,
                    self.native.native_gqa, static_write=write)
                if i == 0 and self.config.value_residual:
                    first_v = raw_v
            logits = F.linear(self.native.norm(x), self.native.lm_head.weight).float()
            if self.config.sigmoid_logit_cap:
                logits = 23 * torch.sigmoid((logits + 5) / 7.5)
        return logits[:, 0]


def load_chat_model(state, *, device, target_dtype, prefixlm_prefill=None,
                    rope_impl="auto", tokenizer_path=None):
    config = ModelConfig.from_dict(state["config"])
    if state["schema"] == "rockso_hybrid_checkpoint_1":
        contract = state["contract"]
        sources, token_info = contract["source_sha256"], contract["tokenizer"]
        trained = contract["arguments"]["attention_mode"]
        native_gqa = contract["arguments"].get("native_gqa", False)
    else:
        sources, token_info = state["source_sha256"], state["tokenizer"]
        trained = state["attention_mode"]
        native_gqa = False
    for name in ("rockso_hybrid/model.py", "rockso_hybrid/config.py", "rockso_hybrid/masking.py", "rockso_hybrid/residual.py"):
        actual = sha256(ROOT / name)
        if actual != sources.get(name):
            raise ValueError(f"Hybrid checkpoint implementation differs: {name}")
        if name == "rockso_hybrid/model.py" and actual != CACHED_MODEL_SHA256:
            raise ValueError("Update the cached chat adapter for this native model revision")
    if rope_impl not in ("auto", "rotate_half", "rotate-half", "rotatehalf", "hrm", "olmo"):
        raise ValueError("The hybrid uses rotate-half RoPE; use --rope-impl auto or rotate_half")
    _, actual_tokenizer = tokenizer_receipt(tokenizer_path or token_info["path"], config.vocab_size)
    # Relocation is allowed; tokenizer contents and numerical metadata must match.
    actual_identity = {k: v for k, v in actual_tokenizer.items() if k != "path"}
    saved_identity = {k: v for k, v in token_info.items() if k != "path"}
    if actual_identity != saved_identity:
        raise ValueError("Chat tokenizer differs from the hybrid checkpoint")
    dtype = target_dtype or state["model"]["embed.weight"].dtype
    native = RocksoHybrid(config).to(device=device, dtype=dtype).preserve_precision()
    native.load_state_dict(state["model"], strict=True)
    native.configure_runtime(backend="sdpa", checkpointing=False, native_gqa=native_gqa)
    native.eval()
    prefill = trained == "prefixlm" if prefixlm_prefill is None else bool(prefixlm_prefill)
    model = HybridChatModel(native, prefill).eval()
    model.checkpoint_step = int(state.get("step", 0))
    model.checkpoint_tokens_seen = int(state.get("tokens_seen", 0))
    model.inference_provenance = {
        "schema": state["schema"], "config": config.to_dict(),
        "tokenizer": actual_tokenizer, "dtype": str(dtype),
        "fp8": False, "autocast": False,
        "static_decode_linear_policy": "independent_rows_reference_reductions",
        "static_decode_reduction_policy": "independent_rows_linear_and_norm_reductions",
        "source_sha256": {n: sources[n] for n in (
            "rockso_hybrid/model.py", "rockso_hybrid/config.py",
            "rockso_hybrid/masking.py", "rockso_hybrid/residual.py")},
        "cached_adapter_sha256": sha256(Path(__file__)),
    }
    cfg = config.to_dict()
    cfg.update(block_size=config.max_seq_len, num_kv_heads=config.n_kv_head,
               attn_mask_mode=trained, prefixlm_impl="sdpa_bool")
    print(f"  Step: {state.get('step', '?')}, Tokens: {state.get('tokens_seen', 0):,}")
    print(f"  Source kind: {state['schema']} (strict native load)")
    print(f"  Config: {config.n_layer}L {config.n_head}H {config.n_embd}D; "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"  Hybrid: attention_gate={config.attention_gate} value_embedding={config.value_embedding} "
          f"XSA={config.xsa} MUDD={list(config.mudd_layers)} value_residual={config.value_residual}")
    print(f"  Runtime dtype: {dtype}; FP32 norms/gates/routers; FP8 off")
    print(f"  RoPE: rotate_half; prefill={'PrefixLM' if prefill else 'causal'}; SDPA KV cache")
    if prefill != (trained == "prefixlm"):
        print(f"  Prefill override: checkpoint was trained with {trained}")
    print(f"  Ready on {device}")
    return model, cfg


@torch.no_grad()
def selftest_model(model, tokenizer, prompt, atol=None):
    """Compare prefill and four cached decode steps with native full forwards."""
    native = model.native
    device = native.embed.weight.device
    ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    if not ids.numel() or ids.shape[1] + 4 > model.block_size:
        raise ValueError("Selftest prompt must fit with four decode tokens")
    limit = 2e-2 if atol is None else atol
    results = []
    for causal in (True, False):
        sequence = ids
        prefix_length = 0 if causal else ids.shape[1]
        logits, cache = model(ids, prefill_causal=causal)
        for step in range(5):
            meta = torch.zeros_like(sequence)
            meta[:, :prefix_length] = 1
            ends = torch.full((sequence.shape[0], 1), sequence.shape[1], device=device, dtype=torch.int32)
            masks = build_masks(ends, meta, mode="prefixlm" if not causal else "causal",
                                backend="sdpa", local_window=model.config.local_window)
            expected = native(sequence, masks=masks)
            actual_last, expected_last = logits[:, -1], expected[:, -1]
            error = float((actual_last - expected_last).abs().max())
            top1 = bool((actual_last.argmax(-1) == expected_last.argmax(-1)).all())
            if error > limit or not top1:
                raise AssertionError(f"Hybrid cached parity failed: causal={causal} step={step} "
                                     f"max_error={error} top1_match={top1}")
            results.append({"causal":causal,"decode_step":step,"max_error":error,"top1_match":top1})
            if step < 4:
                token = expected_last.argmax(-1, keepdim=True)
                sequence = torch.cat((sequence, token), dim=1)
                logits, cache = model(token, kv_caches=cache)
    print(f"[selftest] hybrid causal + PrefixLM prefill and four KV decode steps OK; "
          f"max_error={max(r['max_error'] for r in results):.6g}; all top-1 tokens match")
    return results
