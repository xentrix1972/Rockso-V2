import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .config import ModelConfig
from .residual import residual_mix


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Rotary(nn.Module):
    def __init__(self, dim, base, precision):
        super().__init__()
        self.dim, self.precision = dim, precision
        self.register_buffer("inv_freq", 1.0 / base ** (torch.arange(0, dim, 2).float() / dim), persistent=False)

    def forward(self, x, positions):
        # Rockso rotate_half convention. Partial RoPE keeps the remaining lanes unchanged.
        # The production trainer runs this einsum under BF16 autocast, including
        # BF16-rounded frequencies and BF16 cos/sin. Preserve that in the bridge.
        if self.precision == "float32":
            # Keep positions, frequencies, phase, trig and rotation in FP32 even
            # under an outer autocast context; round only the final Q/K result.
            with torch.autocast(device_type=x.device.type, enabled=False):
                freqs = positions.float()[..., None] * self.inv_freq.float()
                emb = torch.cat((freqs, freqs), -1).unsqueeze(1)
                rot, rest = x[..., :self.dim].float(), x[..., self.dim:]
                half = self.dim // 2
                rotated = torch.cat((-rot[..., half:], rot[..., :half]), -1)
                out = (rot * emb.cos() + rotated * emb.sin()).to(x.dtype)
        else:
            freqs = torch.einsum("bt,d->btd", positions.to(x.dtype), self.inv_freq.to(x.dtype))
            emb = torch.cat((freqs, freqs), -1).unsqueeze(1)
            rot, rest = x[..., :self.dim], x[..., self.dim:]
            half = self.dim // 2
            rotated = torch.cat((-rot[..., half:], rot[..., :half]), -1)
            out = (rot * emb.cos() + rotated * emb.sin()).to(x.dtype)
        return torch.cat((out, rest), -1) if rest.shape[-1] else out


class ReLUSquared(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)).square())


class SwiGLU(nn.Module):
    """Separate gate/up matrices preserve independent NorMuon updates."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)  # SiLU gate
        self.w2 = nn.Linear(hidden, dim, bias=False)  # output
        self.w3 = nn.Linear(dim, hidden, bias=False)  # linear up

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DepthMixer(nn.Module):
    """Two unnormalized, token-dependent mixtures of three residual depths."""
    def __init__(self, dim, rank, shared_reduce):
        super().__init__()
        self.reduce = shared_reduce
        self.expand = nn.Linear(rank, 6, bias=True)
        nn.init.zeros_(self.expand.weight)
        with torch.no_grad():
            self.expand.bias.copy_(torch.tensor([0., 0., 1., 0., 0., 1.]))

    def forward(self, x, taps):
        # Violetto normalizes the router input; use Rockso's explicit 1e-5 eps
        # instead of the dtype-dependent BF16 RMS epsilon in its HF wrapper.
        normalized = F.rms_norm(x.float(), (x.shape[-1],), eps=1e-5)
        h = F.gelu(F.linear(normalized, self.reduce.weight), approximate="none")
        weights = F.linear(h, self.expand.weight, self.expand.bias).view(*x.shape[:2], 2, 3)
        outputs = []
        for kind in range(2):
            out = weights[..., kind, 0, None] * taps[0].float()
            for j in range(1, 3):
                out = out + weights[..., kind, j, None] * taps[j].float()
            outputs.append(out.to(x.dtype))
        return outputs[0], outputs[1]


_compiled_flex = None


def flex(q, k, v, mask, scale, native_gqa):
    global _compiled_flex
    if _compiled_flex is None:
        from torch.nn.attention.flex_attention import flex_attention
        _compiled_flex = torch.compile(flex_attention, dynamic=False)
    return _compiled_flex(q, k, v, block_mask=mask, scale=scale, enable_gqa=native_gqa)


class Block(nn.Module):
    def __init__(self, cfg, index, shared_mudd_reduce=None):
        super().__init__()
        self.cfg, self.index = cfg, index
        d, hd, kv = cfg.n_embd, cfg.head_dim, cfg.n_kv_head * cfg.head_dim
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.q_norm, self.k_norm = RMSNorm(hd, 1e-6), RMSNorm(hd, 1e-6)
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv, bias=False)
        self.v_proj = nn.Linear(d, kv, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.mlp = (SwiGLU if cfg.mlp_activation == "swiglu" else ReLUSquared)(d, cfg.mlp_hidden)
        self.rope = Rotary(cfg.rope_dim, cfg.rope_base, cfg.rope_math)
        if cfg.value_residual:
            self.vr_alpha1 = nn.Parameter(torch.ones(1))
            self.vr_alpha2 = nn.Parameter(torch.zeros(1))
            self.vr_scale = nn.Parameter(torch.ones(1))
        self.attention_gate = nn.Linear(cfg.attention_gate_channels, cfg.n_head, bias=False) if cfg.attention_gate else None
        if self.attention_gate is not None:
            nn.init.zeros_(self.attention_gate.weight)
        self.ve_gate = nn.Linear(cfg.ve_gate_channels, cfg.n_kv_head, bias=False) if cfg.ve_layer(index) else None
        if self.ve_gate is not None:
            nn.init.zeros_(self.ve_gate.weight)
        self.xsa_alpha = nn.Parameter(torch.zeros(cfg.n_head)) if cfg.xsa else None
        self.mudd = DepthMixer(d, cfg.mudd_rank, shared_mudd_reduce) if index in cfg.mudd_layers else None
        if cfg.residual_multipliers:
            self.resid_lambda_attn = nn.Parameter(torch.ones(()))
            self.post_lambda_attn = nn.Parameter(torch.ones(()))
            self.resid_lambda_mlp = nn.Parameter(torch.ones(()))
            self.post_lambda_mlp = nn.Parameter(torch.ones(()))

    def attention_residual(self, residual, branch):
        if self.cfg.residual_multipliers:
            return residual_mix(residual, branch, self.resid_lambda_attn, self.post_lambda_attn)
        return residual + branch

    def mlp_residual(self, residual, branch):
        if self.cfg.residual_multipliers:
            return residual_mix(residual, branch, self.resid_lambda_mlp, self.post_lambda_mlp)
        return residual + branch

    def forward(self, x, positions, mask, first_v, value_embed, taps, backend, native_gqa):
        c = self.cfg
        b, t, _ = x.shape
        if self.mudd is None:
            h, residual = self.norm1(x), x
        else:
            attn_input, residual = self.mudd(x, taps)
            h = self.norm1(attn_input)
        q = self.q_proj(h).view(b, t, c.n_head, c.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
        raw_v = v
        q, k = self.q_norm(q), self.k_norm(k)
        if not (c.global_nope and not c.local(self.index)):
            q, k = self.rope(q, positions), self.rope(k, positions)
        group = c.n_head // c.n_kv_head
        expanded = not native_gqa and group > 1
        if expanded:
            k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        if c.value_residual and first_v is not None:
            # Preserve Rockso's reduction order for the learned VR scalars.
            # Mixing before replication has identical forward values but different
            # BF16 scalar gradients. Native GQA is an explicit separate ablation.
            first = first_v.repeat_interleave(group, 1) if expanded else first_v
            v = self.vr_scale * (self.vr_alpha1 * v + self.vr_alpha2 * first) * torch.rsqrt(
                self.vr_alpha1.square() + self.vr_alpha2.square() + 1e-8)
        if self.ve_gate is not None:
            gate = 2 * torch.sigmoid(F.linear(h[..., :c.ve_gate_channels].float(), self.ve_gate.weight))
            addition = value_embed.view(b, t, c.n_kv_head, c.head_dim).transpose(1, 2)
            gate = gate.transpose(1, 2)[..., None]
            if expanded:
                addition, gate = addition.repeat_interleave(group, 1), gate.repeat_interleave(group, 1)
            v = (v.float() + gate * addition.float()).to(q.dtype)
        v = v.to(q.dtype)
        if backend == "flex":
            out = flex(q, k, v, mask, c.attention_scale, native_gqa)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                is_causal=mask is None, scale=c.attention_scale, enable_gqa=native_gqa)
        if self.xsa_alpha is not None:
            # Signed learned value-direction suppression, not removal of the self key.
            self_v = v.repeat_interleave(group, 1) if native_gqa and group > 1 else v
            vn = F.normalize(self_v.float(), dim=-1, eps=1e-4)
            yf = out.float()
            projection = (yf * vn).sum(-1, keepdim=True)
            out = (yf - self.xsa_alpha.tanh()[None, :, None, None] * projection * vn).to(out.dtype)
        if self.attention_gate is not None:
            gate = 2 * torch.sigmoid(F.linear(h[..., :c.attention_gate_channels].float(), self.attention_gate.weight))
            out = (out.float() * gate.transpose(1, 2)[..., None]).to(out.dtype)
        out = out.transpose(1, 2).contiguous().view(b, t, c.n_embd)
        x = self.attention_residual(residual, self.out_proj(out))
        x = self.mlp_residual(x, self.mlp(self.norm2(x)))
        return x, raw_v


class RocksoHybrid(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        shared_reduce = nn.Linear(config.n_embd, config.mudd_rank, bias=False) if config.mudd_layers else None
        self.blocks = nn.ModuleList(Block(config, i, shared_reduce) for i in range(config.n_layer))
        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.value_embed = nn.Embedding(config.vocab_size, config.n_kv_head * config.head_dim) if config.value_embedding else None
        if self.value_embed is not None:
            nn.init.zeros_(self.value_embed.weight)
        self.checkpointing = False
        self.keep_last = 0
        self.backend = "sdpa"
        self.native_gqa = False

    def reset_training_parameters(self):
        # Explicit train-from-scratch initialization; HF loading stubs are not initializers.
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(name.endswith(s) for s in
                    ("q_proj", "k_proj", "v_proj", "out_proj", "mlp.w1", "mlp.w2", "mlp.w3")):
                std = 0.02 / math.sqrt(2 * self.config.n_layer) if name.endswith(("out_proj", "mlp.w2")) else 0.02
                nn.init.normal_(module.weight, std=std)
        nn.init.normal_(self.embed.weight, std=0.02)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def preserve_precision(self):
        # Call after .to(dtype). RoPE frequencies must not be rounded to BF16.
        for name, param in self.named_parameters():
            if any(s in name for s in ("norm", "attention_gate", "ve_gate", "xsa_alpha", "mudd", "_lambda_")):
                param.data = param.data.float()
        for block in self.blocks:
            d = self.config.rope_dim
            block.rope.inv_freq = 1.0 / self.config.rope_base ** (
                torch.arange(0, d, 2, device=self.embed.weight.device).float() / d)
        return self

    def configure_runtime(self, *, backend="sdpa", checkpointing=False, keep_last=0,
                          compile_blocks=False, native_gqa=False):
        if keep_last < 0 or keep_last > 4:
            raise ValueError("The established memory budget allows at most four uncheckpointed tail layers")
        if backend not in ("sdpa", "flex"):
            raise ValueError("Unknown attention backend")
        self.backend, self.checkpointing = backend, checkpointing
        self.keep_last, self.native_gqa = keep_last, native_gqa
        if compile_blocks:
            for block in self.blocks:
                block.forward = torch.compile(block.forward, dynamic=False)
        return self

    def forward(self, input_ids, position_ids=None, masks=None, *, return_hidden=False):
        c = self.config
        if input_ids.shape[1] > c.max_seq_len:
            raise ValueError("Sequence exceeds configured context")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)[None].expand_as(input_ids)
        if c.local_window and masks is None:
            raise ValueError("Local/global attention requires explicitly constructed masks")
        x = self.embed(input_ids)
        ve = self.value_embed(input_ids) if self.value_embed is not None else None
        history, first_v = {}, None
        retained = {j for site in c.mudd_layers for j in c.taps(site)[:-1]}
        for i, block in enumerate(self.blocks):
            if i in retained:
                history[i] = x
            taps = tuple(history[j] if j != i else x for j in c.taps(i)) if block.mudd is not None else ()
            window = c.local_window if c.local(i) else 0
            mask = masks[window] if masks is not None else None
            args = (x, position_ids, mask, first_v, ve if block.ve_gate is not None else None,
                    taps, self.backend, self.native_gqa)
            if self.training and self.checkpointing and i < c.n_layer - self.keep_last:
                x, raw_v = checkpoint(block, *args, use_reentrant=False)
            else:
                x, raw_v = block(*args)
            if i == 0 and c.value_residual:
                first_v = raw_v
        x = self.norm(x)
        if return_hidden:
            return x
        logits = F.linear(x, self.lm_head.weight).float()
        return 23 * torch.sigmoid((logits + 5) / 7.5) if c.sigmoid_logit_cap else logits

    def parameter_groups(self):
        muon, adam = [], []
        for name, p in self.named_parameters():
            large_projection = p.ndim == 2 and any(name.endswith(s) for s in
                ("q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight", "mlp.w1.weight", "mlp.w2.weight", "mlp.w3.weight"))
            (muon if large_projection else adam).append((name, p))
        return muon, adam
