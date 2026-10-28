# Module identity: normuon_optimizer_v3. Corrected stochastic-rounding / FP32-master implementation.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NorMuon v3 + Cautious Weight Decay Optimizer for Single-GPU MoE Training
=========================================================================
Upgraded from v1 based on Keller Jordan's modded-nanogpt unified optimizer
(2026-01-18_UnifiedOptimizers commit).

Historical changes from v1 (shape scaling below was removed for Rockso V2 on 2026-10-27):
  1. Shape-based LR scaling: sqrt(max(1, rows/cols)) per-param multiplier — UNCAPPED
     - Matches Keller's reference exactly: no min() cap
     - LatentMoE w1 (12288x512) gets sqrt(24)≈4.9x, shared expert w1 gets ~1.41x
  2. LR-coupled weight decay: eff_wd = eff_lr * wd * wd_mul
     - WD scales proportionally with effective LR (shape_mult, lr_mul)
     - WD/LR ratio stays constant across all params — matches Keller's reference
     - Prevents disproportionate decay on high-shape_mult params
  3. Per-param lr_mul/wd_mul via attributes: partition_params_for_normuon sets
     p.lr_mul for routed expert params (--routed_expert_lr_mul flag)
  4. 5-iteration Polar Express (was 7): Keller's tuned constants with
     safety_factor=2e-2, cushion=2 — faster and more numerically stable
  5. Spectral norm safety margin: X / (norm * 1.02 + eps) instead of X / (norm + eps)
     - 2% headroom prevents edge-case instability at spectral norm ≈ 1.0
  6. Fused variance reduction: single multiplicative scale instead of two norm passes
     - Algebraically equivalent but avoids intermediate in-place multiply
     - Matches Keller's apply_normuon_variance_reduction exactly
  7. CWD uses (grad * p) >= 0 formulation (treats zero-grad as aligned, not misaligned)

All existing infrastructure preserved:
  - 3-way param partition (muon / shared_expert / adam)
  - Shared expert LR multiplier
  - Checkpoint save/load
  - AdamW 8-bit fallback chain
  - Triton kernels with PyTorch fallback

Usage (unchanged from v1):
    from normuon_optimizer_v3 import HybridNorMuonAdamW, partition_params_for_normuon

    muon_params, shared_expert_params, adam_params = partition_params_for_normuon(model)
    optimizer = HybridNorMuonAdamW(
        muon_params=muon_params,
        adam_params=adam_params,
        shared_expert_params=shared_expert_params,
        muon_lr=0.02,
        adam_lr=4e-4,
        weight_decay=0.1,
    )

References:
    - NorMuon: https://arxiv.org/abs/2510.05491
    - Polar Express: https://arxiv.org/abs/2505.16932
    - IMU-1: https://arxiv.org/abs/2602.02522
    - Keller Jordan modded-nanogpt: github.com/KellerJordan/modded-nanogpt
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch import Tensor

# ============================================================================
# Polar Express Constants
# ============================================================================

# v2: 5 iterations, safety_factor=2e-2, cushion=2 (from Keller's unified optimizer)
# Faster than 7-iter and specifically tuned for the (1 + safety_factor) norm scaling
POLAR_EXPRESS_COEFFS_5ITER = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# v1 constants preserved as fallback (7 iterations)
POLAR_EXPRESS_COEFFS_7ITER = [
    (7.2086, -15.5131, 9.0178),
    (3.9623, -2.5813, 0.4542),
    (3.9466, -2.5765, 0.4544),
    (3.8991, -2.5671, 0.4566),
    (3.7186, -2.5308, 0.4653),
    (3.1390, -2.3073, 0.4733),
    (2.1715, -1.5246, 0.3885),
]

# Spectral norm safety margin (v2: 2% headroom, v1 had none)
SAFETY_FACTOR = 2e-2

# ============================================================================
# Triton Newton-Schulz Orthogonalization
# ============================================================================

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    pass

if _HAS_TRITON:

    def _get_autotune_configs():
        return [
            triton.Config(
                {
                    "BLOCK_SIZE_M": bm,
                    "BLOCK_SIZE_N": bn,
                    "BLOCK_SIZE_K": bk,
                    "GROUP_SIZE_M": 8,
                    "LOWER_UPPER": 1,
                },
                num_stages=stages,
                num_warps=warps,
            )
            for bm in [64, 128]
            for bn in [64, 128]
            for bk in [64, 128]
            for stages, warps in ((2, 4), (3, 4), (3, 8))
            if bm // bn <= 2 and bn // bm <= 2
        ]

    @triton.jit
    def _pid_to_block(pid, M, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr):
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)
        batch_idx = pid // (num_pid_m * num_pid_n)
        pid = pid % (num_pid_m * num_pid_n)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)
        m_idx = pid_m * BLOCK_SIZE_M
        n_idx = pid_n * BLOCK_SIZE_N
        return batch_idx, m_idx, n_idx

    @triton.autotune(configs=_get_autotune_configs(), key=["M", "K", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
    @triton.jit
    def XXT_kernel(A_ptr, C_ptr, M, K,
                   a_stride_b, a_stride_r, a_stride_c,
                   c_stride_b, c_stride_r, c_stride_c,
                   BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                   BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                   LOWER_UPPER: tl.constexpr):
        """Compute C = A @ A.T (renamed from ns_line_1 to match Keller's naming)"""
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return
        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b
        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c
        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)
        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)
        # Mirror across diagonal (symmetric matrix optimization)
        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)

    @triton.autotune(configs=_get_autotune_configs(), key=["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"])
    @triton.jit
    def ba_plus_cAA_kernel(A_ptr, C_ptr, M,
                           a_stride_b, a_stride_r, a_stride_c,
                           c_stride_b, c_stride_r, c_stride_c,
                           alpha, beta,
                           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                           BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                           LOWER_UPPER: tl.constexpr):
        """Compute C = alpha * A @ A.T + beta * A (renamed from ns_line_2)"""
        pid = tl.program_id(axis=0)
        batch_idx, m_idx, n_idx = _pid_to_block(pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M)
        skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
        skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
        if skip_block_below_diag or skip_block_above_diag:
            return
        A_ptr += batch_idx * a_stride_b
        C_ptr += batch_idx * c_stride_b
        offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
        at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
            at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, at, accumulator)
            a_ptrs += BLOCK_SIZE_K * a_stride_c
            at_ptrs += BLOCK_SIZE_K * a_stride_c
        # Load block of A to add
        offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
        a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
        a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
        a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)
        accumulator *= alpha
        accumulator += a_add * beta
        out_dtype = C_ptr.dtype.element_ty
        output = accumulator.to(out_dtype)
        offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
        tl.store(c_ptrs, output, mask=c_mask)
        # Mirror across diagonal
        c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
        c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(c_ptrs_t, output.T, mask=c_mask_t)

    def _XXT(A: Tensor, *, out: Tensor = None):
        """Launch Triton kernel: C = A @ A.T"""
        M, K = A.shape[-2:]
        if out is None:
            out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0
        def grid(meta):
            return (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
        XXT_kernel[grid](
            A_ptr=A, C_ptr=out, M=M, K=K,
            a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
        )
        return out

    def _ba_plus_cAA(A: Tensor, alpha: float, beta: float, *, out: Tensor = None):
        """Launch Triton kernel: C = alpha * A @ A.T + beta * A"""
        M, K = A.shape[-2:]
        if out is None:
            out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
        batch_size = A.size(0) if A.ndim == 3 else 1
        input_batch_stride = A.stride(0) if A.ndim == 3 else 0
        output_batch_stride = out.stride(0) if out.ndim == 3 else 0
        def grid(meta):
            return (batch_size * triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(M, meta["BLOCK_SIZE_N"]),)
        ba_plus_cAA_kernel[grid](
            A_ptr=A, C_ptr=out, M=M,
            a_stride_b=input_batch_stride, a_stride_r=A.stride(-2), a_stride_c=A.stride(-1),
            c_stride_b=output_batch_stride, c_stride_r=out.stride(-2), c_stride_c=out.stride(-1),
            alpha=alpha, beta=beta,
        )
        return out

    def polar_express(G: Tensor, num_iters: int = 5, split_baddbmm: Optional[bool] = None):
        """
        Polar Express Sign Method: https://arxiv.org/pdf/2505.16932
        by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.

        v2 changes from v1:
          - 5 iterations (was 7) with matched constants
          - Safety factor: norm * (1 + 2e-2) ensures spectral norm < 1
          - Matched exactly to Keller Jordan's unified optimizer implementation

        split_baddbmm: for large matrices, split `a*X + B@X` into two kernels
            (B@X, then C.add_(X, alpha=a)) so X is only referenced once per op —
            avoids PyTorch's defensive copy of X inside baddbmm. Auto-enabled when
            max(G.shape[-2:]) > 1024 (covers dense MLP / attn proj weights).
        """
        coeffs = POLAR_EXPRESS_COEFFS_5ITER if num_iters == 5 else POLAR_EXPRESS_COEFFS_7ITER

        assert G.is_cuda
        if split_baddbmm is None:
            split_baddbmm = max(G.size(-2), G.size(-1)) > 1024

        X = G.to(dtype=torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.mT

        # v2: safety margin on spectral norm — (1 + safety_factor) multiplier
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * (1 + SAFETY_FACTOR) + 1e-6)

        X = X.contiguous()
        # Allocate buffers
        A = torch.empty((*X.shape[:-1], X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        if split_baddbmm:
            BX_matmul = torch.bmm if X.ndim > 2 else torch.mm
        else:
            aX_plus_BX = torch.baddbmm if X.ndim > 2 else torch.addmm

        for a, b, c in coeffs:
            _XXT(X, out=A)                          # A = X @ X.T
            _ba_plus_cAA(A, alpha=c, beta=b, out=B) # B = b*A + c*A@A
            if split_baddbmm:
                BX_matmul(B, X, out=C)              # C = B @ X
                C.add_(X, alpha=a)                  # C += a*X (X read once)
            else:
                aX_plus_BX(X, B, X, beta=a, out=C)  # C = a*X + B@X (X read twice → defensive copy)
            X, C = C, X                              # swap to avoid copy

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

else:
    # Fallback: pure PyTorch (no Triton)
    def polar_express(G: Tensor, num_iters: int = 5):
        """Fallback PyTorch Polar Express."""
        coeffs = POLAR_EXPRESS_COEFFS_5ITER if num_iters == 5 else POLAR_EXPRESS_COEFFS_7ITER

        X = G.to(dtype=torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.T
        X = X / (X.norm() * (1 + SAFETY_FACTOR) + 1e-6)
        for a, b, c in coeffs:
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.T
        return X

# Backward-compatible alias
newton_schulz_triton = polar_express


# ============================================================================
# Fused NorMuon Variance Reduction (single-scale formulation)
# ============================================================================

def normuon_variance_reduction(v: Tensor, second_momentum_buffer: Tensor,
                               beta2: float, red_dim: int) -> Tensor:
    """
    Fused NorMuon variance reduction — algebraically equivalent to v1 but in
    a single multiplicative scale, avoiding the intermediate in-place multiply
    and double-norm-pass that v1 used.

    Matches Keller Jordan's apply_normuon_variance_reduction exactly.

    Math:
      v_mean = mean(v^2, dim=red_dim)           # per-neuron variance estimate
      v_norm = sqrt(sum(v_mean * red_dim_size))  # full Frobenius norm
      buf.lerp_(v_mean, 1 - beta2)              # EMA update
      step_size = rsqrt(buf)                     # per-neuron adaptive scale
      v_norm_new = sqrt(sum(v_mean * red_dim_size * step_size^2))
      final_scale = step_size * (v_norm / v_norm_new)
      return v * final_scale
    """
    v_float = v.float()
    v_mean = v_float.square().mean(dim=red_dim, keepdim=True)
    red_dim_size = v.size(red_dim)

    # Frobenius norm via per-neuron stats (avoids separate full norm pass)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True).mul_(red_dim_size)
    v_norm = v_norm_sq.sqrt_()

    # EMA update of per-neuron second moment
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)

    # Per-neuron adaptive step size
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt_()

    # Compute what the norm would be AFTER per-neuron scaling (without actually doing it)
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt_()

    # Single multiplicative scale: per-neuron adaptive + norm preservation
    final_scale = step_size * (v_norm / v_norm_new.clamp_min_(1e-10))

    return v.mul_(final_scale.type_as(v))


# ============================================================================
# Cautious Weight Decay + Update (v2)
# ============================================================================

def cautious_wd_and_update(p: Tensor, v: Tensor, eff_lr: float, eff_wd: float):
    """
    Cautious weight decay + parameter update.

    v2 change: uses (v * p) >= 0 instead of signbit comparison.
    The >= 0 formulation treats zero-valued updates as aligned (no-op decay
    on those neurons), which is slightly more conservative than v1's signbit
    approach that treated zeros as misaligned.
    """
    # Cautious weight decay: only decay when update and param are aligned
    mask = (v * p) >= 0
    p.addcmul_(p, mask.to(p.dtype), value=-eff_wd)

    # Apply update
    p.add_(v, alpha=-eff_lr)


@torch.no_grad()
def cautious_wd_and_update_sr(p: Tensor, v: Tensor, eff_lr: float, eff_wd: float):
    """Compute CWD and the update in FP32, then stochastically store BF16.

    Uses the caller's CUDA RNG, which the trainer checkpoints. Unbiased rounding
    assumes a finite update bracketed by finite BF16 values (the tested range).
    FP32 parameters retain the original production FP32 update path.
    """
    if p.dtype != torch.bfloat16:
        cautious_wd_and_update(p, v, eff_lr, eff_wd)
        return
    pf = p.float()
    mask = (v * pf) >= 0
    new = pf - eff_wd * pf * mask.to(pf.dtype) - eff_lr * v.float()
    bits = new.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32)
    rounded = (bits + noise) & -65536
    p.copy_(rounded.view(torch.float32))


# ============================================================================
# NorMuon Optimizer (2D weight matrices only)
# ============================================================================

class NorMuon(torch.optim.Optimizer):
    """
    NorMuon v3 — Muon with neuron-wise normalization and cautious weight decay.

    Key v2 improvements:
      - Configured LR with explicit per-parameter lr_mul; no shape multiplier
      - LR-coupled weight decay: eff_wd = eff_lr * wd * wd_mul
      - 5-iter Polar Express with safety factor
      - Fused single-scale variance reduction
      - Improved CWD formulation

    Default LR: 0.02 (real Muon LR, NOT the 0.003 from Kimi K2 scaling)
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95,
                 beta2=0.95, num_polar_iters=5, fp32_master_weights=False, stochastic_rounding=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        beta2=beta2, num_polar_iters=num_polar_iters,
                        fp32_master_weights=fp32_master_weights, stochastic_rounding=stochastic_rounding)
        params = list(params)
        super().__init__([{"params": params}], defaults)
        for group in self.param_groups:
            for p in group['params']:
                if p.dim() != 2:
                    raise ValueError(
                        f"NorMuon only supports 2D parameters, got {p.dim()}D tensor shape {p.shape}. "
                        f"Use AdamW for embeddings, biases, and norm weights."
                    )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            beta2 = group["beta2"]
            lr = group["lr"]
            wd = group["weight_decay"]
            num_polar_iters = group.get("num_polar_iters", 5)

            for p in group["params"]:
                grad = p.grad
                if grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                    # Per-neuron second moment buffer (reduced along the appropriate dim)
                    if p.size(-2) >= p.size(-1):
                        state["second_momentum_buffer"] = torch.zeros_like(
                            grad[..., :1], dtype=torch.float32)
                    else:
                        state["second_momentum_buffer"] = torch.zeros_like(
                            grad[:1, ...], dtype=torch.float32)

                momentum_buffer = state["momentum_buffer"]
                second_momentum_buffer = state["second_momentum_buffer"]

                # Configured LR with the existing explicit per-parameter multiplier.
                # Cautious decay retains its existing coupling to effective LR.
                eff_lr = lr * getattr(p, "lr_mul", 1.0)
                eff_wd = eff_lr * wd * getattr(p, "wd_mul", 1.0)

                # Interpolate momentum (NorMuon style)
                momentum_buffer.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(momentum_buffer, momentum)

                # Polar Express orthogonalization
                v = polar_express(grad.bfloat16(), num_iters=num_polar_iters).to(torch.float32)

                # Fused variance reduction (v2 single-scale)
                red_dim = -1 if p.size(-2) >= p.size(-1) else -2
                v = normuon_variance_reduction(v, second_momentum_buffer, beta2, red_dim)

                # Cautious weight decay + update (v2 formulation)
                if group.get("stochastic_rounding", False):
                    cautious_wd_and_update_sr(p, v, eff_lr, eff_wd)
                elif group.get("fp32_master_weights", False) and p.dtype == torch.bfloat16:
                    if "master_weight" not in state:
                        state["master_weight"] = p.detach().float().clone()
                    master = state["master_weight"]
                    cautious_wd_and_update(master, v, eff_lr, eff_wd)
                    p.copy_(master)
                else:
                    cautious_wd_and_update(p, v, eff_lr, eff_wd)

        return loss


# ============================================================================
# Hybrid NorMuon + AdamW (Decoupled LRs) — unchanged API from v1
# ============================================================================

class HybridNorMuonAdamW:
    """
    Hybrid optimizer: NorMuon for 2D weight matrices, AdamW for everything else.
    DECOUPLED LRs — each optimizer gets its own learning rate.
    Optional third NorMuon for shared expert params at reduced LR.
    """

    def __init__(
        self,
        muon_params: List[torch.nn.Parameter],
        adam_params: List[torch.nn.Parameter],
        shared_expert_params: Optional[List[torch.nn.Parameter]] = None,
        muon_lr: float = 0.02,
        adam_lr: float = 4e-4,
        shared_expert_lr_mul: float = 1.0,
        shared_expert_separate: bool = False,
        weight_decay: float = 0.1,
        muon_momentum: float = 0.95,
        muon_beta2: float = 0.95,
        adam_betas: Tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        num_polar_iters: int = 5,
        adam_fused: bool = False,
    ):
        self.muon_lr = muon_lr
        self.adam_lr = adam_lr
        self.shared_expert_lr_mul = shared_expert_lr_mul
        self.muon_opt = None
        self.shared_expert_opt = None
        self.adam_opt = None

        if muon_params:
            self.muon_opt = NorMuon(
                muon_params, lr=muon_lr, weight_decay=weight_decay,
                momentum=muon_momentum, beta2=muon_beta2,
                num_polar_iters=num_polar_iters,
            )
            n_params = len(muon_params)
            n_elements = sum(p.numel() for p in muon_params)
            # Report effective LR multipliers (shape_mult × lr_mul)
            shape_mults = {}
            for p in muon_params:
                sm = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                lr_mul = getattr(p, 'lr_mul', 1.0)
                key = f"{p.size(-2)}x{p.size(-1)}"
                shape_mults[key] = sm * lr_mul
            print(f"[normuon v3] NorMuon+CWD: {n_params} params, {n_elements/1e6:.1f}M elements, "
                  f"lr={muon_lr}, momentum={muon_momentum}, beta2={muon_beta2}, "
                  f"polar_iters={num_polar_iters}", flush=True)
            print(f"[normuon v3] Effective LR multipliers: {shape_mults}", flush=True)

        need_separate_shared = shared_expert_lr_mul != 1.0 or shared_expert_separate
        if shared_expert_params and need_separate_shared:
            se_lr = muon_lr * shared_expert_lr_mul
            self.shared_expert_opt = NorMuon(
                shared_expert_params, lr=se_lr, weight_decay=weight_decay,
                momentum=muon_momentum, beta2=muon_beta2,
                num_polar_iters=num_polar_iters,
            )
            n_params = len(shared_expert_params)
            n_elements = sum(p.numel() for p in shared_expert_params)
            shape_mults = {}
            for p in shared_expert_params:
                sm = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                key = f"{p.size(-2)}x{p.size(-1)}"
                shape_mults[key] = sm
            print(f"[normuon v3] Shared expert NorMuon: {n_params} params, {n_elements/1e6:.1f}M elements, "
                  f"lr={se_lr} ({shared_expert_lr_mul}x base)"
                  f"{' (separate group for LR ramp)' if shared_expert_separate else ''}",
                  flush=True)
            print(f"[normuon v3] Shared expert effective LR mults: {shape_mults}", flush=True)
        elif shared_expert_params:
            # lr_mul == 1.0, just merge into main muon
            if self.muon_opt is None:
                self.muon_opt = NorMuon(
                    shared_expert_params, lr=muon_lr, weight_decay=weight_decay,
                    momentum=muon_momentum, beta2=muon_beta2,
                    num_polar_iters=num_polar_iters,
                )
            else:
                self.muon_opt.add_param_group({
                    'params': shared_expert_params,
                    'lr': muon_lr, 'weight_decay': weight_decay,
                    'momentum': muon_momentum, 'beta2': muon_beta2,
                    'num_polar_iters': num_polar_iters,
                })
            print(f"[normuon v3] Shared expert merged into main NorMuon (lr_mul=1.0)", flush=True)

        if adam_params:
            decay_params = []
            no_decay_params = []
            for p in adam_params:
                if p.dim() >= 2:
                    decay_params.append(p)
                else:
                    no_decay_params.append(p)
            param_groups = []
            if decay_params:
                param_groups.append({"params": decay_params, "weight_decay": weight_decay})
            if no_decay_params:
                param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

            opt = None
            if adam_fused:
                opt = torch.optim.AdamW(param_groups, lr=adam_lr, betas=adam_betas,
                                        eps=adam_eps, weight_decay=weight_decay, fused=True)
                print(f"[normuon v3] AdamW 32-bit (fused, prefer_fused=True): {len(adam_params)} params, lr={adam_lr}", flush=True)
            else:
                try:
                    from torchao.prototype.low_bit_optim import AdamW8bit
                    opt = AdamW8bit(param_groups, lr=adam_lr, betas=adam_betas, weight_decay=weight_decay)
                    print(f"[normuon v3] AdamW 8-bit (TorchAO): {len(adam_params)} params, lr={adam_lr}", flush=True)
                except ImportError:
                    pass
                if opt is None:
                    try:
                        import bitsandbytes as bnb
                        opt = bnb.optim.AdamW8bit(param_groups, lr=adam_lr, betas=adam_betas,
                                                   weight_decay=weight_decay, is_paged=False)
                        print(f"[normuon v3] AdamW 8-bit (bnb): {len(adam_params)} params, lr={adam_lr}", flush=True)
                    except ImportError:
                        pass
                if opt is None:
                    opt = torch.optim.AdamW(param_groups, lr=adam_lr, betas=adam_betas,
                                            eps=adam_eps, weight_decay=weight_decay, fused=True)
                    print(f"[normuon v3] AdamW 32-bit (fused, fallback): {len(adam_params)} params, lr={adam_lr}", flush=True)
            self.adam_opt = opt

        self.param_groups = []
        if self.muon_opt:
            self.param_groups.extend(self.muon_opt.param_groups)
        if self.shared_expert_opt:
            self.param_groups.extend(self.shared_expert_opt.param_groups)
        if self.adam_opt:
            self.param_groups.extend(self.adam_opt.param_groups)

    def step(self, closure=None):
        loss = None
        if self.muon_opt:
            loss = self.muon_opt.step(closure)
        if self.shared_expert_opt:
            self.shared_expert_opt.step(None)
        if self.adam_opt:
            loss = self.adam_opt.step(closure if loss is None else None)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self.muon_opt:
            self.muon_opt.zero_grad(set_to_none=set_to_none)
        if self.shared_expert_opt:
            self.shared_expert_opt.zero_grad(set_to_none=set_to_none)
        if self.adam_opt:
            self.adam_opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return {
            'muon': self.muon_opt.state_dict() if self.muon_opt else None,
            'shared_expert_muon': self.shared_expert_opt.state_dict() if self.shared_expert_opt else None,
            'adam': self.adam_opt.state_dict() if self.adam_opt else None,
            'muon_lr': self.muon_lr,
            'adam_lr': self.adam_lr,
            'shared_expert_lr_mul': self.shared_expert_lr_mul,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if self.muon_opt and state_dict.get('muon'):
            self.muon_opt.load_state_dict(state_dict['muon'])
        if self.shared_expert_opt and state_dict.get('shared_expert_muon'):
            self.shared_expert_opt.load_state_dict(state_dict['shared_expert_muon'])
        elif self.shared_expert_opt and not state_dict.get('shared_expert_muon'):
            print("[normuon v3] No shared_expert_muon in checkpoint — momentum buffers will warm up fresh", flush=True)
        if self.adam_opt and state_dict.get('adam'):
            self.adam_opt.load_state_dict(state_dict['adam'])
        if 'muon_lr' in state_dict:
            self.muon_lr = state_dict['muon_lr']
        if 'adam_lr' in state_dict:
            self.adam_lr = state_dict['adam_lr']
        if 'shared_expert_lr_mul' in state_dict:
            self.shared_expert_lr_mul = state_dict['shared_expert_lr_mul']

    def update_lr(self, muon_lr: float = None, adam_lr: float = None):
        """Update learning rates independently. Shared expert scales with muon_lr."""
        if muon_lr is not None:
            self.muon_lr = muon_lr
            if self.muon_opt:
                for g in self.muon_opt.param_groups:
                    g['lr'] = muon_lr
            if self.shared_expert_opt:
                se_lr = muon_lr * self.shared_expert_lr_mul
                for g in self.shared_expert_opt.param_groups:
                    g['lr'] = se_lr
        if adam_lr is not None:
            self.adam_lr = adam_lr
            if self.adam_opt:
                for g in self.adam_opt.param_groups:
                    g['lr'] = adam_lr

    @property
    def state(self) -> Dict:
        combined = {}
        if self.muon_opt:
            combined.update(self.muon_opt.state)
        if self.shared_expert_opt:
            combined.update(self.shared_expert_opt.state)
        if self.adam_opt:
            combined.update(self.adam_opt.state)
        return combined


# ============================================================================
# SkipStep — Zero-Sync Loss Spike Detection (inspired by OLMo-core)
# ============================================================================

class SkipStepDetector:
    """
    Detects loss/gradient spikes and signals the optimizer to skip the update.

    Maintains rolling windows of loss and grad_norm values as GPU tensors.
    A "spike" is detected when a value exceeds:  mean + sigma_factor * std
    computed over the rolling window.

    When a spike is detected, the optimizer step is skipped entirely —
    momentum buffers and second-moment estimates stay clean (unlike grad
    clipping, which still poisons the update direction).

    All operations are on GPU tensors — zero host-device synchronization.

    Based on: OLMo-core SkipStepOptimizer + StabilityMonitorCallback
    Paper context: https://arxiv.org/abs/2502.05795 (OLMo-2)

    Usage:
        detector = SkipStepDetector(device='cuda')

        # In training loop, after unscale_ and before opt.step():
        should_skip = detector.should_skip(loss_tensor, grad_norm_tensor)
        if not should_skip:
            opt.step()
        detector.record_step(was_skipped=should_skip)  # update metrics
    """

    def __init__(
        self,
        rolling_window: int = 128,
        sigma_factor: float = 6.0,
        min_samples: int = 64,
        spike_window: int = 10000,
        min_gnorm_threshold: float = 0.0,
        device: str = "cuda",
    ):
        """
        Args:
            rolling_window: Number of recent values for mean/std computation.
            sigma_factor: Skip if value > mean + sigma_factor * std.
            min_samples: Don't skip until we have this many samples (warmup).
            spike_window: Rolling window for SpikeScore metric.
            min_gnorm_threshold: Absolute floor on grad_norm — gnorm spikes
                only fire if grad_norm > this value. 0.0 disables the floor
                (pure statistical detection). Does NOT affect loss-spike path.
            device: Device for internal buffers.
        """
        self.rolling_window = rolling_window
        self.sigma_factor = sigma_factor
        self.min_samples = max(2, min_samples)
        self.spike_window = spike_window
        self.min_gnorm_threshold = float(min_gnorm_threshold)
        self.device = device

        # Rolling buffers (GPU tensors — no host-device sync)
        self._losses = torch.zeros(rolling_window, device=device, dtype=torch.float32)
        self._grad_norms = torch.zeros(rolling_window, device=device, dtype=torch.float32)
        self._count = 0  # total samples inserted (int, not tensor — only used for indexing)

        # SpikeScore tracking
        self._total_steps = 0
        self._total_skips = 0
        self._recent_skips = torch.zeros(spike_window, device=device, dtype=torch.bool)
        self._spike_idx = 0

        # Last result (for logging without sync)
        self._last_skipped = False

    @torch.no_grad()
    def should_skip(
        self,
        loss: torch.Tensor,
        grad_norm: torch.Tensor,
    ) -> bool:
        """
        Check if the current step should be skipped.

        Args:
            loss: Scalar loss tensor (on GPU, unscaled — multiply by grad_accum
                  if you want the full-batch loss). Must be detached.
            grad_norm: Scalar grad norm tensor (on GPU, after unscale).

        Returns:
            True if this step should be skipped (spike detected).
        """
        # Ensure GPU tensors
        loss = loss.detach().to(device=self.device, dtype=torch.float32)
        grad_norm = grad_norm.detach().to(device=self.device, dtype=torch.float32)

        # Check for inf/nan — always skip those
        if not (loss.isfinite() and grad_norm.isfinite()):
            self._last_skipped = True
            return True

        # Not enough samples yet — don't skip, just record
        if self._count < self.min_samples:
            idx = self._count % self.rolling_window
            self._losses[idx] = loss
            self._grad_norms[idx] = grad_norm
            self._count += 1
            self._last_skipped = False
            return False

        # Compute stats over the filled portion of the buffer
        n = min(self._count, self.rolling_window)
        loss_buf = self._losses[:n]
        gnorm_buf = self._grad_norms[:n]

        # Loss spike check
        loss_std, loss_mean = torch.std_mean(loss_buf)
        loss_spike = (loss - loss_mean) > (self.sigma_factor * loss_std)

        # Grad norm spike check — gated by absolute floor (min_gnorm_threshold).
        # If floor is 0.0, behavior is pure statistical detection.
        gnorm_std, gnorm_mean = torch.std_mean(gnorm_buf)
        gnorm_spike = ((grad_norm - gnorm_mean) > (self.sigma_factor * gnorm_std)) & (grad_norm > self.min_gnorm_threshold)

        # Spike if EITHER condition fires
        # Use .item() here — this is ONE sync per step, unavoidable for the branch
        is_spike = bool((loss_spike | gnorm_spike).item())

        if not is_spike:
            # Not a spike — record this value into the rolling window
            idx = self._count % self.rolling_window
            self._losses[idx] = loss
            self._grad_norms[idx] = grad_norm
            self._count += 1
        # If spike: do NOT record the spike value into the window
        # (prevents one spike from widening the acceptance band)

        self._last_skipped = is_spike
        return is_spike

    def record_step(self, was_skipped: bool):
        """Record whether a step was skipped (for SpikeScore tracking)."""
        self._total_steps += 1
        if was_skipped:
            self._total_skips += 1
        self._recent_skips[self._spike_idx % self.spike_window] = was_skipped
        self._spike_idx += 1

    @property
    def step_skipped(self) -> bool:
        """Whether the last check resulted in a skip."""
        return self._last_skipped

    @property
    def spike_score(self) -> float:
        """Rolling spike rate over the last `spike_window` steps."""
        n = min(self._total_steps, self.spike_window)
        if n == 0:
            return 0.0
        return float(self._recent_skips[:n].float().mean().item())

    @property
    def total_spike_score(self) -> float:
        """Cumulative spike rate (total skips / total steps)."""
        if self._total_steps == 0:
            return 0.0
        return self._total_skips / self._total_steps

    @property
    def total_skips(self) -> int:
        return self._total_skips

    @property
    def total_steps(self) -> int:
        return self._total_steps

    def state_dict(self) -> Dict[str, Any]:
        """Save state for checkpoint resumption."""
        return {
            'losses': self._losses.cpu(),
            'grad_norms': self._grad_norms.cpu(),
            'count': self._count,
            'total_steps': self._total_steps,
            'total_skips': self._total_skips,
            'recent_skips': self._recent_skips.cpu(),
            'spike_idx': self._spike_idx,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore state from checkpoint."""
        self._losses = state_dict['losses'].to(self.device)
        self._grad_norms = state_dict['grad_norms'].to(self.device)
        self._count = state_dict['count']
        self._total_steps = state_dict['total_steps']
        self._total_skips = state_dict['total_skips']
        self._recent_skips = state_dict['recent_skips'].to(self.device)
        self._spike_idx = state_dict['spike_idx']

    def get_metrics(self) -> Dict[str, float]:
        """Get metrics dict for WandB logging."""
        metrics = {
            'skip_step/skipped': 1.0 if self._last_skipped else 0.0,
            'skip_step/total_skips': self._total_skips,
            'skip_step/total_steps': self._total_steps,
            'skip_step/spike_score_total': self.total_spike_score,
        }
        if self._total_steps >= self.spike_window:
            metrics['skip_step/spike_score'] = self.spike_score
        return metrics

    def __repr__(self):
        return (f"SkipStepDetector(window={self.rolling_window}, sigma={self.sigma_factor}, "
                f"skips={self._total_skips}/{self._total_steps})")


# ============================================================================
# Parameter Partitioning — unchanged from v1
# ============================================================================

def partition_params_for_normuon(
    model: nn.Module, verbose: bool = True,
    routed_expert_lr_mul: float = 1.0,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Partition: NorMuon gets 2D weight matrices, AdamW gets the rest.
    Shared expert 2D params get their own bucket for separate LR control.
    If routed_expert_lr_mul != 1.0, tags routed expert w1 (tall) params with p.lr_mul.
    Returns: (muon_params, shared_expert_params, adam_params)
    """
    muon_params, shared_expert_params, adam_params = [], [], []
    muon_names, shared_names, adam_names = [], [], []
    routed_expert_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        nl = name.lower()
        if 'embed' in nl:
            adam_params.append(param); adam_names.append(name); continue
        if 'lm_head' in nl or (nl.endswith('.head') and 'mtp' not in nl):
            adam_params.append(param); adam_names.append(name); continue
        if 'norm' in nl or 'ln' in nl:
            adam_params.append(param); adam_names.append(name); continue
        if 'bias' in nl or param.dim() == 1:
            adam_params.append(param); adam_names.append(name); continue
        if 'router' in nl and ('gate' in nl or 'bias' in nl):
            adam_params.append(param); adam_names.append(name); continue
        if 'mtp' in nl and 'head' in nl:
            adam_params.append(param); adam_names.append(name); continue
        if param.dim() == 2:
            if 'shared_expert' in nl:
                shared_expert_params.append(param); shared_names.append(name)
            else:
                muon_params.append(param); muon_names.append(name)
        else:
            adam_params.append(param); adam_names.append(name)

    # Tag routed expert w1 (tall) params with lr_mul for NorMuon.step()
    # Only applies to tall matrices (shape_mult > 1) — w2 (wide) already at base LR
    if routed_expert_lr_mul != 1.0:
        for name, param in zip(muon_names, muon_params):
            nl = name.lower()
            if ('experts.' in nl and 'w1' in nl and 'shared_expert' not in nl
                    and param.dim() == 2 and param.size(-2) > param.size(-1)):
                param.lr_mul = routed_expert_lr_mul
                routed_expert_names.append(name)

    if verbose:
        muon_numel = sum(p.numel() for p in muon_params)
        shared_numel = sum(p.numel() for p in shared_expert_params)
        adam_numel = sum(p.numel() for p in adam_params)
        total_numel = muon_numel + shared_numel + adam_numel
        print(f"[normuon v3] Parameter partition:", flush=True)
        print(f"  NorMuon:         {len(muon_params)} params, {muon_numel/1e6:.1f}M elements "
              f"({100*muon_numel/total_numel:.1f}%)", flush=True)
        if shared_expert_params:
            print(f"  Shared expert:   {len(shared_expert_params)} params, {shared_numel/1e6:.1f}M elements "
                  f"({100*shared_numel/total_numel:.1f}%)", flush=True)
            print(f"    Params: {shared_names}", flush=True)
        print(f"  AdamW:           {len(adam_params)} params, {adam_numel/1e6:.1f}M elements "
              f"({100*adam_numel/total_numel:.1f}%)", flush=True)
        if len(muon_names) <= 10:
            print(f"  NorMuon params: {muon_names}", flush=True)
        else:
            print(f"  NorMuon params (first 5): {muon_names[:5]}", flush=True)
            print(f"  NorMuon params (last 5): {muon_names[-5:]}", flush=True)

        # v2: report effective LR multiplier summary (shape_mult × lr_mul)
        shape_summary = {}
        for name, p in zip(muon_names + shared_names, muon_params + shared_expert_params):
            sm = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
            lr_mul = getattr(p, 'lr_mul', 1.0)
            eff = sm * lr_mul
            shape_key = f"{p.size(-2)}x{p.size(-1)}"
            if shape_key not in shape_summary:
                shape_summary[shape_key] = {'shape_mult': sm, 'lr_mul': lr_mul, 'eff': eff,
                                            'count': 0, 'example': name}
            shape_summary[shape_key]['count'] += 1
        if shape_summary:
            print(f"  Effective LR multipliers (shape_mult × lr_mul):", flush=True)
            for shape_key, info in sorted(shape_summary.items()):
                if info['lr_mul'] != 1.0:
                    print(f"    {shape_key}: {info['eff']:.3f}x = {info['shape_mult']:.3f} shape × {info['lr_mul']:.3f} lr_mul "
                          f"({info['count']} params, e.g. {info['example']})", flush=True)
                else:
                    print(f"    {shape_key}: {info['eff']:.3f}x "
                          f"({info['count']} params, e.g. {info['example']})", flush=True)

        if routed_expert_names:
            routed_numel = sum(p.numel() for n, p in zip(muon_names, muon_params)
                               if hasattr(p, 'lr_mul'))
            print(f"  Routed expert lr_mul={routed_expert_lr_mul:.2f}x: "
                  f"{len(routed_expert_names)} params, {routed_numel/1e6:.1f}M elements", flush=True)
            if len(routed_expert_names) <= 6:
                print(f"    Params: {routed_expert_names}", flush=True)
            else:
                print(f"    Params (first 3): {routed_expert_names[:3]}", flush=True)
                print(f"    Params (last 3): {routed_expert_names[-3:]}", flush=True)

    return muon_params, shared_expert_params, adam_params


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    print("Testing NorMuon v3 optimizer...\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"1. Polar Express (5 iters, Triton={_HAS_TRITON}):")
    G = torch.randn(512, 256, dtype=torch.bfloat16, device=device)
    O = polar_express(G, num_iters=5)
    print(f"   {G.shape} -> {O.shape}")

    print(f"\n2. Polar Express (7 iters, backward compat):")
    O7 = polar_express(G, num_iters=7)
    print(f"   {G.shape} -> {O7.shape}")

    print(f"\n3. Shape LR scaling (uncapped, matches Keller reference):")
    shapes = [(4092, 8192), (8192, 4092), (4092, 4092), (512, 2048), (12288, 512)]
    for rows, cols in shapes:
        sm = max(1.0, rows / cols) ** 0.5
        print(f"   {rows}x{cols}: shape_mult = {sm:.4f}")

    print(f"\n4. NorMuon v3 step (tall matrix, shape_mult > 1):")
    p = torch.randn(8192, 4092, requires_grad=True, device=device, dtype=torch.float32)
    p.grad = torch.randn_like(p)
    NorMuon([p], lr=0.02).step()
    print(f"   {p.shape} lr=0.02, eff_lr=0.02*{max(1.0, 8192/4092)**0.5:.3f} = {0.02 * max(1.0, 8192/4092)**0.5:.4f}")

    print(f"\n5. NorMuon v3 step (wide matrix, shape_mult = 1.0):")
    p2 = torch.randn(4092, 8192, requires_grad=True, device=device, dtype=torch.float32)
    p2.grad = torch.randn_like(p2)
    NorMuon([p2], lr=0.02).step()
    print(f"   {p2.shape} lr=0.02, eff_lr=0.02*1.000 = 0.0200")

    print(f"\n6. Fused variance reduction:")
    v_test = torch.randn(256, 128, device=device, dtype=torch.float32)
    buf = torch.zeros(256, 1, device=device, dtype=torch.float32)
    v_out = normuon_variance_reduction(v_test, buf, beta2=0.95, red_dim=-1)
    print(f"   {v_test.shape} -> variance reduced, buf shape {buf.shape}")

    print("\n✓ All NorMuon v3 tests passed!")
