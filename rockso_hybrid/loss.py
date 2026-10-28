"""Full-vocabulary, response-token-normalized objectives (no sampled softmax)."""
import torch
from torch.nn import functional as F


class ChunkedLinearCE(torch.autograd.Function):
    """Bounded-row reference CE, with recomputed logits in backward.

    The sigmoid variant follows Violetto's 23*sigmoid((z+5)/7.5), including
    its derivative. This is a correctness path, not a claimed fast kernel.
    Higher-order differentiation is intentionally unsupported.
    """
    @staticmethod
    def forward(ctx, x, weight, targets, cap, chunk_rows):
        ctx.save_for_backward(x, weight, targets)
        ctx.cap, ctx.chunk_rows = cap, chunk_rows
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        total = torch.zeros((), dtype=acc_dtype, device=x.device)
        for start in range(0, x.shape[0], chunk_rows):
            z = F.linear(x[start:start + chunk_rows], weight).to(acc_dtype)
            if cap:
                z = 23 * torch.sigmoid((z + 5) / 7.5)
            total += F.cross_entropy(z, targets[start:start + chunk_rows], reduction="sum")
        return total / max(1, x.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        x, weight, targets = ctx.saved_tensors
        acc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        dx = torch.empty_like(x)
        dw = torch.zeros(weight.shape, dtype=acc_dtype, device=weight.device)
        for start in range(0, x.shape[0], ctx.chunk_rows):
            xi, yi = x[start:start + ctx.chunk_rows], targets[start:start + ctx.chunk_rows]
            z = F.linear(xi, weight).to(acc_dtype)
            sigmoid = torch.sigmoid((z + 5) / 7.5) if ctx.cap else None
            logits = 23 * sigmoid if ctx.cap else z
            dz = logits.softmax(-1)
            dz[torch.arange(yi.numel(), device=yi.device), yi] -= 1
            if ctx.cap:
                dz *= (23 / 7.5) * sigmoid * (1 - sigmoid)
            # Upstream microbatch weighting must reach both X and W gradients.
            dz *= grad_output.to(acc_dtype) / max(1, x.shape[0])
            dx[start:start + ctx.chunk_rows] = (dz @ weight.to(acc_dtype)).to(x.dtype)
            dw.add_(dz.T @ xi.to(acc_dtype))
        return dx, dw.to(weight.dtype), None, None, None


def causal_loss(hidden, weight, labels, *, backend="reference", sigmoid_cap=False, chunk_rows=128):
    if backend not in ("reference", "liger"):
        raise ValueError(f"Unknown loss backend: {backend}")
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    targets = labels[:, 1:].reshape(-1)
    keep = targets != -100
    x = hidden[:, :-1].reshape(-1, hidden.shape[-1])[keep]
    y = targets[keep]
    if y.numel() == 0:
        return x.sum() * 0.0
    if backend == "liger":
        if sigmoid_cap:
            raise ValueError("Liger's softcap is not Violetto's sigmoid. Use the reference loss for this ablation.")
        if hidden.device.type != "cuda":
            raise ValueError("Liger requires CUDA")
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        return LigerFusedLinearCrossEntropyLoss()(weight, x.contiguous(), y.contiguous())
    return ChunkedLinearCE.apply(x, weight, y, sigmoid_cap, chunk_rows)
