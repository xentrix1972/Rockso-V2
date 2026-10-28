"""FP32 residual mixing without retaining FP32 copies of BF16 activations."""
import torch


class _ResidualMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, residual, branch, residual_scale, branch_scale):
        ctx.save_for_backward(residual, branch, residual_scale, branch_scale)
        # Double is used only by the derivative qualification. Training and
        # inference accumulate in FP32 and round the final residual once.
        dtype = torch.float64 if residual.dtype == torch.float64 else torch.float32
        return (residual.to(dtype) * residual_scale.to(dtype) +
                branch.to(dtype) * branch_scale.to(dtype)).to(residual.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        residual, branch, residual_scale, branch_scale = ctx.saved_tensors
        dtype = torch.float64 if residual.dtype == torch.float64 else torch.float32
        grad = grad_output.to(dtype)
        dr = (grad * residual_scale.to(dtype)).to(residual.dtype) if ctx.needs_input_grad[0] else None
        db = (grad * branch_scale.to(dtype)).to(branch.dtype) if ctx.needs_input_grad[1] else None
        ds = (grad * residual.to(dtype)).sum().to(residual_scale.dtype) if ctx.needs_input_grad[2] else None
        dt = (grad * branch.to(dtype)).sum().to(branch_scale.dtype) if ctx.needs_input_grad[3] else None
        return dr, db, ds, dt


def residual_mix(residual, branch, residual_scale, branch_scale):
    return _ResidualMix.apply(residual, branch, residual_scale, branch_scale)
