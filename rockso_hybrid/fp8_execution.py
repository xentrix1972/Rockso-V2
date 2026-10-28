"""GEMM-only zero padding, retaining native TorchAO modules and autograd.

The execution hook is private to this trainer process. It changes only FP8
contractions with the configured logical MLP width. Parameters, independent
scales, optimizer matrices and all returned tensor shapes stay unchanged.
"""
import torch
import triton
import triton.language as tl
from torchao.float8.config import Float8LinearConfig
from torchao.float8.float8_linear import Float8Linear
import torchao.float8.float8_ops as ops

_native_addmm = ops.addmm_float8_unwrapped
_padding = None


@triton.jit
def _copy_pad(X,Y,M:tl.constexpr,N:tl.constexpr,P:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,B:tl.constexpr):
 idx=tl.program_id(0)*B+tl.arange(0,B)
 row=idx//P;col=idx%P
 value=tl.load(X+row*S0+col*S1,(idx<M*P)&(col<N),other=0)
 tl.store(Y+idx,value,idx<M*P)

@torch.library.custom_op("rockso_perf::pad_fp8_bytes",mutates_args=(),device_types="cuda")
def pad_bytes(x:torch.Tensor,width:int)->torch.Tensor:
 y=torch.empty((x.shape[0],width),device=x.device,dtype=x.dtype)
 _copy_pad[(triton.cdiv(y.numel(),1024),)](x.view(torch.uint8),y.view(torch.uint8),x.shape[0],x.shape[1],width,x.stride(0),x.stride(1),1024)
 return y

@pad_bytes.register_fake
def _(x,width):return torch.empty((x.shape[0],width),device=x.device,dtype=x.dtype)


def _padded_addmm(a_data, a_scale, b_data, b_scale, output_dtype,
                  output_scale=None, bias=None, use_fast_accum=False):
    logical, physical = _padding
    if a_data.shape[1] == logical and b_data.shape[0] == logical:
        # Quantization and amax already happened in the unmodified TorchAO path.
        # Preserve its required row-major A and column-major B layouts.
        a_data = pad_bytes(a_data, physical)
        b_data = pad_bytes(b_data.t(), physical).t()
    return _native_addmm(a_data, a_scale, b_data, b_scale, output_dtype,
                         output_scale, bias, use_fast_accum)


def install_mlp_padding(model, pad_to):
    global _padding
    if pad_to == 0:
        ops.addmm_float8_unwrapped = _native_addmm
        _padding = None
        return
    hidden = model.config.mlp_hidden
    if model.config.mlp_activation != "swiglu":
        raise ValueError("This qualified execution path is for SwiGLU")
    if pad_to <= hidden or pad_to % 128:
        raise ValueError("Padding must exceed the logical MLP width and be a multiple of 128")
    for block in model.blocks:
        for name in ("w1", "w2", "w3"):
            layer = getattr(block.mlp, name)
            if type(layer) is not Float8Linear or layer.config != Float8LinearConfig():
                raise ValueError("Padding requires native default tensorwise TorchAO Float8Linear")
    if _padding is not None and _padding != (hidden, pad_to):
        raise ValueError("A process may use only one qualified FP8 padding shape")
    _padding = (hidden, pad_to)
    ops.addmm_float8_unwrapped = _padded_addmm
