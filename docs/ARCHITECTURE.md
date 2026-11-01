# Architecture and numerical paths

The explicit configuration is `configs/rockso_v2_32.json`:

- 32 transformer layers, width 2048; 32 query heads and 8 key/value heads.
- Vocabulary 65,536 with tied input/output embeddings; sequence length 2048.
- SwiGLU intermediate width 6832, with q/k/v/output and MLP projections eligible
  for FP8 execution. GEMM-only padding to 6912 preserves logical parameter and
  optimizer shapes.
- Half RoPE, base 10000, with FP32 RoPE math; global NoPE disabled.
- Attention output gates (128 channels), value embeddings at configured sites
  (start 1, every 3, gate channels 12), and XSA signed value-direction suppression.
  XSA is not a self-key removal mask.
- MUDD at zero-based layers 16 and 31, rank 32, with taps `(0, layer//2, layer)`
  and shared reduction.
- Learned FP32 residual and branch scalars. Value residual, sigmoid logit cap
  and smoothing are disabled in this recipe. Local window is zero.

Training uses PrefixLM Flex masks with response-only shifted loss and Liger loss
execution. BF16 storage is combined with FP32 numerical paths for norms, gates,
XSA, MUDD and scalar parameters. Activation checkpointing retains the last two
layers' activations in the documented recipe.

`HybridOptimizer` routes large attention/MLP matrices to corrected `NorMuon`.
Other parameters use standard PyTorch AdamW, with FP32 master weights/moments for
BF16 embeddings. Matrix updates use stochastic BF16 writes instead of full FP32
matrix masters when both master-weight and stochastic-NorMuon flags are selected.

The current launcher uses Nick's fresh-run base rates below. Existing explicit
group multipliers and decay settings are retained; the value-embedding and branch
rates follow from those multipliers. Earlier rates are recorded in
[the run history](NORMUON_HISTORY.md).

| Group | Peak rate | Decay |
| --- | --- | --- |
| NorMuon matrices | 0.007 | 0.025, coupled cautious decay |
| Adam/tied embedding | 0.0005 | 0.025 where applicable |
| Value embedding | 0.032 (64 × Adam base) | 0.000390625 |
| Residual scalars | 0.0005 | 0 |
| Branch scalars | 0.002 (4 × Adam base) | 0 |

NorMuon momentum ramps from 0.85 on update 1 to 0.95 at update 1,000, then remains
constant. Momentum and LR schedules use the absolute saved update index on resume.
Shape scaling is absent from the active update coefficient; explicit parameter
`lr_mul` and `wd_mul` support remains unchanged.
