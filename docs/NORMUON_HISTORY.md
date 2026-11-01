# NorMuon correction and learning-rate history

## Original-run correction at step 50,330

On 2026-10-27, the production continuation used the saved checkpoint at step
50,330 and removed only the extra shape multiplier from the NorMuon update
coefficient. Corrected updates began at 50,331. The corrected source was copied
directly for this repository; older archived trainer copies were not used.

```python
# Historical coefficient (removed):
eff_lr = lr * math.sqrt(max(1, p.size(0) / p.size(1))) * getattr(p, "lr_mul", 1.0)
# Current coefficient:
eff_lr = lr * getattr(p, "lr_mul", 1.0)
eff_wd = eff_lr * wd * getattr(p, "wd_mul", 1.0)
```

At that correction, the configured base NorMuon LR stayed **0.011**. For
6832-by-2048 gate/up matrices, the effective peak changed from approximately
**0.020091** to **0.011**. The
coupled cautious-decay coefficient follows that effective rate; it was not
converted to Violetto or decoupled decay. Parameter grouping, momentum,
orthogonalization, variance reduction and stochastic-rounding logic are retained.

The runtime calls `NorMuon` directly. Legacy `HybridNorMuonAdamW` and
`partition_params_for_normuon` console summaries in the vendor file still show
historical shape multipliers. Those unused summaries do not describe the active
`NorMuon.step` or this recipe. The vendor file is preserved byte-for-byte from
the original production snapshot, including embedded attribution and historical notes.

## Subsequent fresh run with lower base rates

Nick reported that the previous run stalled and confirmed the replacement is a
**fresh run**, with **Adam base LR 0.0005** and **NorMuon base LR 0.007**. The
previous recipe used Adam **0.0012** and NorMuon **0.011** after the shape fix.
High learning rate is suspected as the cause of the stall; this is an owner report,
not a causal result established by a controlled comparison.

The launch example now uses the new base rates. Other recipe settings, including
explicit value-embedding/branch multipliers, decay, warmup and schedule lengths,
are retained from the original snapshot. This update does not inspect or modify
the new live run, and no new run URL or start date is assumed. Use an empty output
directory for a fresh run; `--resume auto` resumes if checkpoints already exist.

The removed matrix-shape multiplier stays removed. At the new stable peak,
gate/up uses **0.007**, and the coupled cautious-decay coefficient is
`0.007 * 0.025 = 0.000175` when `lr_mul` and `wd_mul` are 1. This follows the same optimizer
formula; no decay, momentum, grouping or rounding behavior was changed.

## Telemetry distinction in the original continuation

The repository's `telemetry.py` reports `lr/normuon_gate_up_effective` directly
from the configured NorMuon learning rate: **0.007** at the new stable peak,
or the actual scheduled rate during warmup/decay.

At the original 50,330 continuation, the process retained the previous logger in
memory and could still report about **0.020091** for that derived chart, even
though its corrected optimizer used **0.011** at the stable peak. That was a
logging discrepancy. A fresh launch from the corrected repository source uses
the corrected logger. No restart was made for repository preparation or logging.

## Regression evidence

`tests/test_normuon_contract.py` executes AST-extracted optimizer/schedule code
with scalar test doubles, without importing the training stack. Eight tests
cover shape independence across production matrix shapes, explicit multipliers,
coupled cautious decay/sign behavior, plain/master/stochastic update routing,
skipped gradients, zero rate, base-rate schedules and future-launch telemetry.
Shape, schedule and telemetry checks include both the historical rates and the
new **0.007 / 0.0005** recipe.

The tests passed against this staged source. In-memory negative checks rejected
reintroducing the shape multiplier and replacing coupled decay with uncoupled
decay. These checks do not validate GPU kernels, actual stochastic rounding or
training quality. No GPU test or training experiment was run for this packaging.

SHA-256 of the unchanged optimizer source from the original production snapshot:
`fc2605a1168ce64a3df53f4a91af74be4402ab60cf3b2076e1568237891d7cd7`.
