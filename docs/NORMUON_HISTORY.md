# NorMuon correction at step 50,330

On 2026-10-27, the production continuation used the saved checkpoint at step
50,330 and removed only the extra shape multiplier from the NorMuon update
coefficient. Corrected updates begin at 50,331. The current source was copied
directly for this repository; older archived trainer copies were not used.

```python
# Historical coefficient (removed):
eff_lr = lr * math.sqrt(max(1, p.size(0) / p.size(1))) * getattr(p, "lr_mul", 1.0)
# Current coefficient:
eff_lr = lr * getattr(p, "lr_mul", 1.0)
eff_wd = eff_lr * wd * getattr(p, "wd_mul", 1.0)
```

The configured base NorMuon LR stays **0.011**. For 6832-by-2048 gate/up matrices,
the effective peak changes from approximately **0.020091** to **0.011**. The
coupled cautious-decay coefficient follows that effective rate; it was not
converted to Violetto or decoupled decay. Parameter grouping, momentum,
orthogonalization, variance reduction and stochastic-rounding logic are retained.

The runtime calls `NorMuon` directly. Legacy `HybridNorMuonAdamW` and
`partition_params_for_normuon` console summaries in the vendor file still show
historical shape multipliers. Those unused summaries do not describe the active
`NorMuon.step` or this recipe. The vendor file is preserved byte-for-byte from
current production, including embedded attribution and historical notes.

## Telemetry distinction

`telemetry.py` on disk now reports `lr/normuon_gate_up_effective` directly from
the configured NorMuon learning rate. At the stable peak that is **0.011**.
The live process that continued from 50,330 already had the previous logger in
memory and can still report about **0.020091** for that derived chart. Its
optimizer update is corrected; the stale chart is a logging discrepancy.
Future normal launches use the corrected on-disk logger. No restart was made
or requested for repository preparation or to refresh logging.

## Regression evidence

`tests/test_normuon_contract.py` executes AST-extracted optimizer/schedule code
with scalar test doubles, without importing the training stack. Eight tests
cover shape independence across production matrix shapes, explicit multipliers,
coupled cautious decay/sign behavior, plain/master/stochastic update routing,
skipped gradients, zero rate, base-rate schedules and future-launch telemetry.

The tests passed against this staged source. In-memory negative checks rejected
reintroducing the shape multiplier and replacing coupled decay with uncoupled
decay. These checks do not validate GPU kernels, actual stochastic rounding or
training quality. No GPU test or training experiment was run for this packaging.

SHA-256 of the unchanged current-production optimizer source:
`fc2605a1168ce64a3df53f4a91af74be4402ab60cf3b2076e1568237891d7cd7`.
