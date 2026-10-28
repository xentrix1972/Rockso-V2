# Retained attribution and provenance

The vendored `rockso_hybrid/vendor/normuon_optimizer_v3.py` header attributes its
development to Keller Jordan's `modded-nanogpt` unified optimizer and names the
`2026-01-18_UnifiedOptimizers` change. The header and internal attribution are
preserved. Exact upstream commit and code-license correspondence still require
review; this file does not assign an upstream license.

References retained in that source:

- NorMuon: https://arxiv.org/abs/2510.05491
- Polar Express: https://arxiv.org/abs/2505.16932
- IMU-1: https://arxiv.org/abs/2602.02522
- Keller Jordan modded-nanogpt: https://github.com/KellerJordan/modded-nanogpt

The optimizer's optional `SkipStepDetector` also attributes its design to
OLMo-core `SkipStepOptimizer` and `StabilityMonitorCallback`, with OLMo-2 paper
context at https://arxiv.org/abs/2502.05795. This attribution remains in source;
the exact upstream revision and applicable code-license correspondence are not
yet verified.

`eval_lm_ppl.py` attributes WikiText preprocessing to lm-eval-harness's
`wikitext/preprocess_wikitext.py`. The native harness includes eight fixed GSM8K
CoT question/answer prompt examples attributed to EleutherAI
lm-evaluation-harness's `lm_eval/tasks/gsm8k/gsm8k-cot.yaml` (`gsm8k_cot` v3).
These embedded protocol fixtures are retained to avoid silently changing the
evaluation. Their exact upstream revisions, code/content licenses and required
notices must be verified before public redistribution.

Rockso model/runtime and evaluation adapters were copied from the current local
production source. Native task/code/BBH support came from the accompanying local
evaluation utilities. `docs/SOURCE_SNAPSHOT.json` records content hashes and
packaging transformations; it does not include private original absolute paths.
All copied attribution comments are preserved. Historical handoffs and private
conversations are intentionally excluded.

PyTorch, TorchAO, Triton, Liger, Transformers, Datasets and other installed
dependencies are external dependencies, not vendored distributions. Full benchmark
datasets are external; the eight embedded GSM8K prompt fixtures are the exception
described above. The BBH helper retains its upstream LICENSE download
and integrity checks; HumanEval/MBPP revision and checksum checks are retained.
Consult their own licenses before downloading, using or redistributing assets.
