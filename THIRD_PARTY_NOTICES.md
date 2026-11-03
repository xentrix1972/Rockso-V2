# Third-party notices and provenance

Third-party terms are separate from Apache-2.0 on original Rockso additions.
Retained license texts are in [`third_party/licenses`](third_party/licenses).
Reviewed November 3, 2026. Pinned review versions identify the sources examined;
they do not establish an undocumented historical import revision.

## modded-nanogpt / NorMuon

Local file: `rockso_hybrid/vendor/normuon_optimizer_v3.py`.
Copyright (c) 2024 Keller Jordan. **MIT**;
[retained text](third_party/licenses/modded-nanogpt-MIT.txt).

The header explicitly names the `2026-01-18_UnifiedOptimizers` record, introduced
by [commit 71af6208373d6065da641eb3cf9cc0acb1ce6b93](https://github.com/KellerJordan/modded-nanogpt/commit/71af6208373d6065da641eb3cf9cc0acb1ce6b93).
The [reviewed source record](https://github.com/KellerJordan/modded-nanogpt/blob/71af6208373d6065da641eb3cf9cc0acb1ce6b93/records/track_1_short/2026-01-18_UnifiedOptimizers/unified-optimizer/2fc79469-a527-4bde-8540-8426ed3352d1.txt)
contains all 15 matching five-iteration Polar Express coefficients.
The [upstream license](https://github.com/KellerJordan/modded-nanogpt/blob/71af6208373d6065da641eb3cf9cc0acb1ce6b93/LICENSE)
is MIT. The source credits its optimizer, variance reduction and Triton kernel basis.

Rockso modifications include wrapper/partitioning, precision and stochastic
writeback paths, optional spike detection, and removal of the shape-dependent
learning-rate multiplier. This is a modified implementation, not upstream verbatim.
Historical comments and citations are retained.

Research credit: [NorMuon](https://arxiv.org/abs/2510.05491),
[Polar Express](https://arxiv.org/abs/2505.16932) (Noah Amsel, David Persson,
Christopher Musco and Robert M. Gower), and [IMU-1](https://arxiv.org/abs/2602.02522).
The code permission comes from the upstream license, not the papers.

## OLMo-core / optional SkipStepDetector

The detector in the optimizer credits `SkipStepOptimizer` and
`StabilityMonitorCallback`. Credit: The Allen Institute for Artificial Intelligence.
**Apache-2.0**; [retained text](third_party/licenses/OLMo-core-Apache-2.0.txt).

Reviewed commit `ccf4ef4b5f6552e92f933589e9e6377d16fafb40`:
[optimizer](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/optim/skip_step_optimizer.py),
[monitor](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/src/olmo_core/train/callbacks/stability_monitor.py),
[license](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/LICENSE),
[notices](https://github.com/allenai/OLMo-core/blob/ccf4ef4b5f6552e92f933589e9e6377d16fafb40/THIRD_PARTY_NOTICES.md).

Rockso uses a separate tensor-window detector, not OLMo's optimizer subclass or
callback. Both upstream reference files match the archived local OLMo files
byte-for-byte. The historical import commit itself is not recorded. The reviewed upstream third-party entries
concern other modules, not these two reference files. The source also credits the
[OLMo-2 paper](https://arxiv.org/abs/2502.05795).

## EleutherAI lm-evaluation-harness

Copyright (c) 2020 EleutherAI. **MIT**;
[retained text](third_party/licenses/lm-evaluation-harness-MIT.txt).
Reviewed commit `d6de81643928d653435c431bae19945d41d32520`;
[upstream license](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/LICENSE.md).

- `eval_lm_ppl.py::wikitext_detokenizer` matches the executable body of
  [the upstream function](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/wikitext/preprocess_wikitext.py),
  after adapting its input from the document `page` field to a string.
- `rockso_hrm_native_eval.py::GSM8K_COT_EXAMPLES` contains the same eight ordered
  question/answer pairs as
  [`gsm8k-cot.yaml`, task version 3](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/tasks/gsm8k/gsm8k-cot.yaml),
  verified after YAML whitespace folding. Rockso stores Python tuples and adds
  its native prompt wrapper; the examples' contents and order are unchanged.

The upstream harness distribution supplies the MIT notice. This review does not
assign that license to every underlying benchmark dataset or independently
reconstruct the examples' earlier publication history.

## Limite / Violetto architecture reference

`rockso_hybrid/model.py::DepthMixer` explicitly references Violetto's normalized
MUDD router input. Credit: Paradigma's Limite/Violetto project. The reviewed
[official source repository](https://github.com/paradigma-inc/limite-violetto/tree/7ddad34f5e3d25ddea68c6745fc00373b82bca21)
uses [Apache-2.0](https://github.com/paradigma-inc/limite-violetto/blob/7ddad34f5e3d25ddea68c6745fc00373b82bca21/LICENSE)
([retained text](third_party/licenses/Limite-Violetto-Apache-2.0.txt)).
Its [MUDD components](https://github.com/paradigma-inc/limite-violetto/blob/7ddad34f5e3d25ddea68c6745fc00373b82bca21/src/limite_vllm/_components.py)
provide a current comparison reference. The earlier local download receipt
(October 7, 2026) recovers HF review-source revision
`9f4d1567975a44c35127503deadfe52d434d734f`:
[model source](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/9f4d1567975a44c35127503deadfe52d434d734f/modeling_limite.py),
[Apache-2.0 model card](https://huggingface.co/paradigma-inc/limite-1b-violetto/blob/9f4d1567975a44c35127503deadfe52d434d734f/README.md).
The archived model source verifies against Git blob
`afb1dac793c992998a1d9924fc296b41fa924344` in the download metadata.
This recovers the reviewed source identity, not every later local adaptation step.

Rockso's comments identify its explicit RMS epsilon and its own rotary convention
and precision paths. The 32-layer architecture and weights are not Violetto's.
The native model files remain byte-for-byte unchanged for checkpoint source
verification, including their explanatory comments. This acknowledgement is not
a claim of identical implementations or performance.

## Sapient Intelligence / HRM-Text and separate assets

Credit: [Sapient Intelligence's HRM-Text](https://github.com/sapientinc/HRM-Text)
and [sampled RSI dataset/tokenizer](https://huggingface.co/datasets/sapientinc/hrm-text-rsi-sampled-20260923).
The linked Rockso weights release identifies that tokenizer as its upstream source.
Neither tokenizer nor corpus is bundled in Git. The reviewed dataset card has no
explicit dataset/tokenizer license. The separate HRM-Text code repository's
Apache license is not assumed to license every data/tokenizer artifact. Confirm
those terms before assigning a weights or tokenizer reuse license.

PyTorch, TorchAO, Triton, Liger, Transformers, Datasets and other packages are
external dependencies. Full benchmark assets are external except for the eight
protocol fixtures above. Existing BBH/HumanEval/MBPP asset revision, checksum and
license handling is retained. Downloading assets does not change their terms.
No upstream project endorses this release.
