# Setup and environment

For the public weights-only export, start with [CPU inference](INFERENCE.md).
It has a separate small dependency manifest and does not require the training
stack below. The recorded training environment is not a CPU inference lockfile.

Use Linux/WSL2 with Python 3.10. The production recipe uses a CUDA GPU supporting
the retained BF16/FP8 paths, PyTorch Flex attention, TorchAO FP8, Triton and Liger.
Hardware capacity and kernel compatibility must be qualified separately. This
repository preparation did not allocate GPU memory or test a new environment.

Create a separate environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-common.txt
```

Install the GPU stack separately from trusted upstream distributions/builds using
versions compatible with your driver and hardware. Do not assume a current stable
wheel recreates these development builds. The observed reference environment was:

| Component | Reference version |
| --- | --- |
| Python | 3.10.12 |
| torch | 2.12.0.dev20260316+cu130 |
| torchao | 0.17.0+git637c4ac12 |
| triton | 3.6.0+git9844da95 |
| liger-kernel | 0.7.0 |
| transformers | 4.57.1 |
| datasets | 4.4.1 |
| numpy | 1.26.4 |
| pyarrow | 23.0.1 |
| safetensors | 0.7.0 |
| tqdm | 4.67.3 |
| wandb | 0.23.1 |

Versions were read from package metadata without importing the ML stack. The
reference build artifacts are not bundled, so this is an environment record, not
a complete lockfile. Native evaluation additionally needs SymPy; DROP needs a
compatible `lm-eval` installation. The vendored optimizer contains optional
bitsandbytes fallbacks, but the selected V2 runtime uses standard PyTorch AdamW.

Packed training data must be a Hugging Face `load_from_disk` dataset (or a
DatasetDict containing `train`) with 2048-token `input_ids`, binary
`metadata_mask` (1 = prefix/meta/padding; 0 = response), and increasing positive
`span_ends` document boundaries. Optional `position_ids` must be consistent across
the batch. Supply a one-dimensional integer `order.npy` and its ledger JSON.
The tokenizer must have exactly 65,536 tokens; pad must differ from EOS.

The trainer validates these contracts, records source and data metadata, and
uses consumed ledger rows rather than worker read-ahead for continuation. The
original private data preparation pipeline and corpus are not part of this copy.

Only trusted native checkpoints should be used: they contain Python/NumPy RNG
and optimizer objects and are loaded with `weights_only=False`. Do not point the
examples at arbitrary downloaded checkpoint files. The separate
[`infer_export.py`](../infer_export.py) loader supports the pinned Hugging Face
weights-only export using `weights_only=True`; see [its exact contract](INFERENCE.md).

The source-only checks need no installation of `requirements-common.txt`:

```bash
python3 -B tests/test_normuon_contract.py
python3 -B scripts/check_source.py
```
