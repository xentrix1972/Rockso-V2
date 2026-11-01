# Rockso V2: Single-RTX-5090 Language Model Training

This trainer is optimized to train language models on a **single NVIDIA RTX 5090
(32 GB)**. **Rockso V2** is the model being trained with it. The repository includes
the trainer, runtime, chat and evaluation tools, plus the explicit training recipe.

The recipe combines several precision, memory and kernel choices:

- **Mixed precision:** BF16 model storage, TorchAO FP8 execution for large
  attention and MLP projections, and FP32 numerical paths for norms, gates, RoPE
  and learned scalars. TF32 is permitted for FP32 matrix multiplication.
- **Memory management:** activation checkpointing except for the last two layers,
  gradient accumulation, and stochastic BF16 NorMuon matrix updates without full
  FP32 matrix master copies. Embedding optimizer weights and moments remain FP32.
- **Compiled and fused execution:** `torch.compile` on transformer blocks,
  compiled FlexAttention for the packed PrefixLM masks, and Liger fused linear
  cross-entropy for the response-only loss.
- **Matrix alignment:** a Triton padding kernel expands the relevant FP8 MLP
  contractions from width 6832 to 6912 for execution, while retaining the logical
  parameter shapes, quantization scales and optimizer matrices.

See the [architecture and numerical paths](docs/ARCHITECTURE.md),
[reference software stack](docs/SETUP.md) and
[launch recipe](scripts/launch_rsi_32.sh) for the exact choices. Model weights and
training data are separate. This private source copy was prepared without changing
the live training run; no new GPU performance measurements were made.

**NorMuon is the corrected version without the shape-based learning-rate
multiplier.** At production checkpoint **50,330**, the update coefficient changed
from `lr * sqrt(max(1, rows / cols)) * lr_mul` to `lr * lr_mul`.
That correction kept the then-configured base rate at **0.011**, reducing the
6832-by-2048 gate/up coefficient from about **0.020091** to **0.011**.
Coupled cautious weight decay is preserved.

**Current fresh-run recipe: Adam base LR 0.0005; NorMuon base LR 0.007.**
Nick confirmed these lower rates are for a fresh run after the previous run
stalled. Excessive learning rate is the suspected cause, not an established
diagnosis. The matrix-shape multiplier remains removed, so the gate/up peak
coefficient is **0.007**. The launcher retains the other previously documented
settings. See [the correction and run history](docs/NORMUON_HISTORY.md).

Original-run charts: [Rockso V2 on W&B](https://wandb.ai/xentrix1972-nick-bochy/Rockso_v2/runs/25bc0bfca35b).
A Hugging Face model location has not yet been verified; no weights URL is assumed.

## Start here

1. Use Linux or WSL2 and a separate Python environment. Read
   [setup and environment requirements](docs/SETUP.md).
2. Supply your packed data, ledger and exact tokenizer. They are not in Git.
3. Review `configs/rockso_v2_32.json` and `scripts/launch_rsi_32.sh` before a new run.
4. Run the lightweight checks below before making changes.

```bash
python3 -B tests/test_normuon_contract.py
python3 -B scripts/check_source.py
bash -n scripts/launch_rsi_32.sh
```

These checks use only the Python standard library and shell syntax validation.
They do not load PyTorch, a model, a checkpoint or the GPU.

## Training example

Only launch on a machine available for training, after provisioning and qualifying
the GPU environment. This example describes the retained recipe; it does not
authorize interrupting an existing run.

```bash
export ROCKSO_DATA=/absolute/path/to/packed_dataset
export ROCKSO_TOKENIZER=/absolute/path/to/tokenizer
export ROCKSO_OUT=./runs/my_run
bash scripts/launch_rsi_32.sh
```

The launcher uses a Python executable from `ROCKSO_PYTHON` (default `python3`).
Ledger defaults are `$ROCKSO_DATA/ledger/order.npy` and
`$ROCKSO_DATA/ledger/ledger.json`; override with `ROCKSO_LEDGER` and
`ROCKSO_LEDGER_META`. `--resume auto` starts fresh in an empty output or resumes
the highest saved native checkpoint in that output. Choose a fresh output for
a new lineage, including the lower-rate fresh run described above. Resume restores
optimizer/RNG/cursor state and validates the
architecture, tokenizer and data contract.

W&B publishing is disabled in this portable example. To enable it using your own
existing login, pass `--wandb-mode online --wandb-entity YOUR_ENTITY
--wandb-project YOUR_PROJECT`. Never add credentials to source or Git.

The recipe uses batch 20, accumulation 8 and sequence 2048: **327,680 packed slots
per update**. Seed 42; 1,000-update warmup; stable through 141,193; linear decay
through 176,491. That horizon came from the original four-epoch ledger and must be
reviewed for different data. `--keep-last 2` controls activation retention, not
checkpoint deletion. Runtime and optimizer settings are explicit in the launcher;
bare trainer defaults are not this recipe.

## Architecture and evaluation

The model has 32 layers, width 2048, 32 query / 8 key-value heads, tied 65,536-token
embeddings and SwiGLU width 6832. It combines PrefixLM attention, half RoPE,
attention gates, value embeddings, XSA, sparse MUDD and learned residual/branch
scalars. [Architecture details](docs/ARCHITECTURE.md) explain the numerical paths.

[Evaluation instructions](docs/EVALUATION.md) cover PPL, native task evaluation,
HumanEval/MBPP assets and sandboxing, and exact condition/template conventions.
For interactive chat, with the model's matching native checkpoint and tokenizer:

```bash
python rockso_chat_v5_prefixlm_145m.py \
  --checkpoint /absolute/path/to/native_checkpoint.pt \
  --tokenizer /absolute/path/to/tokenizer --condition direct
```

The historical `145m` script filename does not describe this V2 model's size.
No model inference or GPU tests were run while packaging this repository.

## What is reproducible

This repository preserves the training recipe and evaluation code. Full training
reproduction also requires the private packed corpus, exact ordering ledger,
tokenizer, environment builds and any continuation checkpoint. None is bundled.
Evaluation requires matching weights/tokenizer plus the chosen benchmark assets.
The private RSI tail probe measures training-mixture exposure; it is not a
permanent document-disjoint validation set.

## Contents and licensing

`rockso_hybrid/` contains model/runtime/optimizer code; `train.py` and telemetry
modules run training and monitoring. Root evaluation/chat files are retained with
portable inputs. `docs/SOURCE_SNAPSHOT.json` records source hashes and packaging
changes; `TRACKED_FILES.txt` is the explicit publication allowlist.

Weights, checkpoints, optimizer/RNG state, full datasets, caches, binary diagnostics,
private conversations, handoffs, credentials and original Git history are excluded.
The native evaluator retains eight attributed GSM8K prompt examples as protocol
fixtures; their license/provenance review is recorded in the notices below.

**No license grant is made for Nick's original code.** The repository is intended
to remain private. See [LICENSING.md](LICENSING.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before any public release.
