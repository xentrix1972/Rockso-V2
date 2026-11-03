# Evaluation and chat conditions

**Public weights-only download:** use the separate [CPU inference guide](INFERENCE.md).
It verifies and loads `rockso_hybrid_export_1` without optimizer/RNG state. The
commands below describe the existing full-checkpoint evaluation workflow; do not
substitute a public export where a full native checkpoint is required. Export
generation checks are not benchmark results.

Run these commands from the repository root in the documented Python environment,
on a machine available for evaluation. They load model weights and can use the GPU;
they must not be run alongside the protected live training session.

The repository contains code, not checkpoints, tokenizers, benchmark caches or the
private training corpus. Supply a trusted native `rockso_hybrid_checkpoint_1`
checkpoint and its exact 65,536-token tokenizer. The checkpoint loader uses
PyTorch's full checkpoint format (`weights_only=False`). A Hugging Face export is
not interchangeable with a native training checkpoint in these commands.

```bash
export CHECKPOINT=/absolute/path/to/step000xxxxx.pt
export TOKENIZER=/absolute/path/to/tokenizer
mkdir -p evals
```

The model implementation hashes embedded in a checkpoint must match the packaged
model/configuration/masking/loss/residual source. The PPL adapter permits a tokenizer
directory to move, while still checking all file hashes, vocabulary and special-token
metadata. Do not bypass these checks to evaluate a different architecture or tokenizer.

## Runtime requirements

Use the environment described in the main README. Evaluation uses PyTorch,
Transformers, Datasets, NumPy, PyArrow and tqdm; the imported native runtime also
requires its listed TorchAO/Triton dependencies even when evaluation FP8 is off.
The native task harness additionally uses SymPy. DROP's reference scorer requires
`lm-eval` (`lm_eval.tasks.drop.utils`); install a compatible version if using DROP.
Package versions and benchmark revisions are part of a comparable evaluation.

Native generation uses the V2 cached decoder with PrefixLM prefill and causal
decoding. Optional `--cuda-graph-decode` performs the retained startup parity check;
it can fall back when qualification fails. Likelihood-based multiple-choice scoring
uses its reference path. This preparation tested the separate CPU export loader;
no GPU evaluation was performed. See [the validation scope](RELEASE_VALIDATION.md).

## Perplexity

An external-corpus command that does not require the private corpus:

```bash
python eval_prefixlm_ppl.py "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" --preset wikitext2 \
  --mode concat --max-length 2048 --stride 512 \
  --attn-mask-mode prefixlm --prefixlm-impl sdpa_bool \
  --device cuda --bf16 --out-json evals/wikitext2_stepxxxxx.json
```

This preset uses `EleutherAI/wikitext_document_level`, `wikitext-2-raw-v1`, test
split and the `page` field with the retained WikiText preprocessing. It downloads
through Datasets into `.cache/huggingface/datasets` unless `--cache-dir` overrides it.
The preset does not pin a dataset revision: record the resolved dataset/cache
identity if you need an exact rerun. Other retained presets include `wikitext103`
and `ptb`; inspect `eval_lm_ppl.py` for their dataset choices.

Without a `--meco-tag`, plain corpus text has no instruction prefix and this scorer
uses the equivalent causal path. Keep context length, stride, concatenation mode,
preprocessing and tokenizer fixed when comparing checkpoints.

To score the separately supplied frozen RSI tail probe:

```bash
export PROBE=/absolute/path/to/rsi_tail1024_v1
python eval_prefixlm_ppl.py "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" --preset hrm \
  --hrm-data-dir "$PROBE" --hrm-ledger "$PROBE/ledger.json" \
  --hrm-num-blocks 1024 --attn-mask-mode prefixlm \
  --prefixlm-impl sdpa_bool --device cuda --bf16 \
  --out-json evals/rsi_tail_stepxxxxx.json
```

The probe and its provenance ledger are excluded from Git. The original probe
uses the final 1,024 packed blocks, original token IDs and native conditions.
It scores complete documents separately, excludes prompt and padding targets,
includes the EOA token, and aggregates by response-token count before exponentiating.
Reports include token/document denominators and per-condition results.

This is a **training-mixture tail probe**, not a permanent document-disjoint held-out
set. Its positions remain in the training ledger; repeated earlier source views
can overlap. Record the checkpoint cursor and probe exposure. Full training and
private-probe reproduction require the missing private inputs.

## Native task evaluation

An explicit GSM8K example:

```bash
python rockso_hrm_native_eval_prefixlm_codex.py \
  --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" \
  --implicit ./Implicit_curriculum_math_dense_v8_prefixlm_codex.py \
  --task gsm8k --condition synth,cot --num-shots 0 \
  --device cuda --dtype bf16 --rope-impl rotate_half \
  --batch-size 1 --max-context 2048 --max-new-tokens 1280 \
  --limit 500 --keep-items --json evals/gsm8k_stepxxxxx.json
```

`--limit 0` selects the full task. Use a distinct output filename for each evaluation;
the wrapper retains generation streams and will not silently overwrite an existing
stream. Conditions for other tasks come from the native harness unless explicitly
overridden. For MMLU zero-shot, add `--no-special-shots` to suppress inherited
subject-specific shot overrides. GPQA requires separately authorized local assets;
set `ROCKSO_GPQA_ROOT` to their directory. BBH retains the pinned official revision
`9ee07bd481feebf959a6b59d61ea57bdcf30964d` and its asset/license download checks.

Historical evaluation W&B adapters for an older run are deliberately not packaged.
`--wandb_project` / `--wandb_entity` fail before evaluation; use local JSON.
This does not change the trainer's separate W&B monitoring support.

## HumanEval and MBPP

These use the original test sets through the native HRM instruction wrapper.
They are not claims of standard leaderboard protocol parity, HumanEval+/MBPP+,
or contamination-free training. Keep checkpoint, tokenizer, prompt format, shots,
condition, decoder, context budget, extraction policy and execution runtime fixed.

Public assets are not bundled. Obtain these exact pinned files, with their upstream
licenses, and preserve the directory structure:

| Task | Hugging Face dataset | Revision | Subdirectory under `ROCKSO_EVAL_ASSETS` |
| --- | --- | --- | --- |
| MBPP | `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | `mbpp_4bb6404fdc6c` |
| HumanEval | `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | `humaneval_7dce6050a7d6` |

MBPP needs `full/test-00000-of-00001.parquet` and
`full/prompt-00000-of-00001.parquet`. HumanEval needs
`openai_humaneval/test-00000-of-00001.parquet`. The original SHA-256 checks remain
in `rockso_code_eval.py`; a mismatching asset is rejected. Full test counts are
500 for MBPP and 164 for HumanEval.

Generated code is executed only through the retained **bubblewrap sandbox**.
Before code benchmarks can run, separately provision Linux bubblewrap and a
Python 3 dependency directory compatible with `/usr/bin/python3`. The directory
must include the benchmark's required packages (including NumPy for its self-test).
This repository does not contain or provision that runtime, so code benchmarks are
unavailable until it is supplied and the retained sandbox self-test passes.

```bash
export ROCKSO_EVAL_ASSETS=/absolute/path/to/benchmark_assets
export ROCKSO_CODE_BWRAP=/usr/bin/bwrap
export ROCKSO_CODE_RUNTIME=/absolute/path/to/python_dependency_directory
```

The sandbox keeps the original isolated namespaces, read-only `/usr` and dependency
mounts, cleared environment, hidden host files/GPU, one-GiB address-space limit,
three-second execution limit and seven-second wall timeout. It fails if its binary
or dependency directory is missing. Do not replace this with unsandboxed execution.

After provisioning those inputs, the zero-shot HumanEval command is:

```bash
python rockso_hrm_native_eval_prefixlm_codex.py \
  --checkpoint "$CHECKPOINT" --tokenizer "$TOKENIZER" \
  --implicit ./Implicit_curriculum_math_dense_v8_prefixlm_codex.py \
  --task humaneval --condition synth,cot --num-shots 0 \
  --code-prompt-mode instruct --code-prompt-format approach_first \
  --code-output-mode fenced --device cuda --dtype bf16 \
  --rope-impl rotate_half --cuda-graph-decode --batch-size 1 \
  --max-context 2048 --max-new-tokens 512 --limit 0 --keep-items \
  --json evals/humaneval_stepxxxxx_approach_first_fenced.json
```

For MBPP change `--task humaneval` to `--task mbpp` and choose a new JSON filename.
Keep `--num-shots 0`. The historical MBPP example-based protocol is available with
`--code-prompt-format original --num-shots 3`; do not mix its scores with zero-shot.

The `approach_first` format requests at most three plain-text sentences explaining
the approach, then the complete implementation/imports in one Python code block.
HumanEval keeps the public description and exact function scaffold. MBPP supplies
the published description and public `test_list`, without solution examples.
Hidden tests/canonical solutions are not placed in these prompts.

The `fenced` output policy accepts whole plain Python or one complete Python fence,
allows prose outside the fence, and stops at the closing fence. It does not repair
code or rename functions. `--code-output-mode legacy` restores historical extraction.
Raw generations, stop reasons, coverage and context-limit counts are retained.

## Condition and template contract

Native prompt wrapping is:

```text
<|im_start|>{condition markers}{question}<|im_end|>{response}<|box_end|>
```

| Meaning | Marker |
| --- | --- |
| Direct | `<|object_ref_start|>` |
| Chain-of-thought condition | `<|object_ref_end|>` |
| Noisy condition | `<|quad_start|>` |
| Synthetic condition | `<|quad_end|>` |
| End of answer (EOA) | `<|box_end|>` (ID 7 in this tokenizer) |

`synth,cot` concatenates the synthetic then CoT markers. Code and math examples use
that condition. Ordinary Direct chat uses `direct`; tool-call prompting also uses
Direct. These markers select the trained condition; they do not change token budget.
Use the exact tokenizer associated with the checkpoint rather than assuming these
IDs hold for another tokenizer.

BFCL's external adapter, frozen manifests and supporting tooling are outside this
curated package. No BFCL runner or score is claimed. The included chat tool format
only renders tool-call text; it does not execute external functions.

## Packaging boundaries

Model math, scoring formulas, prompt builders, benchmark revisions/checksums and
the generated-code isolation protocol are retained. Changes are limited to portable
paths, explicit tokenizer inputs, relocation-safe tokenizer identity checking,
removing unsupported legacy model loaders and historical evaluation W&B hooks, and
selecting the dataset/report helper functions needed by PPL. `eval_lm_ppl.py` is a
helper module; use `eval_prefixlm_ppl.py` as the PPL entrypoint.

All source attribution comments are retained. Dataset, harness and upstream code
licenses need their own review before redistribution/public release; no new license
grant is implied here.
