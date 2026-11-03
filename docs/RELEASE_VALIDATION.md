# Release preparation validation — November 3, 2026

Base GitHub `main`: `5c405eba5e60cf879eb95766566b235b4229ea81`.
The release changes add documentation, license/notice files and a separate CPU
inference command. Existing training/model/runtime/optimizer sources and the
fresh-run launcher/configuration remain unchanged. The private live training
directory and Hugging Face repository were not modified.

## Clean CPU environment

A new isolated Linux/WSL2 environment was created with Python 3.10.12,
`torch==2.10.0+cpu` from the official PyTorch CPU index, Transformers 4.57.1,
Tokenizers 0.22.1 and the pinned `requirements-inference.txt` dependencies.
The real frozen export was loaded with `weights_only=True`, `mmap=True`,
`map_location='cpu'`; CPU execution used two threads. CUDA remained uninitialized.

The [machine-readable receipt](CPU_RELEASE_VALIDATION.json) records:

- Verified model SHA-256; strict loading of all 565 state-dict tensors.
- 1,846,878,764 unique parameters; all stored tensor values and dtypes preserved.
- Tied embedding/head and shared MUDD reduction parameter identity preserved.
- Original tokenizer directory and the published Hub naming layout accepted
  after byte/ID validation; a deliberately changed tokenizer file rejected.
- Greedy `direct` responses: `What is 2 + 2?` → `4` (IDs `[46,7]`),
  `What is the capital of France?` → `Paris` (IDs `[47129,7]`). Both stopped on
  the native answer-end marker.
- An eight-token `synth,cot` execution check produced
  `The sum of two numbers is calculated by` and hit `max_new_tokens`.
  It is an incomplete response, not a solved reasoning example.

The published-layout contract test uses copied, hash-verified tokenizer files
with the documented `config.json` rename. A final CLI test also freshly downloaded
all four tokenizer files from the pinned public Hub revision and returned `4`,
stopping on ID 7. Neither check redownloaded the 3.69 GB checkpoint.
The checkpoint is the frozen local byte-identical export identified by the
pinned Hub provenance. An earlier check also returned `4` using the existing
PyTorch 2.10.0.dev20251207+cu130 environment on CPU.

## Contract and static checks

```bash
CUDA_VISIBLE_DEVICES= python -B tests/test_infer_export.py
python3 -B tests/test_normuon_contract.py
python3 -B tests/test_eval_formats.py
python3 -B scripts/check_source.py
bash -n scripts/launch_rsi_32.sh
```

The five new CPU tests cover alias/dtype/value preservation; rejection of wrong
schema, mode, source hashes, missing tensors and conflicting saved aliases;
bidirectional prefix/causal answer masks; condition order/envelopes; answer-stop
and context limits. The existing optimizer/evaluation contract tests, source
allowlist/parse checks and launcher syntax check also pass.

Provenance checks matched the WikiText function body after input adaptation,
all eight ordered GSM8K question/answer pairs after YAML whitespace folding, and
all 15 Polar Express coefficients in the named upstream record. Archived OLMo
reference files match the reviewed upstream files; the archived Violetto source
matches its recorded Git blob identity. See [third-party notices](../THIRD_PARTY_NOTICES.md).

## Bounded source/history screening

Reviewed all 49 original tracked text files and all 57 unique blob versions across
the five commits reachable from `main`. The Git refs endpoint returned only
`refs/heads/main` before this preparation branch. No high-signal credential-pattern
matches or credential/checkpoint/data artifact filenames were found. Existing
author email metadata and the intentional W&B link remain part of the history.

Rules covered GitHub/Hugging Face/Slack token patterns, AWS access-key patterns,
private-key headers, credential-bearing URLs, long assigned secret literals and
signed URLs. Release additions receive the same pattern checks. This is not a
complete security audit: it excludes unreachable objects, special PR refs,
releases, Actions artifacts, settings, collaborators and external services, and
cannot detect every possible secret or private detail. No history was rewritten.

No GPU numerical parity, training execution, optimizer update, end-to-end
benchmark suite or performance measurement was performed. The source archive
and model-only export do not reproduce the private training corpus or full run.
