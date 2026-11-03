# Load the public weights-only snapshot

The [public model snapshot](https://huggingface.co/drrockso72/rockso1.85b_checkpoints/tree/e40c9c6070092dbbcbdbcfa3997ead331b351496)
is a native `rockso_hybrid_export_1` file, not a Transformers AutoModel model or a
resumable training checkpoint. The separate `infer_export.py` command loads it
on CPU with the unchanged native architecture and explicit PrefixLM masking.

## Install and download

From the repository root, use a separate Linux/WSL2 Python 3.10 environment:

```bash
python3.10 -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
hf download drrockso72/rockso1.85b_checkpoints \
  --revision e40c9c6070092dbbcbdbcfa3997ead331b351496 \
  --local-dir ./weights \
  --include model.pt tokenizer.json tokenizer_config.json special_tokens_map.json tokenizer_source_config.json
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 python infer_export.py \
  --checkpoint ./weights/model.pt --tokenizer ./weights \
  --condition direct --prompt 'What is 2 + 2?' --max-new-tokens 8
```

The command verifies the pinned model SHA-256 before loading. The file is
3,694,628,377 bytes; SHA-256:

```text
1b228a0fbd4ab04a5a2c598ed2cb35d1b8c678550d69d185cb2c195c746e3869
```

Expect several GB of host memory use in addition to the 3.69 GB weight file.
This simple reference recomputes the sequence for each token, so CPU generation
can be slow. It defaults to two CPU threads and 32 new tokens. It does not use a
GPU, FP8, compilation, a KV cache or tool execution. It is a single-turn greedy
reference, not a replacement for the full chat/evaluation applications.

## Native contract

- `direct`: `<|im_start|><|object_ref_start|>PROMPT<|im_end|>`.
- `cot`: `<|im_start|><|object_ref_end|>PROMPT<|im_end|>`.
- `synth,cot`: `<|im_start|><|quad_end|><|object_ref_end|>PROMPT<|im_end|>`.

For this exact tokenizer, these begin with `[2,4]`, `[2,5]` and `[2,9,5]`.
Question end is ID 3. Generation starts immediately afterward and ends on
`<|box_end|>` (ID 7). The tokenizer's EOS is ID 1; it is not the answer-end marker.
The entire initial prompt, including question end, is bidirectional. That boundary
stays fixed: generated answer tokens are causal and may attend to the full prefix.
No generic Qwen chat wrapper, extra BOS, or `<|box_start|>` is inserted.

The 2,048-token context includes prefix and answer. The command rejects requests
that cannot fit rather than silently truncating or changing the attention mode.
Use `--system '...'` only when that text belongs in the instruction; it is joined
to the user instruction with two newlines, matching the native wrapper.
Other historical condition combinations are not claimed to be current V2 lanes.

The loader checks the four native inference source hashes, exact tokenizer-file
bytes, special IDs, strict state-dict keys/shapes, all tensor values/dtypes and
both shared parameter groups. It materializes native nonpersistent RoPE buffers.
The Hub root's `tokenizer_source_config.json` is checked as the original tokenizer
`config.json`; the original four-file tokenizer directory also works. Extra Hub
metadata files do not become tokenizer identity inputs.

## Identity and validation limits

The saved export has 565 state-dict tensors and 1,846,878,764 unique parameters.
The companion full checkpoint and capture receipt identify **step 12,168** and
**3,987,210,240 input tokens** of the fresh run. These counters are not fields in
the weights-only export. See the pinned Hub
[provenance](https://huggingface.co/drrockso72/rockso1.85b_checkpoints/blob/e40c9c6070092dbbcbdbcfa3997ead331b351496/model.provenance.json).
The export cannot resume optimizer, RNG or data-cursor state.

This release preparation tests short CPU generation and the loader/masking
contracts. See [validation evidence](RELEASE_VALIDATION.md) for exact versions,
commands and outcomes. No GPU numerical parity, broad quality benchmark, training
reproduction or support for AutoModel/vLLM/llama.cpp is asserted.

The CLI uses `torch.load(..., weights_only=True, mmap=True, map_location='cpu')`.
It never falls back to unrestricted pickle loading. Hash checks establish the
selected file's identity, not a general safety guarantee for arbitrary files.
For a different trusted export, specify its independently verified
`--expected-sha256`; architecture/source/tokenizer checks still apply.

Public download does not supply a weights/tokenizer/data license. See
[LICENSING.md](../LICENSING.md) for the scope of the approved original-code license.
