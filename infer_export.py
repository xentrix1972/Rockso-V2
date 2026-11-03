#!/usr/bin/env python3
"""CPU reference generation for Rockso's native weights-only export.

Copyright 2026 xentrix1972 and Rockso contributors.
Licensed under Apache-2.0; see LICENSE and THIRD_PARTY_NOTICES.md.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parent
RELEASE_SHA256 = '1b228a0fbd4ab04a5a2c598ed2cb35d1b8c678550d69d185cb2c195c746e3869'
SOURCE_FILES = ('rockso_hybrid/model.py', 'rockso_hybrid/config.py',
                'rockso_hybrid/masking.py', 'rockso_hybrid/residual.py')
TOKENS = {'boq': '<|im_start|>', 'eoq': '<|im_end|>', 'eoa': '<|box_end|>',
          'direct': '<|object_ref_start|>', 'cot': '<|object_ref_end|>',
          'synth': '<|quad_end|>'}
TOKEN_IDS = {'boq': 2, 'eoq': 3, 'eoa': 7, 'direct': 4, 'cot': 5, 'synth': 9}
CONDITIONS = {'direct': ('direct',), 'cot': ('cot',), 'synth,cot': ('synth', 'cot')}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sources(state):
    for name in SOURCE_FILES:
        if sha256(ROOT / name) != state['source_sha256'].get(name):
            raise ValueError('Inference source differs from export: ' + name)


def bind_model(state):
    """Strict load with stored precision, tied parameters, and native RoPE buffers."""
    import torch
    from rockso_hybrid import ModelConfig, RocksoHybrid
    if state.get('schema') != 'rockso_hybrid_export_1':
        raise ValueError('Expected rockso_hybrid_export_1; full resume checkpoints use other tools')
    if state.get('attention_mode') != 'prefixlm':
        raise ValueError('This reference command requires a PrefixLM export')
    validate_sources(state)
    config = ModelConfig.from_dict(state['config'])
    with torch.device('meta'):
        model = RocksoHybrid(config)
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    aliases = [names for names in aliases.values() if len(names) > 1]
    for names in aliases:
        for name in names[1:]:
            if not torch.equal(state['model'][names[0]], state['model'][name]):
                raise ValueError('Saved shared parameters disagree: ' + name)
    model.load_state_dict(state['model'], strict=True, assign=True)
    for names in aliases:
        parent, _, leaf = names[0].rpartition('.')
        shared = getattr(model.get_submodule(parent), leaf)
        for name in names[1:]:
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, shared)
    model.preserve_precision()  # Native FP32 islands and nonpersistent RoPE frequencies.
    model.configure_runtime(backend='sdpa', checkpointing=False, compile_blocks=False,
                            native_gqa=False)
    model.eval()
    for name, actual in model.state_dict().items():
        expected = state['model'][name]
        if actual.dtype != expected.dtype or not torch.equal(actual, expected):
            raise ValueError('Native load changed stored tensor values/dtype: ' + name)
    if any(t.device.type != 'cpu' for t in list(model.parameters()) + list(model.buffers())):
        raise ValueError('This command only supports CPU tensors and buffers')
    return model, aliases


def load_tokenizer(directory, receipt):
    from transformers import PreTrainedTokenizerFast
    directory = Path(directory)
    expected_files = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'}
    if set(receipt['files']) != expected_files:
        raise ValueError('Unsupported tokenizer receipt file set')
    for name, digest in receipt['files'].items():
        # The Hub root uses config.json for the model. Its original tokenizer
        # config is published under this explicit alternate name.
        file = directory / name
        if name == 'config.json' and (directory / 'tokenizer_source_config.json').is_file():
            file = directory / 'tokenizer_source_config.json'
        if not file.is_file() or sha256(file) != digest:
            raise ValueError('Tokenizer content differs: ' + name)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(directory), local_files_only=True)
    if (len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id) != (
            receipt['vocab_size'], receipt['pad_id'], receipt['eos_id']):
        raise ValueError('Tokenizer vocabulary or special IDs differ')
    for name, text in TOKENS.items():
        if tokenizer.encode(text, add_special_tokens=False) != [TOKEN_IDS[name]]:
            raise ValueError('Native control token mismatch: ' + name)
    return tokenizer


def prompt_ids(tokenizer, prompt, condition='direct', system=None):
    if condition not in CONDITIONS:
        raise ValueError('Use a documented current V2 condition: ' + ', '.join(CONDITIONS))
    instruction = prompt.strip() if not system else system.strip() + '\n\n' + prompt.strip()
    if not instruction:
        raise ValueError('Prompt must not be empty')
    text = TOKENS['boq'] + ''.join(TOKENS[c] for c in CONDITIONS[condition])
    return tokenizer.encode(text + instruction + TOKENS['eoq'], add_special_tokens=False)


def generation_masks(sequence, prefix_length, local_window=0):
    import torch
    from rockso_hybrid.masking import build_masks
    metadata = torch.zeros_like(sequence)
    metadata[:, :prefix_length] = 1
    ends = torch.full((sequence.shape[0], 1), sequence.shape[1], dtype=torch.int32)
    return build_masks(ends, metadata, mode='prefixlm', backend='sdpa', local_window=local_window)


def generate(model, tokenizer, ids, max_new_tokens):
    import torch
    if max_new_tokens < 1 or not ids or len(ids) + max_new_tokens > model.config.max_seq_len:
        raise ValueError('Prompt plus requested answer must fit the configured total context')
    prefix_length = len(ids)  # Frozen through every decode step, including the question end.
    sequence = torch.tensor([ids], dtype=torch.long)
    generated = []
    stop_reason = 'max_new_tokens'
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            masks = generation_masks(sequence, prefix_length, model.config.local_window)
            logits = model(sequence, masks=masks)[:, -1]
            if not torch.isfinite(logits).all():
                raise ValueError('Non-finite next-token logits')
            token = int(logits.argmax(-1).item())
            generated.append(token)
            if token == TOKEN_IDS['eoa']:
                stop_reason = 'answer_end'
                break
            sequence = torch.cat((sequence, torch.tensor([[token]], dtype=torch.long)), dim=1)
    answer_ids = generated[:-1] if stop_reason == 'answer_end' else generated
    return {'text': tokenizer.decode(answer_ids, skip_special_tokens=False),
            'generated_ids': generated, 'prefix_length': prefix_length, 'stop_reason': stop_reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--tokenizer', required=True, type=Path)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--system')
    parser.add_argument('--condition', choices=CONDITIONS, default='direct')
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--expected-sha256', default=RELEASE_SHA256,
                        help='Explicit identity for another trusted native export; default is the pinned release')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    actual_hash = sha256(args.checkpoint)
    if actual_hash != args.expected_sha256.lower():
        parser.error('Checkpoint SHA-256 mismatch')
    import torch
    import transformers
    import tokenizers
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU reference must start without CUDA initialization')
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=True)
    model, aliases = bind_model(state)
    tokenizer = load_tokenizer(args.tokenizer, state['tokenizer'])
    result = generate(model, tokenizer, prompt_ids(tokenizer, args.prompt, args.condition, args.system),
                      args.max_new_tokens)
    result.update(checkpoint_sha256=actual_hash, schema=state['schema'], condition=args.condition,
                  tensor_count=len(state['model']), unique_parameters=sum(p.numel() for p in model.parameters()),
                  aliases=aliases, stored_values_and_dtypes_preserved=True,
                  torch_version=torch.__version__, cuda_initialized=torch.cuda.is_initialized(),
                  python_version=platform.python_version(), transformers_version=transformers.__version__,
                  tokenizers_version=tokenizers.__version__,
                  backend='cpu_sdpa_full_recompute', sampling='greedy')
    if result['cuda_initialized']:
        raise RuntimeError('Unexpected CUDA initialization')
    if args.report:
        args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(result['text'])
    print('Stop reason: ' + result['stop_reason'])


if __name__ == '__main__':
    main()
