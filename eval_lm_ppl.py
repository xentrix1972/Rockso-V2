"""PPL dataset/report helpers selected from the shared Rockso eval_lm_ppl.py.

The native V2 evaluator supplies its own model loader and scoring routine.
Dataset presets, preprocessing, sequence construction and formatting below
are retained from the current shared evaluator; no legacy trainer is imported.
"""
import re
from typing import Dict, List, Optional
from datasets import load_dataset

PRESETS: Dict[str, Dict[str, str]] = {
    "wikitext2": {
        "dataset": "EleutherAI/wikitext_document_level",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "page",
        "preprocess": "wikitext",
    },
    "wiki2": {
        "dataset": "EleutherAI/wikitext_document_level",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "page",
        "preprocess": "wikitext",
    },
    # Convenience alias. There is WikiText-2 and WikiText-103; no standard
    # WikiText-102 HF config. This keeps old shorthand from becoming a trap.
    "wiki102": {
        "dataset": "EleutherAI/wikitext_document_level",
        "config": "wikitext-2-raw-v1",
        "split": "test",
        "text_field": "page",
        "preprocess": "wikitext",
    },
    "wikitext103": {
        "dataset": "allenai/paloma",
        "config": "wikitext_103",
        "split": "test",
        "text_field": "text",
    },
    "wiki103": {
        "dataset": "allenai/paloma",
        "config": "wikitext_103",
        "split": "test",
        "text_field": "text",
    },
    "ptb": {
        "dataset": "allenai/paloma",
        "config": "ptb",
        "split": "test",
        "text_field": "text",
    },
    "paloma_wikitext103": {
        "dataset": "allenai/paloma",
        "config": "wikitext_103",
        "split": "test",
        "text_field": "text",
    },
}


MECO_VARIANTS = [
    None,
    "WEB: cc",
    "WEB: synth",
    "WEB: wiki",
    "CODE: python",
    "MATH: finemath",
    "TEXTBOOK: synth",
]


def wikitext_detokenizer(text: str) -> str:
    # Mirrors lm-eval-harness wikitext/preprocess_wikitext.py.
    string = text
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def _word_count(text: str) -> int:
    return len(re.split(r"\s+", text))


def load_records(args) -> List[Dict]:
    if args.preset:
        preset_key = args.preset.lower()
        if preset_key not in PRESETS:
            raise SystemExit(f"Unknown preset {args.preset!r}. Choices: {', '.join(sorted(PRESETS))}")
        spec = dict(PRESETS[preset_key])
    else:
        if not args.dataset:
            raise SystemExit("Provide --preset or --dataset.")
        spec = {
            "dataset": args.dataset,
            "config": args.dataset_config,
            "split": args.split,
            "text_field": args.text_field,
            "preprocess": args.preprocess,
        }

    print(
        f"[data] Loading {spec['dataset']} config={spec.get('config')} split={spec['split']}",
        flush=True,
    )
    ds = load_dataset(
        spec["dataset"],
        spec.get("config") or None,
        split=spec["split"],
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    field = spec["text_field"]
    if field not in ds.column_names:
        raise SystemExit(f"Text field {field!r} not in dataset columns: {ds.column_names}")

    records = []
    for raw in ds[field]:
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = wikitext_detokenizer(raw) if spec.get("preprocess") == "wikitext" else raw
        records.append({
            "raw": raw,
            "text": text,
            # lm-eval wikitext intentionally counts original raw page words/bytes.
            "words": _word_count(raw),
            "bytes": len(raw.encode("utf-8")),
        })
    if args.max_samples is not None:
        records = records[: args.max_samples]

    chars = sum(len(r["text"]) for r in records)
    print(
        f"[data] Loaded {len(records):,} non-empty records field={field!r} "
        f"chars={chars:,} preprocess={spec.get('preprocess') or 'none'}",
        flush=True,
    )
    return records


def build_sequences(tokenizer, records: List[Dict], meco_tag: Optional[str], mode: str) -> List[List[int]]:
    prefix_ids = tokenizer.encode(f"{meco_tag}\n", add_special_tokens=False) if meco_tag else []

    if mode == "concat":
        joined = "\n\n".join(r["text"] for r in records)
        ids = tokenizer.encode(joined, add_special_tokens=False)
        if prefix_ids:
            ids = prefix_ids + ids
        return [ids]

    seqs = []
    for rec in records:
        ids = tokenizer.encode(rec["text"], add_special_tokens=False)
        if prefix_ids:
            ids = prefix_ids + ids
        if len(ids) > 1:
            seqs.append(ids)
    return seqs


def print_result(name: str, result: Dict):
    print(
        f"  {name:<22} tok_ppl={result['token_perplexity']:>9.3f} "
        f"word_ppl={result['word_perplexity']:>9.3f} "
        f"byte_ppl={result['byte_perplexity']:>7.4f} "
        f"bpb={result['bits_per_byte']:.4f} "
        f"loss={result['avg_loss']:.5f} tokens={result['tokens']:,} windows={result['windows']:,}",
        flush=True,
    )
