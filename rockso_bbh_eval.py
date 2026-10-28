"""Pinned BIG-Bench Hard assets and Rockso-native generation evaluation.

This is a native protocol, not a claim of bit-identical lm-eval results. Official
questions and fixed exemplar banks are preserved; HRM wrapping, answer parsing,
context budgeting and drop-inclusive macro accuracy are explicitly recorded.
No benchmark data enters a training directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import zip_longest
from pathlib import Path

REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
SOURCE = "https://github.com/suzgunmirac/BIG-Bench-Hard"
DEFAULT_DATA_DIR = Path(os.environ.get("ROCKSO_EVAL_ASSETS", "eval_data")) / f"bbh_{REVISION[:12]}"
SCORER = "bbh_native_final_exact_v1_1"
TASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding",
    "disambiguation_qa", "dyck_languages", "formal_fallacies", "geometric_shapes",
    "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting",
)
FREE_TASKS = {"boolean_expressions", "causal_judgement", "dyck_languages", "formal_fallacies",
              "multistep_arithmetic_two", "navigate", "object_counting", "sports_understanding",
              "web_of_lies", "word_sorting"}
MARKER = re.compile(r"\b(?:(?:so|therefore),?\s+)?(?:the\s+)?(?:final\s+)?answer\s*(?:is\b|:|=)\s*", re.I)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def parse_subjects(value):
    if not value:
        return list(TASKS)
    requested = {s.strip().lower().replace("-", "_") for s in value.split(",") if s.strip()}
    if not requested or requested - set(TASKS):
        raise ValueError(f"Unknown BBH tasks: {sorted(requested-set(TASKS))}; available: {', '.join(TASKS)}")
    return [task for task in TASKS if task in requested]


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_assets(data_dir=None, offline=False):
    root = Path(data_dir or DEFAULT_DATA_DIR)
    manifest_path = root / "ASSETS.json"
    expected = [f"bbh/{task}.json" for task in TASKS]
    expected += [f"cot-prompts/{task}.txt" for task in TASKS]
    expected += ["LICENSE", "README.md"]
    if not manifest_path.exists():
        if offline:
            raise FileNotFoundError(f"BBH cache missing at {root}; omit --bbh-offline once to download it")
        print(f"[bbh] caching official data/prompts at {REVISION[:12]} (8 download threads; no GPU)", flush=True)

        def download(relative):
            url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/{relative}"
            with urllib.request.urlopen(url, timeout=45) as response:
                content = response.read()
            if relative.endswith(".json"):
                payload = json.loads(content)
                if not isinstance(payload.get("examples"), list) or not payload["examples"]:
                    raise ValueError(f"Invalid BBH asset: {relative}")
            destination = root / relative
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ValueError(f"Refusing to overwrite a different local BBH asset: {destination}")
            else:
                _atomic_write(destination, content)
            return relative, {"sha256": digest_bytes(content), "bytes": len(content), "url": url}

        with ThreadPoolExecutor(max_workers=8) as pool:
            files = dict(pool.map(download, expected))
        manifest = {"source": SOURCE, "revision": REVISION, "files": files}
        _atomic_write(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != REVISION or set(manifest.get("files", {})) != set(expected):
        raise ValueError(f"Unexpected BBH cache manifest: {manifest_path}")
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file() or digest_bytes(path.read_bytes()) != record["sha256"]:
            raise ValueError(f"BBH cache integrity failure: {path}; restore the pinned asset")
    return root, manifest


def parse_exemplars(text):
    _, separator, content = text.replace("\r\n", "\n").partition("\n-----\n")
    if not separator:
        raise ValueError("BBH prompt is missing its official header separator")
    pieces = re.split(r"(?m)^Q: ", content.strip())
    description = pieces[0].strip()
    examples = []
    for piece in pieces[1:]:
        question, marker, answer = piece.partition("\nA:")
        if not marker:
            raise ValueError("Malformed BBH exemplar")
        examples.append((question.strip(), answer.strip()))
    if len(examples) != 3 or not description:
        raise ValueError(f"Expected exactly three official BBH exemplars; found {len(examples)}")
    return description, examples


def braced_value(text, open_index):
    depth = 0
    for index in range(open_index, len(text)):
        if index and text[index-1] == "\\":
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_index+1:index], index+1
    return None


def answer_candidate(generation):
    """Extract a declared final answer without looking at the reference answer.

    Do not hunt for answer-looking letters/numbers inside a reasoning trace.
    A malformed last declaration cannot fall back to an earlier correct one.
    """
    text = str(generation or "").strip().replace("**", "")
    markers = list(MARKER.finditer(text))
    if markers:
        text = text[markers[-1].end():].lstrip(": \t\r\n")
        method = "final_answer_marker"
    else:
        method = "whole_response"
    boxes = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    if boxes:
        box = boxes[-1]
        value = braced_value(text, box.end()-1)
        if value is not None:
            content, end = value
            trailing = re.sub(r"\\[\]\)]|\$|[\s.]", "", text[end:])
            prefix = text[:box.start()].strip()
            ambiguous = markers and (re.match(r"not\b", prefix, re.I) or re.search(r"\b(?:either|or)\b", prefix, re.I))
            if not trailing and not ambiguous:
                return content.strip(), "boxed"
        # An incomplete/ambiguous final box cannot recover an earlier answer.
        return text, method
    if markers:
        text = clean_surface(text).split("\n", 1)[0].strip()
    return text, method


def clean_surface(text):
    text = str(text or "").strip()
    # Remove a sentence terminator, never semantic punctuation such as the
    # Dyck closing brackets or an integer's minus sign.
    if text.endswith("."):
        text = text[:-1].rstrip()
    for _ in range(12):
        previous = text
        for left, right in [("**", "**"), ("`", "`"), ('"', '"'), ("'", "'"),
                            ("\\[", "\\]"), ("\\(", "\\)"), ("$$", "$$"), ("$", "$")]:
            if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
                text = text[len(left):-len(right)].strip()
        wrapper = re.match(r"\\(?:text|mathrm|mathbf|boxed|fbox)\s*\{", text)
        if wrapper:
            value = braced_value(text, wrapper.end()-1)
            if value is not None and value[1] == len(text):
                text = value[0].strip()
        if text == previous:
            break
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def normalize_answer(text, task, labels=()):
    text = clean_surface(text)
    if task not in FREE_TASKS:
        match = re.fullmatch(r"(?:\(([A-Z])\)|([A-Z]))", text, re.I)
        label = (match.group(1) or match.group(2)).upper() if match else None
        if match:
            return label if label in labels else None
        described = re.fullmatch(r"(?:\(([A-Z])\)\s+|([A-Z])\s*[:.)-]\s*)(.+)", text, re.I)
        if described:
            label = (described.group(1) or described.group(2)).upper()
            description = described.group(3)
            if label in labels and not re.match(r"(?:or|and|either)\b", description, re.I) and not re.search(r"\([A-Z]\)", description):
                return label
            return None
        # Three official records in these tasks have literal text targets
        # (their options were split at commas upstream). Preserve those golds;
        # never infer a repaired label or special-case parsing by the gold.
        if task in {"movie_recommendation", "ruin_names"} and text and "\n" not in text:
            return " ".join(text.lower().split())
        return None
    if task == "boolean_expressions":
        return text.lower() if text.lower() in {"true", "false"} else None
    if task in {"causal_judgement", "navigate", "sports_understanding", "web_of_lies"}:
        return text.lower() if text.lower() in {"yes", "no"} else None
    if task == "formal_fallacies":
        return text.lower() if text.lower() in {"valid", "invalid"} else None
    if task in {"multistep_arithmetic_two", "object_counting"}:
        return str(int(text)) if re.fullmatch(r"[+-]?\d+", text) else None
    if task == "dyck_languages":
        return re.sub(r"\s+", "", text) if re.fullmatch(r"[)\]}>\s]+", text) and text.strip() else None
    if task == "word_sorting":
        return " ".join(text.lower().split()) if re.fullmatch(r"[a-zA-Z&']+(?:\s+[a-zA-Z&']+)*", text) else None
    raise ValueError(task)


def score_generation(generation, item):
    candidate, method = answer_candidate(generation)
    prediction = normalize_answer(candidate, item["group"], item["labels"])
    gold = normalize_answer(item["gold"], item["group"], item["labels"])
    if gold is None:
        raise ValueError(f"Unscorable gold for {item['item_id']}: {item['gold']!r}")
    return {"answer_candidate": candidate, "extraction_method": method,
            "strict_pred": prediction, "normalized_gold": gold,
            "strict_correct": prediction is not None and prediction == gold}


def build_items(root, subjects, limit=0):
    if limit < 0:
        raise ValueError("BBH --limit must be nonnegative")
    banks, task_rows = {}, []
    for task in subjects:
        banks[task] = parse_exemplars((root / "cot-prompts" / f"{task}.txt").read_text())
        rows = json.loads((root / "bbh" / f"{task}.json").read_text())["examples"]
        selected = []
        for index, row in enumerate(rows):
            question, gold = row["input"], row["target"]
            labels = re.findall(r"(?m)^\(([A-Z])\)\s", question) if task not in FREE_TASKS else []
            if task not in FREE_TASKS and (not labels or len(labels) != len(set(labels))):
                raise ValueError(f"Missing/ambiguous option labels: {task}:{index}")
            item = {"item_id": f"{task}:{index:03d}", "source_index": index, "group": task,
                    "input": question, "gold": gold, "labels": labels,
                    "reference_format": "option_label" if re.fullmatch(r"\([A-Z]\)", gold) else "text",
                    "input_sha256": digest_bytes(question.encode())}
            if normalize_answer(gold, task, labels) is None:
                raise ValueError(f"Unscorable target: {item['item_id']}")
            selected.append(item)
        task_rows.append(selected)
    # Global limits cover tasks evenly instead of taking only the first one or
    # two task files. Canonical source order is preserved within every task.
    items = []
    for group in zip_longest(*task_rows):
        for item in group:
            if item is not None:
                items.append(item)
                if limit and len(items) == limit:
                    return items, banks
    return items, banks


def render_prompt(item, bank, mode, shots):
    if not 0 <= shots <= 3:
        raise ValueError("BBH --num-shots must be 0..3; there are three official exemplars")
    if mode not in {"cot", "direct"}:
        raise ValueError(f"Unknown BBH mode: {mode}")
    description, examples = bank
    parts = [description]
    for question, answer in examples[:shots]:
        if mode == "direct":
            answer, _ = answer_candidate(answer)
            answer = clean_surface(answer)
        parts.append(f"Q: {question}\nA: {answer}")
    suffix = " Let's think step by step.\n" if mode == "cot" else ""
    parts.append(f"Q: {item['input']}\nA:{suffix}")
    return "\n\n".join(parts)


def encode_items(items, banks, tokenizer, wrap, condition, mode, shots,
                 max_context, max_new_tokens, adaptive=False):
    if max_context <= 1 or max_new_tokens <= 0 or max_new_tokens >= max_context:
        raise ValueError("BBH requires 0 < max-new-tokens < max-context")
    encoded, records = [], []
    for index, item in enumerate(items):
        actual_shots = shots
        while True:
            prompt = render_prompt(item, banks[item["group"]], mode, actual_shots)
            wrapped = wrap(prompt, condition)
            ids = [int(i) for i in tokenizer.encode(wrapped, add_special_tokens=False)]
            fits = len(ids) + max_new_tokens < max_context
            if fits or not adaptive or actual_shots == 0:
                break
            actual_shots -= 1
        records.append({**item, "index": index, "requested_shots": shots,
                        "fewshot_count": actual_shots, "prompt_tokens": len(ids),
                        "prompt_sha256": digest_bytes(wrapped.encode()),
                        "status": "pending" if fits else "dropped_context",
                        "strict_pred": None, "strict_correct": False,
                        "generation": None})
        if fits:
            encoded.append((index, ids, prompt, wrapped))
    return encoded, records


def summarize(records):
    def bucket(rows):
        evaluated = [r for r in rows if r["status"] == "evaluated"]
        correct = sum(bool(r["strict_correct"]) for r in evaluated)
        invalid = sum(r["strict_pred"] is None for r in evaluated)
        return {"n": len(evaluated), "n_total": len(rows), "dropped": len(rows)-len(evaluated),
                "correct": correct, "exact_match": correct / len(rows),
                "evaluated_exact_match": correct / len(evaluated) if evaluated else None,
                "invalid": invalid, "invalid_rate": invalid / len(evaluated) if evaluated else None,
                "actual_shots_histogram": dict(sorted(Counter(str(r["fewshot_count"]) for r in evaluated).items()))}
    if not records:
        raise ValueError("No BBH items selected")
    result = bucket(records)
    subjects = {task: bucket([r for r in records if r["group"] == task])
                for task in sorted({r["group"] for r in records})}
    result["micro_acc"] = result.pop("exact_match")
    result["macro_acc"] = sum(row["exact_match"] for row in subjects.values()) / len(subjects)
    result["subjects"] = subjects
    result["task_count"] = len(subjects)
    result["scorer"] = SCORER
    return result


def run_bbh_task(args, harness):
    if args.split not in {None, "test"}:
        raise ValueError("BBH has one evaluation split: --split test")
    shots = 3 if args.num_shots is None else int(args.num_shots)
    if not 0 <= shots <= 3:
        raise ValueError("BBH --num-shots must be 0..3")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.choice_shuffle_seed is not None and args.choice_shuffle_seed >= 0:
        raise ValueError("BBH preserves official option order; omit --choice-shuffle-seed")
    mode = args.bbh_mode
    condition = harness.task_condition(args, "synth,cot" if mode == "cot" else "direct")
    subjects = parse_subjects(args.subjects)
    root, assets = ensure_assets(args.bbh_data_dir, args.bbh_offline)
    items, banks = build_items(root, subjects, args.limit)
    tokenizer = harness.AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    encoded, records = encode_items(items, banks, tokenizer, harness.wrap_hrm, condition, mode, shots,
                                    args.max_context, args.max_new_tokens, args.adaptive_shots)
    print(f"[bbh] {len(items)} selected; {len(encoded)} fit; {len(items)-len(encoded)} context drops; "
          f"mode={mode} requested_shots={shots} condition={condition}", flush=True)
    print(f"[bbh] actual shots among fitting prompts: {dict(Counter(records[i]['fewshot_count'] for i, *_ in encoded))}", flush=True)
    if not encoded:
        raise ValueError("No BBH prompts fit; try --num-shots 0 or --adaptive-shots. No model was loaded.")
    loader = harness.import_implicit_loader(args.implicit)
    model, step, tokens_seen = loader.load_model(args.checkpoint, tokenizer_path=args.tokenizer,
                                               device=args.device, dtype=harness.resolve_dtype(args.dtype))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    suppress = [] if args.no_suppress_special else harness.suppressed_special_ids(tokenizer)
    stops = [s for s in args.stop_strings.split("|") if s] if args.stop_strings else ["\nQ:", "\n\nQ:", "<|im_end|>"]
    for start in range(0, len(encoded), args.batch_size):
        batch = encoded[start:start+args.batch_size]
        generations = harness.generate_greedy_batch(model=model, tokenizer=tokenizer,
            prompt_ids=[row[1] for row in batch], device=args.device,
            max_new_tokens=args.max_new_tokens, max_context=args.max_context,
            suppress_ids=suppress, stop_strings=stops, pad_id=int(pad_id))
        if len(generations) != len(batch):
            raise RuntimeError("BBH generator returned the wrong number of rows")
        for (index, ids, prompt, wrapped), generation in zip(batch, generations):
            records[index].update(score_generation(generation, items[index]),
                                  generation=generation, status="evaluated")
            if args.include_prompts:
                records[index].update(prompt=prompt, wrapped_prompt=wrapped)
        done = min(start+args.batch_size, len(encoded))
        if args.progress_every and (done // args.progress_every > start // args.progress_every or done == len(encoded)):
            print(f"  generated {done}/{len(encoded)}", flush=True)
    summary = summarize(records)
    itemset_hash = digest_json([{k: item[k] for k in ("item_id", "input_sha256", "gold")} for item in items])
    protocol = {"version": 1, "source": SOURCE, "revision": REVISION,
                "asset_manifest_sha256": digest_json(assets), "mode": mode,
                "requested_shots": shots, "exemplars": "official fixed first_n; direct strips exemplar reasoning",
                "adaptive_shots": bool(args.adaptive_shots), "condition": condition,
                "subjects": subjects, "selection": "canonical task order round-robin, original order within task",
                "limit": args.limit, "itemset_sha256": itemset_hash, "scorer": SCORER,
                "max_context": args.max_context, "max_new_tokens": args.max_new_tokens,
                "stop_strings": stops, "dtype": args.dtype, "batch_size": args.batch_size,
                "tokenizer": str(args.tokenizer), "no_suppress_special": bool(args.no_suppress_special),
                "headline": "macro_acc: equal task weighting; context drops count as wrong",
                "native_not_harness_parity": True}
    protocol["signature"] = digest_json(protocol)
    report = {"diagnostic": "rockso_hrm_native_eval", "task": "bbh", "created_unix": time.time(),
              "checkpoint": args.checkpoint, "step": int(step), "tokens_seen": int(tokens_seen),
              "tokenizer": args.tokenizer, "condition": condition,
              "max_context": args.max_context, "max_new_tokens": args.max_new_tokens,
              "num_shots": shots, "summary": summary, "bbh_protocol": protocol,
              "bbh_assets": assets, "strict_scorer": SCORER,
              "context_drops": [r for r in records if r["status"] == "dropped_context"]}
    if args.keep_items:
        report["items"] = records
    return report


def print_bbh_summary(report):
    summary = report["summary"]
    print(f"bbh_mode={report['bbh_protocol']['mode']} scorer={SCORER}")
    print(f"macro_acc          {summary['macro_acc']:.4f}  (headline; drops count as wrong)")
    print(f"micro_acc          {summary['micro_acc']:.4f}")
    print(f"requested_items={summary['n_total']} tasks={summary['task_count']} invalid={summary['invalid']}")
    print(f"actual_shots={summary['actual_shots_histogram']}")
    print("\nGroups:")
    for task, row in summary["subjects"].items():
        print(f"  {task:<42} n={row['n']:>3}/{row['n_total']:<3} "
              f"exact={row['exact_match']:.4f} dropped={row['dropped']} invalid={row['invalid']}")
