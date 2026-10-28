#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rockso HRM-native benchmark harness.

Clean-room evaluator based on the fresh HRM-Text evaluation blueprint:
  * build benchmark prompts from clean HF datasets
  * wrap prompts as <|im_start|><condition tokens>prompt<|im_end|>
  * avoid OLMES recorded requests / old MeCo tags / loglikelihood wrappers

MCQ reports:
  generate_acc   : one-token greedy generation, matching HRM-Text's MCQ path
  rank_bare_acc  : score bare A/B/C/D as next-token labels
  rank_space_acc : score " A"/" B"/... as continuations
  rank_best_acc  : per-choice best of bare/space continuation logprob
  cot_acc         : generate a short solution, then extract final A/B/C/D

The split between these numbers is the useful part: it tells us whether a bad
external benchmark score is math weakness or answer-boundary/tokenization damage.

GPQA adds deterministic choice shuffling, Diamond/main/extended selection, and
an optional four-view counterbalance that separates answer-content consistency
from A/B/C/D label preference. Context-overlength questions are also reported
with drops counted as wrong so a 197/198 run cannot inflate the headline score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from lm_eval.tasks.drop.utils import process_docs as drop_process_docs
    from lm_eval.tasks.drop.utils import process_results as drop_process_results
except Exception:
    drop_process_docs = None
    drop_process_results = None

OPENQA_TASKS = {"coqa", "naturalqs", "triviaqa", "squad"}
GPQA_LOCAL_ROOT = Path(os.environ.get("ROCKSO_GPQA_ROOT", "eval_data/gpqa"))

HRM_SPECIAL_TOKENS = {
    "boq": "<|im_start|>",
    "eoq": "<|im_end|>",
    "eoa": "<|box_end|>",
    "direct": "<|object_ref_start|>",
    "cot": "<|object_ref_end|>",
    "noisy": "<|quad_start|>",
    "synth": "<|quad_end|>",
}


@dataclass
class MMLUItem:
    subject: str
    prompt: str
    gold: str
    choices: List[str]
    choice_texts: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenItem:
    group: str
    prompt: str
    gold: Any
    meta: Dict[str, Any]


def condition_tokens(condition: str) -> str:
    parts = [p.strip() for p in str(condition or "direct").split(",") if p.strip()]
    if not parts:
        parts = ["direct"]
    unknown = [p for p in parts if p not in HRM_SPECIAL_TOKENS]
    if unknown:
        raise ValueError(f"Unknown HRM condition(s): {unknown}")
    return "".join(HRM_SPECIAL_TOKENS[p] for p in parts)


def wrap_hrm(prompt: str, condition: str) -> str:
    return (
        HRM_SPECIAL_TOKENS["boq"]
        + condition_tokens(condition)
        + prompt.strip()
        + HRM_SPECIAL_TOKENS["eoq"]
    )


def task_condition(args, default: str) -> str:
    return str(args.condition or default)


def format_mcq(question: str, choices: Sequence[str], answer_idx: Optional[int] = None) -> Tuple[str, List[str]]:
    text = question.strip() + "\n"
    labels = []
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        text += f"{letter}. {str(choice).strip()}\n"
        labels.append(letter)
    text += f"Answer: {chr(ord('A') + int(answer_idx))}" if answer_idx is not None else "Answer:"
    return text, labels


def format_mcq_text_answer(question: str, choices: Sequence[str], answer_idx: Optional[int] = None) -> Tuple[str, List[str]]:
    text = question.strip() + "\n"
    labels = []
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        text += f"{letter}. {str(choice).strip()}\n"
        labels.append(letter)
    text += f"Answer: {str(choices[int(answer_idx)]).strip()}" if answer_idx is not None else "Answer:"
    return text, labels


def format_mcq_cot(question: str, choices: Sequence[str], answer_idx: Optional[int] = None) -> Tuple[str, List[str]]:
    text = "Question:\n" + question.strip() + "\n"
    labels = []
    for idx, choice in enumerate(choices):
        letter = chr(ord("A") + idx)
        text += f"{letter}. {str(choice).strip()}\n"
        labels.append(letter)
    if answer_idx is not None:
        text += f"Solution:\nThe answer is {chr(ord('A') + int(answer_idx))}."
    else:
        text += "Solution:\n"
    return text, labels


def maybe_shuffle_choices(
    choices: Sequence[str],
    gold_idx: int,
    seed: Optional[int],
    item_key: str,
) -> Tuple[List[str], int, Optional[List[int]]]:
    choices_out = list(choices)
    if seed is None or seed < 0:
        return choices_out, int(gold_idx), None
    order = list(range(len(choices_out)))
    rng = random.Random(f"{seed}:{item_key}")
    rng.shuffle(order)
    shuffled = [choices_out[i] for i in order]
    new_gold = order.index(int(gold_idx))
    return shuffled, new_gold, order


def build_mmlu_items(
    split: str,
    num_shots: int,
    limit: int,
    subjects: Optional[set[str]],
    special_shots: Optional[Dict[str, int]],
    choice_shuffle_seed: Optional[int],
    mcq_mode: str,
) -> List[MMLUItem]:
    special_shots = special_shots or {}

    shot_ds = load_dataset("cais/mmlu", "all", split="dev")
    shots_by_subject: Dict[str, List[Tuple[str, List[str], int]]] = defaultdict(list)
    for row in shot_ds:
        shots_by_subject[str(row["subject"])].append(
            (str(row["question"]), list(row["choices"]), int(row["answer"]))
        )

    test_ds = load_dataset("cais/mmlu", "all", split=split)
    items: List[MMLUItem] = []
    for row in test_ds:
        subject = str(row["subject"])
        if subjects and subject not in subjects:
            continue
        n_shots = int(special_shots.get(subject, num_shots))
        parts = []
        for q, choices, ans_idx in shots_by_subject[subject][:n_shots]:
            if mcq_mode == "cot":
                shot_text, _ = format_mcq_cot(q, choices, ans_idx)
            elif mcq_mode == "rc":
                shot_text, _ = format_mcq_text_answer(q, choices, ans_idx)
            else:
                shot_text, _ = format_mcq(q, choices, ans_idx)
            parts.append(shot_text)
        target_choices, target_gold_idx, _ = maybe_shuffle_choices(
            list(row["choices"]),
            int(row["answer"]),
            choice_shuffle_seed,
            f"mmlu:{subject}:{len(items)}:{row['question']}",
        )
        if mcq_mode == "cot":
            target_text, labels = format_mcq_cot(str(row["question"]), target_choices, None)
        elif mcq_mode == "rc":
            target_text, labels = format_mcq_text_answer(str(row["question"]), target_choices, None)
        else:
            target_text, labels = format_mcq(str(row["question"]), target_choices, None)
        prompt = ("\n\n".join(parts) + "\n\n" if parts else "") + target_text
        gold = chr(ord("A") + int(target_gold_idx))
        items.append(MMLUItem(subject=subject, prompt=prompt, gold=gold, choices=labels, choice_texts=[str(x).strip() for x in target_choices]))
        if limit and len(items) >= limit:
            break
    return items


def build_mmlu_pro_items(
    split: str,
    num_shots: int,
    limit: int,
    subjects: Optional[set[str]],
    choice_shuffle_seed: Optional[int],
    mcq_mode: str,
) -> List[MMLUItem]:
    shot_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    shots_by_category: Dict[str, List[Tuple[str, List[str], int]]] = defaultdict(list)
    for row in shot_ds:
        category = str(row["category"]).strip()
        shots_by_category[category].append(
            (str(row["question"]), list(row["options"]), int(row["answer_index"]))
        )

    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    items: List[MMLUItem] = []
    subject_filter = {s.strip().lower() for s in subjects} if subjects else None
    for row in test_ds:
        category = str(row["category"]).strip()
        src = str(row.get("src") or "").strip()
        if subject_filter and category.lower() not in subject_filter and src.lower() not in subject_filter:
            continue

        parts = []
        for q, choices, ans_idx in shots_by_category[category][:num_shots]:
            if mcq_mode == "cot":
                shot_text, _ = format_mcq_cot(q, choices, ans_idx)
            elif mcq_mode == "rc":
                shot_text, _ = format_mcq_text_answer(q, choices, ans_idx)
            else:
                shot_text, _ = format_mcq(q, choices, ans_idx)
            parts.append(shot_text)

        target_choices, target_gold_idx, _ = maybe_shuffle_choices(
            list(row["options"]),
            int(row["answer_index"]),
            choice_shuffle_seed,
            f"mmlu_pro:{category}:{row.get('question_id')}:{row['question']}",
        )
        if mcq_mode == "cot":
            target_text, labels = format_mcq_cot(str(row["question"]), target_choices, None)
        elif mcq_mode == "rc":
            target_text, labels = format_mcq_text_answer(str(row["question"]), target_choices, None)
        else:
            target_text, labels = format_mcq(str(row["question"]), target_choices, None)
        prompt = ("\n\n".join(parts) + "\n\n" if parts else "") + target_text
        gold = chr(ord("A") + int(target_gold_idx))
        items.append(
            MMLUItem(
                subject=category,
                prompt=prompt,
                gold=gold,
                choices=labels,
                choice_texts=[str(x).strip() for x in target_choices],
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def build_gpqa_items(
    gpqa_set: str,
    limit: int,
    subjects: Optional[set[str]],
    choice_shuffle_seed: Optional[int],
    mcq_mode: str,
    counterbalance: bool,
) -> List[MMLUItem]:
    """Build GPQA items with deterministic choice provenance.

    GPQA stores the correct answer separately from three distractors. A seeded
    base permutation prevents the correct answer from occupying a fixed label.
    Counterbalance mode emits four cyclic rotations of that permutation, so
    every source question's correct content appears once behind A/B/C/D.
    """
    config = f"gpqa_{gpqa_set}"
    local_csv = GPQA_LOCAL_ROOT / f"{config}.csv"
    if local_csv.exists():
        ds = load_dataset("csv", data_files=str(local_csv), split="train")
    else:
        ds = load_dataset("Idavidrein/gpqa", config, split="train")
    subject_filters = {x.casefold() for x in subjects} if subjects else None
    seed = 0 if choice_shuffle_seed is None else int(choice_shuffle_seed)
    items: List[MMLUItem] = []
    source_questions = 0

    for row_idx, row in enumerate(ds):
        domain = str(row.get("High-level domain") or "unknown").strip() or "unknown"
        subdomain = str(row.get("Subdomain") or "unknown").strip() or "unknown"
        if subject_filters and domain.casefold() not in subject_filters and subdomain.casefold() not in subject_filters:
            continue
        if limit and source_questions >= limit:
            break

        question = str(row["Question"]).strip()
        record_id = str(row.get("Record ID") or f"{config}:{row_idx}")
        source_choices = [
            str(row["Correct Answer"]).strip(),
            str(row["Incorrect Answer 1"]).strip(),
            str(row["Incorrect Answer 2"]).strip(),
            str(row["Incorrect Answer 3"]).strip(),
        ]
        base_order = list(range(4))
        random.Random(f"gpqa:{seed}:{record_id}:{question}").shuffle(base_order)
        rotations = range(4) if counterbalance else range(1)

        for view in rotations:
            order = base_order[view:] + base_order[:view]
            choices = [source_choices[i] for i in order]
            gold_idx = order.index(0)
            if mcq_mode == "cot":
                prompt, labels = format_mcq_cot(question, choices, None)
            elif mcq_mode == "rc":
                prompt, labels = format_mcq_text_answer(question, choices, None)
            else:
                prompt, labels = format_mcq(question, choices, None)
            items.append(
                MMLUItem(
                    subject=domain,
                    prompt=prompt,
                    gold=chr(ord("A") + gold_idx),
                    choices=labels,
                    choice_texts=choices,
                    meta={
                        "gpqa_record_id": record_id,
                        "gpqa_set": gpqa_set,
                        "gpqa_domain": domain,
                        "gpqa_subdomain": subdomain,
                        "gpqa_view": int(view),
                        "gpqa_choice_order": order,
                    },
                )
            )
        source_questions += 1
    return items


def arc_gold_index(row: Dict[str, Any]) -> int:
    labels = [str(x) for x in row["choices"]["label"]]
    answer = str(row["answerKey"])
    if answer in labels:
        return labels.index(answer)
    if answer.isdigit():
        return max(0, int(answer) - 1)
    return ord(answer.upper()) - ord("A")


def build_standard_mcq_items(
    task: str,
    split: str,
    num_shots: int,
    limit: int,
    choice_shuffle_seed: Optional[int],
    mcq_mode: str,
) -> List[MMLUItem]:
    def load(path: str, name: Optional[str], ds_split: str):
        return load_dataset(path, name, split=ds_split) if name is not None else load_dataset(path, split=ds_split)

    if task in {"arc", "arc_easy"}:
        path = "allenai/ai2_arc"
        name = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        shot_split = "validation"

        def row_to_doc(row):
            return str(row["question"]), list(row["choices"]["text"]), arc_gold_index(row)

    elif task == "hellaswag":
        path, name, shot_split = "Rowan/hellaswag", None, "train"

        def row_to_doc(row):
            return (
                f"{row['ctx_a']} {str(row['ctx_b']).capitalize()}\nQuestion: Which is the most logical continuation?",
                list(row["endings"]),
                int(row["label"]),
            )

    elif task == "winogrande":
        path, name, shot_split = "allenai/winogrande", "winogrande_debiased", "train"

        def row_to_doc(row):
            return str(row["sentence"]), [str(row["option1"]), str(row["option2"])], int(row["answer"]) - 1

    elif task == "boolq":
        path, name, shot_split = "google/boolq", None, "train"

        def row_to_doc(row):
            return (
                f"{row['passage']}\nQuestion: {str(row['question']).capitalize()}?",
                ["Yes", "No"],
                1 - int(bool(row["answer"])),
            )

    elif task == "sciq":
        path, name, shot_split = "allenai/sciq", None, "train"

        def row_to_doc(row):
            choices = [
                str(row["correct_answer"]),
                str(row["distractor1"]),
                str(row["distractor2"]),
                str(row["distractor3"]),
            ]
            passage = str(row.get("support") or "").strip()
            question = str(row["question"])
            query = f"{passage}\nQuestion: {question}" if passage else question
            return query, choices, 0

    elif task == "piqa":
        path, name, shot_split = "ybisk/piqa", None, "train"

        def row_to_doc(row):
            gold_idx = int(row["label"])
            if gold_idx not in {0, 1}:
                raise ValueError(f"PIQA row has no usable gold label: {row['label']!r}")
            return (
                f"Goal: {str(row['goal']).strip()}\nQuestion: Which solution is more appropriate?",
                [str(row["sol1"]), str(row["sol2"])],
                gold_idx,
            )

    elif task == "siqa":
        path, name, shot_split = "allenai/social_i_qa", None, "train"

        def row_to_doc(row):
            label = str(row["label"]).strip()
            if label not in {"1", "2", "3"}:
                raise ValueError(f"SIQA row has no usable gold label: {row['label']!r}")
            return (
                f"Context: {str(row['context']).strip()}\nQuestion: {str(row['question']).strip()}",
                [str(row["answerA"]), str(row["answerB"]), str(row["answerC"])],
                int(label) - 1,
            )

    else:
        raise ValueError(task)

    shot_docs = []
    effective_shuffle_seed = choice_shuffle_seed if choice_shuffle_seed is not None else (0 if task == "sciq" else None)
    if num_shots > 0:
        load_kwargs = {"revision": "refs/convert/parquet"} if task in {"piqa", "siqa"} else {}
        shot_ds = (
            load_dataset(path, name, split=f"{shot_split}[:{num_shots}]", **load_kwargs)
            if name is not None
            else load_dataset(path, split=f"{shot_split}[:{num_shots}]", **load_kwargs)
        )
        shot_docs = []
        for shot_i, row in enumerate(shot_ds):
            shot_q, shot_choices, shot_gold = row_to_doc(row)
            if task == "sciq":
                shot_choices, shot_gold, _ = maybe_shuffle_choices(
                    shot_choices,
                    int(shot_gold),
                    effective_shuffle_seed,
                    f"{task}:shot:{shot_i}:{shot_q}",
                )
            shot_docs.append((shot_q, shot_choices, shot_gold))

    if task in {"piqa", "siqa"}:
        ds = load_dataset(path, name, split=split, revision="refs/convert/parquet") if name is not None else load_dataset(
            path, split=split, revision="refs/convert/parquet"
        )
    else:
        ds = load(path, name, split)
    items: List[MMLUItem] = []
    for row in ds:
        query, choices, gold_idx = row_to_doc(row)
        choices, gold_idx, _ = maybe_shuffle_choices(
            choices,
            int(gold_idx),
            effective_shuffle_seed,
            f"{task}:{split}:{len(items)}:{query}",
        )
        parts = []
        for shot_q, shot_choices, shot_gold in shot_docs:
            if mcq_mode == "cot":
                shot_text, _ = format_mcq_cot(shot_q, shot_choices, shot_gold)
            elif mcq_mode == "rc":
                shot_text, _ = format_mcq_text_answer(shot_q, shot_choices, shot_gold)
            else:
                shot_text, _ = format_mcq(shot_q, shot_choices, shot_gold)
            parts.append(shot_text)
        if mcq_mode == "cot":
            target_text, labels = format_mcq_cot(query, choices, None)
        elif mcq_mode == "rc":
            target_text, labels = format_mcq_text_answer(query, choices, None)
        else:
            target_text, labels = format_mcq(query, choices, None)
        prompt = ("\n\n".join(parts) + "\n\n" if parts else "") + target_text
        items.append(
            MMLUItem(
                subject=task,
                prompt=prompt,
                gold=chr(ord("A") + int(gold_idx)),
                choices=labels,
                choice_texts=[str(x).strip() for x in choices],
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def default_mcq_shots(task: str) -> int:
    return {
        "mmlu": 5,
        "mmlu_pro": 5,
        "gpqa": 0,
        "arc": 25,
        "arc_easy": 25,
        "hellaswag": 10,
        "winogrande": 5,
        "boolq": 5,
        "sciq": 5,
        "piqa": 5,
        "siqa": 5,
    }[task]


def extract_gsm_truth(answer: str) -> int:
    return int(str(answer).split("####")[-1].strip().replace(",", ""))


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _matching_brace_span(text: str, open_idx: int) -> Optional[Tuple[int, int]]:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return None
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return open_idx + 1, i
    return None


def last_boxed_only_string(text: str) -> Optional[str]:
    src = str(text)
    for m in reversed(list(re.finditer(r"\\(?:boxed|fbox)\s*\{", src))):
        span = _matching_brace_span(src, m.end() - 1)
        if span is not None:
            return src[span[0]:span[1]].strip()
    return None


def _strip_latex_wrappers(text: str) -> str:
    s = str(text or "").strip()
    boxed = last_boxed_only_string(s)
    if boxed is not None:
        s = boxed
    s = s.strip().strip(".")
    for prefix in ("Final Answer:", "Answer:", "final answer:", "answer:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    s = s.replace("\\quad", "").replace("\\qquad", "")
    s = s.replace("$", "").strip()
    return s


def normalize_math_answer(text: Any) -> Optional[str]:
    if text is None:
        return None
    s = _strip_latex_wrappers(str(text))
    if not s:
        return None
    s = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\%", "%")
    s = s.replace("−", "-")
    s = s.replace(",", "")
    s = re.sub(r"\s+", "", s)
    if s.startswith("{") and s.endswith("}"):
        span = _matching_brace_span(s, 0)
        if span == (1, len(s) - 1):
            s = s[1:-1]
    return s or None


def _latex_frac_to_expr(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        frac_cmd = None
        for cmd in ("\\frac", "\\dfrac", "\\tfrac"):
            if s.startswith(cmd, i):
                frac_cmd = cmd
                break
        if frac_cmd is not None:
            j = i + len(frac_cmd)
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j] == "{":
                a_span = _matching_brace_span(s, j)
                if a_span is not None:
                    k = a_span[1] + 1
                    while k < len(s) and s[k].isspace():
                        k += 1
                    if k < len(s) and s[k] == "{":
                        b_span = _matching_brace_span(s, k)
                        if b_span is not None:
                            a = _latex_frac_to_expr(s[a_span[0]:a_span[1]])
                            b = _latex_frac_to_expr(s[b_span[0]:b_span[1]])
                            out.append(f"(({a})/({b}))")
                            i = b_span[1] + 1
                            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _math_to_sympy_expr(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    x = normalize_math_answer(s)
    if x is None:
        return None
    if x.endswith("%"):
        x = f"({x[:-1]})/100"
    x = _latex_frac_to_expr(x)
    x = x.replace("^", "**")
    x = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", x)
    x = re.sub(r"\\([a-zA-Z]+)", r"\1", x)
    return x


def math_answers_equal(pred: Optional[str], gold: Optional[str]) -> bool:
    p = normalize_math_answer(pred)
    g = normalize_math_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    if p.lower() == g.lower():
        return True
    try:
        import sympy as sp
        p_expr = _math_to_sympy_expr(p)
        g_expr = _math_to_sympy_expr(g)
        if p_expr is None or g_expr is None:
            return False
        pv = sp.sympify(p_expr)
        gv = sp.sympify(g_expr)
        return bool(sp.simplify(pv - gv) == 0)
    except Exception:
        return False


def extract_gsm_strict(text: str) -> Optional[int]:
    boxed = last_boxed_only_string(text)
    if boxed:
        text = boxed
    text = text.replace(",", "").replace("$", "").strip()
    try:
        return int(float(text))
    except (ValueError, OverflowError):
        return None


def extract_gsm_cot_strict(text: str) -> Optional[int]:
    """Strict GSM8K-CoT extraction without changing the historical 0-shot scorer."""
    historical = extract_gsm_strict(text)
    if historical is not None:
        return historical
    matches = re.findall(
        r"the answer is\s+(-?\$?[0-9][0-9,]*(?:\.[0-9]+)?)\.?",
        str(text),
        flags=re.IGNORECASE,
    )
    if not matches:
        return None
    try:
        return int(float(matches[-1].replace(",", "").replace("$", "")))
    except (ValueError, OverflowError):
        return None


def extract_gsm_loose(text: str) -> Optional[int]:
    strict = extract_gsm_strict(text)
    if strict is not None:
        return strict
    boxed = last_boxed_only_string(text)
    src = boxed if boxed is not None else text
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", src.replace(",", ""))
    if not nums:
        return None
    try:
        return int(float(nums[-1]))
    except (ValueError, OverflowError):
        return None


# Fixed eight-example CoT prompt used by EleutherAI lm-evaluation-harness's
# gsm8k_cot v3 task. Keep this list frozen: changing examples changes the eval.
# Source: lm_eval/tasks/gsm8k/gsm8k-cot.yaml (first_n, num_fewshot=8).
GSM8K_COT_EXAMPLES: List[Tuple[str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the "
        "grove today. After they are done, there will be 21 trees. How many trees "
        "did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more "
        "were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many "
        "cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many "
        "pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they "
        "had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
        "lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to "
        "Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and "
        "dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, "
        "then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were "
        "installed each day, from monday to thursday. How many computers are now "
        "in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers "
        "were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer "
        "is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the end of "
        "wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
        "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The "
        "answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does "
        "she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 "
        "dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
]


def gsm8k_fewshot_prefix(num_shots: int) -> str:
    if not 0 <= num_shots <= len(GSM8K_COT_EXAMPLES):
        raise ValueError(
            f"--num-shots for GSM8K must be 0..{len(GSM8K_COT_EXAMPLES)}; "
            "the fixed canonical CoT bank contains eight examples"
        )
    return "".join(
        f"Q: {question}\nA: {answer}\n\n"
        for question, answer in GSM8K_COT_EXAMPLES[:num_shots]
    )


def build_gsm8k_items(split: str, num_shots: int, limit: int) -> List[GenItem]:
    prefix = gsm8k_fewshot_prefix(num_shots)
    ds = load_dataset("gsm8k", "main", split=split)
    items: List[GenItem] = []
    for row in ds:
        question = str(row["question"]).strip()
        items.append(
            GenItem(
                group="gsm8k",
                # Preserve the historical zero-shot prompt byte-for-byte. The
                # Q:/A: frame is introduced only when exemplars are requested.
                prompt=f"{prefix}Q: {question}\nA:" if num_shots else str(row["question"]),
                gold=extract_gsm_truth(str(row["answer"])),
                meta={"fewshot_count": num_shots},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def extract_math_truth(solution: str) -> Optional[str]:
    return normalize_math_answer(last_boxed_only_string(solution))


def extract_math_strict(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    return normalize_math_answer(boxed)


def extract_math_loose(text: str) -> Optional[str]:
    strict = extract_math_strict(text)
    if strict is not None:
        return strict
    src = str(text or "").strip()
    patterns = [
        r"(?:final answer|answer)\s*(?:is|=|:)\s*([^\n]+)",
        r"(?:therefore|thus|so),?\s*(?:the answer is)?\s*([^\n]+)$",
    ]
    for pat in patterns:
        matches = re.findall(pat, src, flags=re.IGNORECASE)
        if matches:
            ans = normalize_math_answer(matches[-1])
            if ans is not None:
                return ans
    m = re.fullmatch(r"\s*\$([^$]+)\$\.?\s*", src, flags=re.DOTALL)
    if m:
        return normalize_math_answer(m.group(1))
    return None


def parse_math_subjects(subjects: Optional[str]) -> List[str]:
    if not subjects:
        return list(MATH_SUBJECTS)
    alias = {
        "counting": "counting_and_probability",
        "counting_probability": "counting_and_probability",
        "intermediate": "intermediate_algebra",
        "number": "number_theory",
    }
    out = []
    for raw in str(subjects).split(","):
        key = raw.strip().lower().replace(" ", "_").replace("-", "_")
        if not key:
            continue
        key = alias.get(key, key)
        if key not in MATH_SUBJECTS:
            raise ValueError(f"Unknown MATH subject '{raw}'. Known: {', '.join(MATH_SUBJECTS)}")
        out.append(key)
    return out or list(MATH_SUBJECTS)


def build_math_items(split: str, limit: int, subjects: Sequence[str]) -> List[GenItem]:
    items: List[GenItem] = []
    per_subject_limit = 0
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split=split)
        for row in ds:
            gold = extract_math_truth(str(row["solution"]))
            if gold is None:
                continue
            items.append(
                GenItem(
                    group=str(row.get("type") or subject).strip().lower().replace(" ", "_").replace("&", "and"),
                    prompt=f"Problem:\n{str(row['problem']).strip()}\n\nSolution:",
                    gold=gold,
                    meta={
                        "subject": subject,
                        "level": str(row.get("level", "")),
                        "type": str(row.get("type", subject)),
                    },
                )
            )
            per_subject_limit += 1
            if limit and len(items) >= limit:
                return items
    return items


DROP_EXAMPLES = [
    "To start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Aqib Talib intercepted a pass from Matthew Stafford and ran it in 28 yards. The Lions responded with a 28-yard field goal. In the second quarter, Detroit took the lead with a 36-yard touchdown catch by Calvin Johnson, and later added more points when Tony Scheffler caught an 11-yard TD pass. Tampa Bay responded with a 31-yard field goal just before halftime. The second half was relatively quiet, with each team only scoring one touchdown. First, Detroit's Calvin Johnson caught a 1-yard pass in the third quarter. The game's final points came when Mike Williams of Tampa Bay caught a 5-yard pass. The Lions won their regular season opener for the first time since 2007\nQ: How many points did the buccaneers need to tie in the first?\nA: 3",
    "Trying to snap a two-game skid, the Bills flew to Gillette Stadium for a Week 3 divisional fight with the New England Patriots. In the first quarter, QB J. P. Losman was immediately injured on the first offensive play of the game. He would finish the series, but ended up on the bench for the rest of the game. After New England took the lead with kicker Stephen Gostkowski's 24-yard field goal, rookie QB Trent Edwards played the rest of the game for Buffalo. The Bills would get their only score of the game as RB Marshawn Lynch got an 8-yard TD run, and a Rian Lindell extra point put the Bills ahead surprisingly 7-3. However, in the second quarter, the Patriots were able to open up their running game when Bills rookie standout Paul Posluszny was lost due to a broken arm. This left passing lanes open, and for the rest of the game, the Patriots dominated. QB Tom Brady's 8-yard TD pass to TE Benjamin Watson and a 3-yard TD pass to WR Randy Moss made it 17-7 at the half. In the third quarter, New England continued its conquest with Brady's 4-yard TD pass to WR Jabar Gaffney and RB Sammy Morris' 4-yard TD run. In the fourth quarter, the Patriots ended the day with Brady and Moss hooking up with each other again on a 45-yard TD pass.\nQ: How many games had the Bills won before this game?\nA: 0",
    "The French king, John II, had been held captive in England. The Treaty of Brétigny set his ransom at 3 million crowns and allowed for hostages to be held in lieu of John. The hostages included two of his sons, several princes and nobles, four inhabitants of Paris, and two citizens from each of the nineteen principal towns of France. While these hostages were held, John returned to France to try and raise funds to pay the ransom. In 1362 John's son Louis of Anjou, a hostage in English-held Calais, escaped captivity. So, with his stand-in hostage gone, John felt honor-bound to return to captivity in England. The French crown had been at odds with Navarre since 1354, and in 1363 the Navarrese used the captivity of John II in London and the political weakness of the Dauphin to try to seize power. Although there was no formal treaty, Edward III supported the Navarrese moves, particularly as there was a prospect that he might gain control over the northern and western provinces as a consequence. With this in mind, Edward deliberately slowed the peace negotiations. In 1364, John II died in London, while still in honourable captivity. Charles V succeeded him as king of France. On 7 May 1364, one month after the dauphin's accession and three days before his coronation as Charles V, the Navarrese suffered a crushing defeat at the Battle of Cocherel.\nQ: How many years before Navarrase used the captivity of John II?\nA: 9",
]


def build_drop_items(split: str, num_shots: int, limit: int) -> List[GenItem]:
    if drop_process_docs is None:
        raise RuntimeError("DROP requires lm_eval.tasks.drop.utils in the active environment")
    assert 0 <= num_shots <= len(DROP_EXAMPLES), "--num-shots for DROP must be 0..3"
    prefix = "".join(f"{shot}\n\n" for shot in DROP_EXAMPLES[:num_shots])
    ds = load_dataset("EleutherAI/drop", split=split)
    docs = list(drop_process_docs(ds))
    items: List[GenItem] = []
    for doc in docs:
        items.append(
            GenItem(
                group="drop",
                prompt=f"{prefix}{doc['passage']}\nQ: {doc['question']}\nA:",
                gold=doc,
                meta={"query_id": doc.get("query_id")},
            )
        )
        if limit and len(items) >= limit:
            break
    return items


def dedupe_strings(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def openqa_item(group: str, prompt: str, answers: Iterable[Any], meta: Optional[Dict[str, Any]] = None) -> GenItem:
    return GenItem(
        group=group,
        prompt=prompt,
        gold=dedupe_strings(answers),
        meta=meta or {},
    )


def build_naturalqs_doc(row: Dict[str, Any], index: int = -1) -> GenItem:
    return openqa_item(
        "naturalqs",
        f"Question: {str(row['question']).strip()}\nAnswer:",
        row.get("answer") or [],
        {"index": index},
    )


def build_triviaqa_doc(row: Dict[str, Any]) -> GenItem:
    ans = row.get("answer") or {}
    aliases = list(ans.get("aliases") or []) + list(ans.get("normalized_aliases") or [])
    value = ans.get("value")
    if value:
        aliases.append(value)
    return openqa_item(
        "triviaqa",
        f"Question: {str(row['question']).strip()}\nAnswer:",
        aliases,
        {"question_id": row.get("question_id"), "answer_value": value},
    )


def build_squad_doc(row: Dict[str, Any], reduced_spacing: bool = True) -> GenItem:
    title = str(row.get("title", "")).strip()
    context = str(row.get("context", "")).strip()
    question = str(row.get("question", "")).strip()
    if reduced_spacing:
        prompt = f"Title: {title}\nBackground: {context}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Title: {title}\n\nBackground: {context}\n\nQuestion: {question}\n\nAnswer:"
    return openqa_item(
        "squad",
        prompt,
        (row.get("answers") or {}).get("text") or [],
        {"id": row.get("id"), "title": title},
    )


def build_coqa_docs(row: Dict[str, Any]) -> List[GenItem]:
    core_id = str(row.get("id", ""))
    source = str(row.get("source", ""))
    story = str(row.get("story", "")).strip()
    questions = list((row.get("questions") or {}).get("input_text") or [])
    all_answers = list((row.get("answers") or {}).get("input_text") or [])
    additional = []
    for extra in (row.get("additional_answers") or {}).values():
        additional.append(list(extra.get("input_text") or []))

    docs: List[GenItem] = []
    previous_qa: List[Dict[str, str]] = []
    for turn_idx, question in enumerate(questions):
        answers: List[str] = []
        if turn_idx < len(all_answers):
            answers.append(str(all_answers[turn_idx]))
        for answer_list in additional:
            if turn_idx < len(answer_list):
                answers.append(str(answer_list[turn_idx]))

        prompt = f"Passage: {story}"
        if previous_qa:
            prompt += "\n\nPreceding questions:"
            for prev in previous_qa:
                prompt += f"\n\nQuestion: {prev['question']}\nAnswer: {prev['answer']}"
        prompt += f"\n\nFinal question:\n\nQuestion: {str(question).strip()}\nAnswer:"

        docs.append(
            openqa_item(
                "coqa",
                prompt,
                answers,
                {"id": f"{core_id}_turn{turn_idx}", "source": source, "turn": turn_idx},
            )
        )
        previous_qa.append({"question": str(question).strip(), "answer": answers[0] if answers else ""})
    return docs


def build_openqa_items(task: str, split: str, num_shots: int, limit: int) -> List[GenItem]:
    if task == "coqa":
        ds = load_dataset("EleutherAI/coqa", split=split, revision="refs/convert/parquet")
        docs: List[GenItem] = []
        for row in ds:
            docs.extend(build_coqa_docs(row))
            if limit and len(docs) >= limit:
                return docs[:limit]
        return docs

    if task == "naturalqs":
        ds = load_dataset("google-research-datasets/nq_open", split=split)
        shots: List[GenItem] = []
        if num_shots > 0:
            train = load_dataset("google-research-datasets/nq_open", split="train")
            shots = [build_naturalqs_doc(row, i) for i, row in enumerate(train.select(range(num_shots)))]
        prefix = "".join(f"{shot.prompt} {shot.gold[0] if shot.gold else ''}\n\n" for shot in shots)
        items = [build_naturalqs_doc(row, i) for i, row in enumerate(ds)]
    elif task == "triviaqa":
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=split)
        shots = []
        if num_shots > 0:
            train = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split="train")
            shots = [build_triviaqa_doc(row) for row in train.select(range(num_shots))]
        prefix = "".join(f"{shot.prompt} {shot.meta.get('answer_value') or (shot.gold[0] if shot.gold else '')}\n\n" for shot in shots)
        items = [build_triviaqa_doc(row) for row in ds]
    elif task == "squad":
        ds = load_dataset("allenai/squad", split=split)
        shots = []
        if num_shots > 0:
            train = load_dataset("allenai/squad", split="train")
            shots = [build_squad_doc(row) for row in train.select(range(num_shots))]
        prefix = "".join(f"{shot.prompt} {shot.gold[0] if shot.gold else ''}\n\n" for shot in shots)
        items = [build_squad_doc(row) for row in ds]
    else:
        raise ValueError(task)

    if prefix:
        for item in items:
            item.prompt = prefix + item.prompt
    return items[:limit] if limit else items


def import_implicit_loader(implicit_path: str):
    p = Path(implicit_path).resolve()
    if str(p.parent) not in sys.path:
        sys.path.insert(0, str(p.parent))
    import importlib

    return importlib.import_module(p.stem)


def resolve_dtype(dtype: str) -> str:
    d = str(dtype or "bf16").lower()
    if d in {"bf16", "bfloat16"}:
        return "bf16"
    if d in {"fp32", "float32"}:
        return "fp32"
    if d in {"fp16", "float16"}:
        return "fp16"
    if d in {"native", "none", "checkpoint"}:
        return "native"
    raise ValueError(f"Unsupported dtype: {dtype}")


def encode_prompt(tokenizer, item: MMLUItem, condition: str, max_context: int) -> Optional[List[int]]:
    text = wrap_hrm(item.prompt, condition)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) >= max_context:
        return None
    return [int(x) for x in ids]


def prompt_debug_record(tokenizer, prompt: str, condition: str, max_context: int) -> Dict[str, Any]:
    wrapped = wrap_hrm(prompt, condition)
    ids = tokenizer.encode(wrapped, add_special_tokens=False)
    return {
        "prompt": prompt,
        "wrapped_prompt": wrapped,
        "prompt_tokens": len(ids),
        "max_context": max_context,
    }


def encode_with_adaptive_shots(
    tokenizer,
    item: MMLUItem,
    condition: str,
    max_context: int,
) -> Optional[List[int]]:
    ids = encode_prompt(tokenizer, item, condition, max_context)
    if ids is not None:
        return ids

    # Conservative fallback: peel few-shots from the front until it fits.
    chunks = item.prompt.split("\n\n")
    while len(chunks) > 1:
        chunks = chunks[1:]
        tmp = MMLUItem(item.subject, "\n\n".join(chunks), item.gold, item.choices, item.choice_texts)
        ids = encode_prompt(tokenizer, tmp, condition, max_context)
        if ids is not None:
            return ids
    return None


def model_logits(model, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
    try:
        out = model(input_ids, position_ids=pos, return_logits=True)
    except TypeError:
        try:
            out = model(input_ids, position_ids=pos)
        except TypeError:
            out = model(input_ids)
    if isinstance(out, dict):
        logits = out.get("logits")
        if logits is None:
            raise RuntimeError(f"model returned dict without logits: {sorted(out.keys())}")
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    rows = torch.arange(B, device=input_ids.device)
    return logits[rows, lengths - 1, :]


def model_full_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    B, T = input_ids.shape
    pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
    try:
        out = model(input_ids, position_ids=pos, return_logits=True)
    except TypeError:
        try:
            out = model(input_ids, position_ids=pos)
        except TypeError:
            out = model(input_ids)
    if isinstance(out, dict):
        logits = out.get("logits")
        if logits is None:
            raise RuntimeError(f"model returned dict without logits: {sorted(out.keys())}")
    elif isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    return logits


def make_batch(rows: List[Tuple[int, List[int]]], pad_id: int, device: str) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
    idxs = [r[0] for r in rows]
    lengths = torch.tensor([len(r[1]) for r in rows], dtype=torch.long, device=device)
    max_len = int(lengths.max().item())
    x = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    for i, (_, ids) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return idxs, x, lengths


def make_seq_batch(rows: List[Tuple[Any, List[int]]], pad_id: int, device: str) -> Tuple[List[Any], torch.Tensor]:
    keys = [r[0] for r in rows]
    max_len = max(len(r[1]) for r in rows)
    x = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
    for i, (_, ids) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return keys, x


SUPPRESSION_POLICY = "tokenizer_declared_specials_only_v1"


def suppressed_special_ids(tokenizer) -> List[int]:
    """Suppress declared controls, preserving ordinary code/math vocabulary.

    Some native controls are marked special in added-token metadata but are
    absent from all_special_ids. Token spelling is not a safe substitute:
    ordinary tokens such as <, >, <=, << and >> must remain generatable.
    """
    eoa_id = tokenizer.convert_tokens_to_ids(HRM_SPECIAL_TOKENS["eoa"])
    special_ids = {int(idx) for idx in tokenizer.all_special_ids}
    special_ids.update(
        int(idx)
        for idx, token in tokenizer.added_tokens_decoder.items()
        if token.special
    )
    special_ids.discard(int(eoa_id))
    return sorted(special_ids)


def label_token_variants(tokenizer, labels: Sequence[str], form: str) -> Dict[str, List[int]]:
    variants = {}
    for lab in labels:
        text = lab if form == "bare" else " " + lab
        ids = tokenizer.encode(text, add_special_tokens=False)
        variants[lab] = [int(x) for x in ids]
    return variants


@torch.no_grad()
def continuation_logprob(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> float:
    cur = list(prompt_ids)
    total = 0.0
    for tid in cont_ids:
        x = torch.tensor([cur], dtype=torch.long, device=device)
        lengths = torch.tensor([len(cur)], dtype=torch.long, device=device)
        logits = model_logits(model, x, lengths)
        lp = F.log_softmax(logits.float(), dim=-1)
        total += float(lp[0, int(tid)].item())
        cur.append(int(tid))
    return total


def argmax_label(scores: Dict[str, float]) -> str:
    return max(scores, key=lambda k: scores[k])


def continuation_logprob_stats(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> Tuple[float, int]:
    return continuation_logprob(model, prompt_ids, cont_ids, device), max(1, len(cont_ids))


@torch.no_grad()
def continuation_logprob_next_only(model, prompt_ids: List[int], cont_ids: List[int], device: str) -> float:
    cur = list(prompt_ids)
    total = 0.0
    for tid in cont_ids:
        x = torch.tensor([cur], dtype=torch.long, device=device)
        lengths = torch.tensor([len(cur)], dtype=torch.long, device=device)
        logits = model_logits(model, x, lengths).float()[0]
        total += float((logits[int(tid)] - torch.logsumexp(logits, dim=-1)).item())
        cur.append(int(tid))
    return total


def text_continuation_variants(tokenizer, text: str) -> Dict[str, List[int]]:
    clean = str(text).strip()
    variants = {
        "bare": clean,
        "space": " " + clean,
    }
    out = {}
    for name, value in variants.items():
        ids = tokenizer.encode(value, add_special_tokens=False)
        if ids:
            out[name] = [int(x) for x in ids]
    return out


@torch.no_grad()
def score_rc_rows(
    model,
    tokenizer,
    items: List[MMLUItem],
    rows: List[Tuple[int, List[int]]],
    device: str,
    pad_id: int,
    max_context: int,
) -> Dict[int, Dict[str, Any]]:
    del pad_id
    by_local: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(dict)
    text_by_local: Dict[int, Dict[str, str]] = defaultdict(dict)
    for local_i, (item_idx, prompt_ids) in enumerate(rows):
        item = items[item_idx]
        for lab, text in zip(item.choices, item.choice_texts):
            for _, cont_ids in text_continuation_variants(tokenizer, text).items():
                seq = list(prompt_ids) + cont_ids
                if len(seq) >= max_context:
                    continue
                total = continuation_logprob_next_only(model, prompt_ids, cont_ids, device)
                n_tok = max(1, len(cont_ids))
                n_char = max(1, len(str(text).strip()))
                stats = {
                    "raw": total,
                    "tok_norm": total / n_tok,
                    "char_norm": total / n_char,
                    "tokens": float(n_tok),
                    "chars": float(n_char),
                }
                prev = by_local[local_i].get(lab)
                if prev is None or stats["tok_norm"] > prev["tok_norm"]:
                    by_local[local_i][lab] = stats
                    text_by_local[local_i][lab] = str(text).strip()

    out = {}
    for local_i, scores in by_local.items():
        if not scores:
            continue
        raw_scores = {lab: vals["raw"] for lab, vals in scores.items()}
        tok_scores = {lab: vals["tok_norm"] for lab, vals in scores.items()}
        char_scores = {lab: vals["char_norm"] for lab, vals in scores.items()}
        out[local_i] = {
            "rc_raw_pred": argmax_label(raw_scores),
            "rc_tok_norm_pred": argmax_label(tok_scores),
            "rc_char_norm_pred": argmax_label(char_scores),
            "rc_raw_scores": raw_scores,
            "rc_tok_norm_scores": tok_scores,
            "rc_char_norm_scores": char_scores,
            "choice_texts": text_by_local[local_i],
        }
    return out


def stop_token_ids(tokenizer) -> set[int]:
    out = set()
    for tid in (tokenizer.eos_token_id, tokenizer.bos_token_id):
        if tid is not None:
            out.add(int(tid))
    for tok in (HRM_SPECIAL_TOKENS["eoa"], HRM_SPECIAL_TOKENS["boq"], HRM_SPECIAL_TOKENS["eoq"]):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            out.add(int(tid))
    return out


def trim_generation(text: str, stop_strings: Sequence[str], preserve_whitespace: bool = False) -> str:
    out = text
    for ss in stop_strings:
        if ss and ss in out:
            out = out.split(ss)[0]
    return out if preserve_whitespace else out.strip()


def extract_mcq_letter(text: str, labels: Sequence[str]) -> str:
    valid = set(labels)
    if not valid:
        return "INVALID"
    label_class = "".join(re.escape(str(lab)) for lab in sorted(valid))
    src = str(text or "").strip()
    patterns = [
        rf"(?:final\s+answer|answer|option|choice)\s*(?:is|=|:)?\s*\(?([{label_class}])\)?",
        rf"\b([{label_class}])\s*(?:is\s+the\s+answer|is\s+correct)\b",
    ]
    for pat in patterns:
        matches = re.findall(pat, src, flags=re.IGNORECASE)
        for match in reversed(matches):
            lab = str(match).upper()
            if lab in valid:
                return lab
    matches = re.findall(rf"(?<![A-Za-z])([{label_class}])(?:[\).:]|\b)", src, flags=re.IGNORECASE)
    for match in reversed(matches):
        lab = str(match).upper()
        if lab in valid:
            return lab
    return "INVALID"


@torch.no_grad()
def generate_greedy_batch(
    model,
    tokenizer,
    prompt_ids: List[List[int]],
    device: str,
    max_new_tokens: int,
    max_context: int,
    suppress_ids: Sequence[int],
    stop_strings: Sequence[str],
    pad_id: int,
    preserve_whitespace: bool = False,
) -> List[str]:
    rows = [list(ids) for ids in prompt_ids]
    generated: List[List[int]] = [[] for _ in rows]
    active = [True for _ in rows]
    stops = stop_token_ids(tokenizer)

    for _ in range(max_new_tokens):
        active_rows = [(i, rows[i]) for i, flag in enumerate(active) if flag]
        if not active_rows:
            break
        idxs, x, lengths = make_batch(active_rows, pad_id, device)
        logits = model_logits(model, x, lengths)
        if suppress_ids:
            logits = logits.clone()
            logits[:, list(suppress_ids)] = -torch.inf
        next_ids = logits.argmax(dim=-1).tolist()
        for row_i, tid in zip(idxs, next_ids):
            tid = int(tid)
            if tid in stops or len(rows[row_i]) + 1 >= max_context:
                active[row_i] = False
                continue
            generated[row_i].append(tid)
            rows[row_i].append(tid)
            text_so_far = tokenizer.decode(generated[row_i], skip_special_tokens=True)
            if any(ss and ss in text_so_far for ss in stop_strings):
                active[row_i] = False

    return [
        trim_generation(tokenizer.decode(ids, skip_special_tokens=True), stop_strings, preserve_whitespace)
        for ids in generated
    ]


def summarize_mcq(items: List[MMLUItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n = len(per_item)
    out["n"] = n
    out["dropped"] = len(items) - n
    pred_keys = [k[:-5] for k in per_item[0].keys() if k.endswith("_pred")] if per_item else []
    for key in pred_keys:
        out[f"{key}_acc"] = sum(1 for r in per_item if r[f"{key}_pred"] == r["gold"]) / max(1, n)
    by_subj: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"n": 0, **{k: 0 for k in pred_keys}})
    for r in per_item:
        s = by_subj[r["subject"]]
        s["n"] += 1
        for key in pred_keys:
            s[key] += int(r[f"{key}_pred"] == r["gold"])
    out["subjects"] = {
        subj: {
            "n": v["n"],
            **{f"{k}_acc": v[k] / max(1, v["n"]) for k in pred_keys},
        }
        for subj, v in sorted(by_subj.items())
    }
    return out


def summarize_gpqa(items: List[MMLUItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = summarize_mcq(items, per_item)
    pred_keys = [k[:-5] for k in per_item[0].keys() if k.endswith("_pred")] if per_item else []
    out["chance_acc"] = 0.25
    out["gold_label_histogram"] = dict(sorted(Counter(r["gold"] for r in per_item).items()))
    out["prediction_histograms"] = {
        key: dict(sorted(Counter(r[f"{key}_pred"] for r in per_item).items()))
        for key in pred_keys
    }
    out["accuracy_by_gold_label"] = {}
    for key in pred_keys:
        rows_by_gold = defaultdict(list)
        for row in per_item:
            rows_by_gold[row["gold"]].append(row)
        out["accuracy_by_gold_label"][key] = {
            gold: {
                "n": len(rows),
                "acc": sum(r[f"{key}_pred"] == gold for r in rows) / max(1, len(rows)),
            }
            for gold, rows in sorted(rows_by_gold.items())
        }

    all_question_ids = {str(item.meta.get("gpqa_record_id", i)) for i, item in enumerate(items)}
    by_question: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_item:
        by_question[str(row.get("gpqa_record_id", row["index"]))].append(row)
    out["question_n"] = len(all_question_ids)
    out["scored_question_n"] = len(by_question)
    out["dropped_question_n"] = len(all_question_ids - set(by_question))
    out["score_step_per_question"] = 1.0 / max(1, len(all_question_ids))
    for key in pred_keys:
        correct = sum(r[f"{key}_pred"] == r["gold"] for r in per_item)
        out[f"{key}_all_acc"] = correct / max(1, len(items))
    expected_views = max((int(r.get("gpqa_view", 0)) for r in per_item), default=0) + 1
    out["views_per_question"] = expected_views

    if expected_views > 1:
        complete = [rows for rows in by_question.values() if len(rows) == expected_views]
        out["counterbalance"] = {
            "complete_questions": len(complete),
            "complete_question_coverage": len(complete) / max(1, len(all_question_ids)),
        }
        for key in pred_keys:
            all_correct = 0
            any_correct = 0
            content_consistent = 0
            consistent_correct = 0
            consistent_wrong = 0
            for rows in complete:
                correctness = [r[f"{key}_pred"] == r["gold"] for r in rows]
                all_correct += int(all(correctness))
                any_correct += int(any(correctness))
                content_ids = []
                for row in rows:
                    pred = row[f"{key}_pred"]
                    if pred in {"A", "B", "C", "D"}:
                        label_idx = ord(pred) - ord("A")
                        content_ids.append(int(row["gpqa_choice_order"][label_idx]))
                    else:
                        content_ids.append(-1)
                if len(set(content_ids)) == 1:
                    content_consistent += 1
                    consistent_correct += int(content_ids[0] == 0)
                    consistent_wrong += int(content_ids[0] != 0)
            question_scores = []
            correct_view_hist = Counter()
            for question_id in all_question_ids:
                rows = by_question.get(question_id, [])
                correct_views = sum(r[f"{key}_pred"] == r["gold"] for r in rows)
                correct_view_hist[correct_views] += 1
                question_scores.append(correct_views / expected_views)
            qn = len(question_scores)
            qmean = sum(question_scores) / max(1, qn)
            if qn > 1:
                qvar = sum((x - qmean) ** 2 for x in question_scores) / (qn - 1)
                qse = math.sqrt(qvar / qn)
            else:
                qse = 0.0
            complete_denom = max(1, len(complete))
            all_denom = max(1, len(all_question_ids))
            out["counterbalance"][key] = {
                "all_views_correct_rate": all_correct / all_denom,
                "any_view_correct_rate": any_correct / all_denom,
                "content_consistency_rate": content_consistent / complete_denom,
                "consistent_correct_questions": consistent_correct,
                "consistent_wrong_questions": consistent_wrong,
                "correct_views_per_question": dict(sorted(correct_view_hist.items())),
                "question_cluster_mean_acc": qmean,
                "question_cluster_se": qse,
                "question_cluster_normal95": [max(0.0, qmean - 1.96 * qse), min(1.0, qmean + 1.96 * qse)],
                "question_cluster_z_vs_chance": (qmean - 0.25) / qse if qse > 0 else None,
            }
    return out


def run_mcq_task(args) -> Dict[str, Any]:
    condition = task_condition(args, "direct,cot" if args.mcq_mode == "cot" else "direct")
    split = args.split or "test"
    num_shots = default_mcq_shots(args.task) if args.num_shots is None else int(args.num_shots)
    effective_choice_shuffle_seed = args.choice_shuffle_seed
    if args.task == "mmlu":
        subjects = set(x.strip() for x in args.subjects.split(",") if x.strip()) if args.subjects else None
        special_shots = {"high_school_european_history": 3}
        if args.no_special_shots:
            special_shots = {}
        items = build_mmlu_items(split, num_shots, args.limit, subjects, special_shots, args.choice_shuffle_seed, args.mcq_mode)
        print(f"Built MMLU items: {len(items)} split={split} subjects={subjects or 'all'}")
    elif args.task == "mmlu_pro":
        subjects = set(x.strip() for x in args.subjects.split(",") if x.strip()) if args.subjects else None
        items = build_mmlu_pro_items(split, num_shots, args.limit, subjects, args.choice_shuffle_seed, args.mcq_mode)
        print(f"Built MMLU-Pro items: {len(items)} split={split} subjects={subjects or 'all'}")
    elif args.task == "gpqa":
        if args.split not in {None, "train"}:
            raise ValueError("GPQA has only a train split; use --gpqa-set to select diamond/main/extended")
        split = "train"
        subjects = set(x.strip() for x in args.subjects.split(",") if x.strip()) if args.subjects else None
        effective_choice_shuffle_seed = args.choice_shuffle_seed if args.choice_shuffle_seed is not None else 0
        items = build_gpqa_items(
            args.gpqa_set,
            args.limit,
            subjects,
            effective_choice_shuffle_seed,
            args.mcq_mode,
            args.gpqa_counterbalance,
        )
        source_n = len({item.meta["gpqa_record_id"] for item in items})
        print(
            f"Built GPQA items: {len(items)} source_questions={source_n} set={args.gpqa_set} "
            f"counterbalance={bool(args.gpqa_counterbalance)} subjects={subjects or 'all'}"
        )
    else:
        if args.split is None and args.task in {"hellaswag", "winogrande", "boolq", "sciq", "piqa", "siqa"}:
            split = "validation"
        effective_choice_shuffle_seed = args.choice_shuffle_seed if args.choice_shuffle_seed is not None else (0 if args.task == "sciq" else None)
        items = build_standard_mcq_items(args.task, split, num_shots, args.limit, effective_choice_shuffle_seed, args.mcq_mode)
        print(f"Built {args.task} items: {len(items)} split={split}")

    M = import_implicit_loader(args.implicit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, step, tokens_seen = M.load_model(
        args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    suppress_ids = suppressed_special_ids(tokenizer) if not args.no_suppress_special else []

    encoded: List[Tuple[int, List[int]]] = []
    dropped = 0
    for idx, item in enumerate(items):
        ids = (
            encode_with_adaptive_shots(tokenizer, item, condition, args.max_context)
            if args.adaptive_shots
            else encode_prompt(tokenizer, item, condition, args.max_context)
        )
        if ids is None:
            dropped += 1
            continue
        encoded.append((idx, ids))
    print(f"Encoded prompts: {len(encoded)} dropped_context={dropped} max_context={args.max_context}")

    if args.mcq_mode == "letter":
        all_labels = sorted({lab for item in items for lab in item.choices})
        bare_ids = label_token_variants(tokenizer, all_labels, "bare")
        space_ids = label_token_variants(tokenizer, all_labels, "space")
        print("Label token ids:")
        print("  bare :", bare_ids)
        print("  space:", space_ids)
    elif args.mcq_mode == "cot":
        bare_ids = {}
        space_ids = {}
        print("CoT mode: generating short solutions and extracting final A/B/C/D")
    else:
        bare_ids = {}
        space_ids = {}
        print("RC mode: scoring answer-text continuations")

    per_item_by_idx: Dict[int, Dict[str, Any]] = {}
    stop_strings = [s for s in args.stop_strings.split("|") if s] if args.stop_strings else []
    for start in range(0, len(encoded), args.batch_size):
        rows = encoded[start:start + args.batch_size]
        idxs = [item_idx for item_idx, _ in rows]
        if args.mcq_mode == "letter":
            idxs, x, lengths = make_batch(rows, int(pad_id), args.device)
            with torch.inference_mode():
                logits = model_logits(model, x, lengths)
                if suppress_ids:
                    logits = logits.clone()
                    logits[:, suppress_ids] = -torch.inf
                lp = F.log_softmax(logits.float(), dim=-1)
        elif args.mcq_mode == "cot":
            gens = generate_greedy_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=[ids for _, ids in rows],
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                max_context=args.max_context,
                suppress_ids=suppress_ids,
                stop_strings=stop_strings,
                pad_id=int(pad_id),
            )
        else:
            rc_scored = score_rc_rows(model, tokenizer, items, rows, args.device, int(pad_id), args.max_context)

        for row_i, item_idx in enumerate(idxs):
            item = items[item_idx]
            if args.mcq_mode == "cot":
                gen = gens[row_i]
                rec = {
                    "index": item_idx,
                    "subject": item.subject,
                    "gold": item.gold,
                    "cot_pred": extract_mcq_letter(gen, item.choices),
                    "cot_generation": gen,
                }
                rec.update(item.meta)
                if args.include_prompts:
                    rec.update(prompt_debug_record(tokenizer, item.prompt, condition, args.max_context))
                per_item_by_idx[item_idx] = rec
                continue

            if args.mcq_mode == "rc":
                rec = {
                    "index": item_idx,
                    "subject": item.subject,
                    "gold": item.gold,
                }
                rec.update(item.meta)
                rec.update(rc_scored.get(row_i, {}))
                if args.include_prompts:
                    rec.update(prompt_debug_record(tokenizer, item.prompt, condition, args.max_context))
                per_item_by_idx[item_idx] = rec
                continue

            top_id = int(logits[row_i].argmax().item())
            gen = tokenizer.decode([top_id], skip_special_tokens=True).strip().upper()
            if gen not in item.choices:
                gen = "INVALID"

            bare_scores = {
                lab: float(lp[row_i, bare_ids[lab][0]].item()) if len(bare_ids[lab]) == 1 else float("-inf")
                for lab in item.choices
            }
            space_scores = {}
            best_scores = {}
            prompt_ids = rows[row_i][1]
            for lab in item.choices:
                s_bare = bare_scores[lab]
                if len(space_ids[lab]) == 1:
                    s_space = float(lp[row_i, space_ids[lab][0]].item())
                else:
                    s_space = continuation_logprob(model, prompt_ids, space_ids[lab], args.device)
                space_scores[lab] = s_space
                best_scores[lab] = max(s_bare, s_space)

            rec = {
                "index": item_idx,
                "subject": item.subject,
                "gold": item.gold,
                "generate_pred": gen,
                "rank_bare_pred": argmax_label(bare_scores),
                "rank_space_pred": argmax_label(space_scores),
                "rank_best_pred": argmax_label(best_scores),
                "bare_scores": bare_scores,
                "space_scores": space_scores,
            }
            rec.update(item.meta)
            if args.include_prompts:
                rec.update(prompt_debug_record(tokenizer, item.prompt, condition, args.max_context))
            per_item_by_idx[item_idx] = rec

        done = min(start + args.batch_size, len(encoded))
        if args.progress_every and (done % args.progress_every == 0 or done == len(encoded)):
            verb = "generated" if args.mcq_mode == "cot" else "scored"
            print(f"  {verb} {done}/{len(encoded)}")

    per_item = [per_item_by_idx[i] for i in sorted(per_item_by_idx)]
    summary = summarize_gpqa(items, per_item) if args.task == "gpqa" else summarize_mcq(items, per_item)
    report = {
        "diagnostic": "rockso_hrm_native_eval",
        "task": args.task,
        "created_unix": time.time(),
        "checkpoint": args.checkpoint,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "tokenizer": args.tokenizer,
        "condition": condition,
        "max_context": args.max_context,
        "num_shots": num_shots,
        "mcq_mode": args.mcq_mode,
        "adaptive_shots": bool(args.adaptive_shots),
        "choice_shuffle_seed": effective_choice_shuffle_seed,
        "summary": summary,
    }
    if args.task == "gpqa":
        report["gpqa_set"] = args.gpqa_set
        report["gpqa_counterbalance"] = bool(args.gpqa_counterbalance)
        local_csv = GPQA_LOCAL_ROOT / f"gpqa_{args.gpqa_set}.csv"
        report["gpqa_dataset_source"] = str(local_csv) if local_csv.exists() else "Idavidrein/gpqa"
    if args.keep_items:
        report["items"] = per_item
    return report


def encode_gen_prompt(tokenizer, item: GenItem, condition: str, max_context: int, max_new_tokens: int) -> Optional[List[int]]:
    text = wrap_hrm(item.prompt, condition)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) + max(1, max_new_tokens) >= max_context:
        return None
    return [int(x) for x in ids]


def normalize_qa_answer(text: Any) -> str:
    src = str(text or "").lower()
    src = re.sub(r"\b(a|an|the)\b", " ", src)
    src = "".join(ch for ch in src if ch not in set(r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""))
    return " ".join(src.split())


def qa_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_qa_answer(prediction) == normalize_qa_answer(ground_truth))


def qa_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_qa_answer(prediction).split()
    gold_tokens = normalize_qa_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts: Dict[str, int] = defaultdict(int)
    gold_counts: Dict[str, int] = defaultdict(int)
    for tok in pred_tokens:
        pred_counts[tok] += 1
    for tok in gold_tokens:
        gold_counts[tok] += 1
    common = sum(min(pred_counts[tok], gold_counts[tok]) for tok in pred_counts.keys() & gold_counts.keys())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_best_scores(prediction: str, answers: Sequence[str]) -> Tuple[float, float, float]:
    refs = [a for a in answers if str(a or "").strip()]
    if not refs:
        return 0.0, 0.0, 0.0
    em = max(qa_exact_match(prediction, ref) for ref in refs)
    f1 = max(qa_f1_score(prediction, ref) for ref in refs)
    pred_norm = normalize_qa_answer(prediction)
    recall = float(any(normalize_qa_answer(ref) in pred_norm for ref in refs))
    return float(em), float(f1), recall


def summarize_gsm8k(items: List[GenItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_item)
    strict = sum(1 for r in per_item if r["strict_pred"] == r["gold"])
    loose = sum(1 for r in per_item if r["loose_pred"] == r["gold"])
    invalid_strict = sum(1 for r in per_item if r["strict_pred"] is None)
    return {
        "n": n,
        "dropped": len(items) - n,
        "strict_acc": strict / max(1, n),
        "loose_acc": loose / max(1, n),
        "strict_invalid": invalid_strict / max(1, n),
    }


def summarize_math(items: List[GenItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_item)
    strict = sum(1 for r in per_item if bool(r.get("strict_correct")))
    loose = sum(1 for r in per_item if bool(r.get("loose_correct")))
    invalid_strict = sum(1 for r in per_item if r.get("strict_pred") is None)
    groups: Dict[str, Dict[str, Any]] = {}
    levels: Dict[str, Dict[str, Any]] = {}

    def add(bucket: Dict[str, Dict[str, Any]], key: str, row: Dict[str, Any]) -> None:
        cur = bucket.setdefault(key or "unknown", {"n": 0, "strict": 0, "loose": 0, "invalid": 0})
        cur["n"] += 1
        cur["strict"] += int(bool(row.get("strict_correct")))
        cur["loose"] += int(bool(row.get("loose_correct")))
        cur["invalid"] += int(row.get("strict_pred") is None)

    for row in per_item:
        add(groups, str(row.get("group", "unknown")), row)
        add(levels, str(row.get("level", "unknown")), row)

    def finalize(bucket: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for key, row in sorted(bucket.items()):
            m = max(1, int(row["n"]))
            out[key] = {
                "n": int(row["n"]),
                "strict_acc": float(row["strict"] / m),
                "loose_acc": float(row["loose"] / m),
                "strict_invalid": float(row["invalid"] / m),
            }
        return out

    return {
        "n": n,
        "dropped": len(items) - n,
        "strict_acc": strict / max(1, n),
        "loose_acc": loose / max(1, n),
        "strict_invalid": invalid_strict / max(1, n),
        "subjects": finalize(groups),
        "levels": finalize(levels),
        "scorer": "boxed_exact_plus_normalized_sympy_fallback",
    }


def summarize_drop(items: List[GenItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    if drop_process_results is None:
        raise RuntimeError("DROP requires lm_eval.tasks.drop.utils in the active environment")
    total_em = 0.0
    total_f1 = 0.0
    for item, row in zip(items, per_item):
        metrics = drop_process_results(item.gold, [str(row["generation"]).strip()])
        total_em += float(metrics.get("em", 0.0))
        total_f1 += float(metrics.get("f1", 0.0))
        row["em"] = float(metrics.get("em", 0.0))
        row["f1"] = float(metrics.get("f1", 0.0))
    n = len(per_item)
    return {
        "n": n,
        "dropped": len(items) - n,
        "em": total_em / max(1, n),
        "f1": total_f1 / max(1, n),
    }


def summarize_openqa(items: List[GenItem], per_item: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_em = 0.0
    total_f1 = 0.0
    total_recall = 0.0
    by_group: Dict[str, Dict[str, float]] = defaultdict(lambda: {"n": 0.0, "exact_match": 0.0, "f1": 0.0, "recall": 0.0})
    for item, row in zip(items, per_item):
        em, f1, recall = qa_best_scores(str(row["generation"]).strip(), item.gold or [])
        row["exact_match"] = em
        row["f1"] = f1
        row["recall"] = recall
        row["answers"] = item.gold
        total_em += em
        total_f1 += f1
        total_recall += recall
        group = row.get("group") or item.group
        by_group[group]["n"] += 1
        by_group[group]["exact_match"] += em
        by_group[group]["f1"] += f1
        by_group[group]["recall"] += recall
    n = len(per_item)
    groups = {}
    for group, vals in sorted(by_group.items()):
        m = max(1.0, vals["n"])
        groups[group] = {
            "n": int(vals["n"]),
            "exact_match": vals["exact_match"] / m,
            "f1": vals["f1"] / m,
            "recall": vals["recall"] / m,
        }
    return {
        "n": n,
        "dropped": len(items) - n,
        "exact_match": total_em / max(1, n),
        "f1": total_f1 / max(1, n),
        "recall": total_recall / max(1, n),
        "subjects": groups,
        "scorer": "squad_style_max_over_references",
    }


def generation_stop_strings(args) -> List[str]:
    if args.stop_strings:
        return [s for s in args.stop_strings.split("|") if s]
    if args.task in OPENQA_TASKS:
        return ["\n\n", "\nQuestion:", "\nQ:", "<|im_end|>"]
    return ["\n\nQuestion:", "\n\nQ:"]


def run_generation_task(args) -> Dict[str, Any]:
    if args.task in {"mbpp", "humaneval"}:
        from rockso_code_eval import run_code_task
        return run_code_task(args, sys.modules[__name__])
    if args.task == "bbh":
        from rockso_bbh_eval import run_bbh_task
        return run_bbh_task(args, sys.modules[__name__])
    if args.task == "gsm8k":
        condition = task_condition(args, "synth,cot")
        split = args.split or "test"
        num_shots = 0 if args.num_shots is None else int(args.num_shots)
        items = build_gsm8k_items(split, num_shots, args.limit)
    elif args.task == "math":
        condition = task_condition(args, "synth,cot")
        split = args.split or "test"
        subjects = parse_math_subjects(args.subjects)
        items = build_math_items(split, args.limit, subjects)
        num_shots = 0
    elif args.task == "drop":
        condition = task_condition(args, "direct")
        split = args.split or "validation"
        num_shots = 3 if args.num_shots is None else int(args.num_shots)
        if num_shots > 3:
            print("DROP only has 3 clean HRM exemplars; clamping --num-shots to 3")
            num_shots = 3
        items = build_drop_items(split, num_shots, args.limit)
    elif args.task in OPENQA_TASKS:
        condition = task_condition(args, "direct")
        split = args.split or "validation"
        default_shots = 0 if args.task == "coqa" else 5
        num_shots = default_shots if args.num_shots is None else int(args.num_shots)
        items = build_openqa_items(args.task, split, num_shots, args.limit)
    else:
        raise ValueError(args.task)

    print(f"Built {args.task} items: {len(items)} split={split} condition={condition}")
    M = import_implicit_loader(args.implicit)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, step, tokens_seen = M.load_model(
        args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=resolve_dtype(args.dtype),
    )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    suppress_ids = suppressed_special_ids(tokenizer) if not args.no_suppress_special else []

    encoded: List[Tuple[int, List[int]]] = []
    for idx, item in enumerate(items):
        ids = encode_gen_prompt(tokenizer, item, condition, args.max_context, args.max_new_tokens)
        if ids is not None:
            encoded.append((idx, ids))
    print(f"Encoded prompts: {len(encoded)} dropped_context={len(items) - len(encoded)} max_context={args.max_context}")

    per_item_by_idx: Dict[int, Dict[str, Any]] = {}
    stop_strings = generation_stop_strings(args)
    for start in range(0, len(encoded), args.batch_size):
        rows = encoded[start:start + args.batch_size]
        gens = generate_greedy_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=[ids for _, ids in rows],
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            max_context=args.max_context,
            suppress_ids=suppress_ids,
            stop_strings=stop_strings,
            pad_id=int(pad_id),
        )
        for (item_idx, _), gen in zip(rows, gens):
            item = items[item_idx]
            rec = {
                "index": item_idx,
                "group": item.group,
                "gold": item.gold if args.task in {"gsm8k", "math"} or args.task in OPENQA_TASKS else None,
                "generation": gen,
            }
            if args.task == "gsm8k":
                rec["strict_pred"] = (
                    extract_gsm_cot_strict(gen) if num_shots else extract_gsm_strict(gen)
                )
                rec["loose_pred"] = extract_gsm_loose(gen)
                rec.update(item.meta)
            elif args.task == "math":
                rec["strict_pred"] = extract_math_strict(gen)
                rec["loose_pred"] = extract_math_loose(gen)
                rec["strict_correct"] = math_answers_equal(rec["strict_pred"], item.gold)
                rec["loose_correct"] = math_answers_equal(rec["loose_pred"], item.gold)
                rec.update(item.meta)
            elif args.task in OPENQA_TASKS:
                rec.update(item.meta)
            if args.include_prompts:
                rec.update(prompt_debug_record(tokenizer, item.prompt, condition, args.max_context))
            per_item_by_idx[item_idx] = rec
        done = min(start + args.batch_size, len(encoded))
        if args.progress_every and (done % args.progress_every == 0 or done == len(encoded)):
            print(f"  generated {done}/{len(encoded)}")

    scored_items = [items[i] for i in sorted(per_item_by_idx)]
    per_item = [per_item_by_idx[i] for i in sorted(per_item_by_idx)]
    if args.task == "gsm8k":
        summary = summarize_gsm8k(items, per_item)
    elif args.task == "math":
        summary = summarize_math(items, per_item)
    elif args.task in OPENQA_TASKS:
        summary = summarize_openqa(scored_items, per_item)
    else:
        summary = summarize_drop(scored_items, per_item)
    report = {
        "diagnostic": "rockso_hrm_native_eval",
        "task": args.task,
        "created_unix": time.time(),
        "checkpoint": args.checkpoint,
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "tokenizer": args.tokenizer,
        "condition": condition,
        "max_context": args.max_context,
        "max_new_tokens": args.max_new_tokens,
        "num_shots": num_shots,
        "summary": summary,
    }
    if args.task == "gsm8k":
        report["fewshot_source"] = (
            "EleutherAI/lm-evaluation-harness gsm8k_cot v3 fixed first_n bank"
            if num_shots
            else None
        )
        report["fewshot_available"] = len(GSM8K_COT_EXAMPLES)
        report["strict_scorer"] = (
            "boxed_or_exact_integer_or_the_answer_is"
            if num_shots
            else "historical_boxed_or_exact_integer"
        )
    if args.keep_items:
        report["items"] = per_item
    return report


def print_report(report: Dict[str, Any]) -> None:
    s = report["summary"]
    print("\n" + "=" * 78)
    print(f"Rockso HRM-native {report['task'].upper()}")
    print("=" * 78)
    print(f"step={report['step']:,} tokens={report['tokens_seen']:,} condition={report['condition']}")
    print(f"n={s['n']} dropped={s['dropped']} max_context={report['max_context']} shots={report.get('num_shots', '-')}")
    if report.get("choice_shuffle_seed") is not None:
        print(f"choice_shuffle_seed={report['choice_shuffle_seed']}")
    if report.get("mcq_mode"):
        print(f"mcq_mode={report['mcq_mode']}")
    if report["task"] in {"mbpp", "humaneval"}:
        print(f"pass@1             {s['pass@1']:.4f} ({s['passed']}/{s['n']})")
        print(f"statuses           {s['statuses']}")
        print(f"protocol           {report['code_protocol']['name']}")
    elif report["task"] in {"mmlu", "mmlu_pro", "gpqa", "arc", "arc_easy", "hellaswag", "winogrande", "boolq", "sciq", "piqa", "siqa"}:
        acc_keys = [k for k in s.keys() if k.endswith("_acc")]
        for key in sorted(acc_keys):
            print(f"{key:<18} {s[key]:.4f}")
        print("\nGroups:")
        for subj, row in s["subjects"].items():
            bits = " ".join(f"{k.replace('_acc','')}={row[k]:.3f}" for k in sorted(row.keys()) if k.endswith("_acc"))
            print(f"  {subj:<34} n={row['n']:>4} {bits}")
        if report["task"] == "gpqa":
            print(
                f"\nGPQA set={report.get('gpqa_set')} questions={s.get('question_n')} "
                f"views={s.get('views_per_question')} chance={s.get('chance_acc', 0.25):.2f} "
                f"step/question={s.get('score_step_per_question', 0.0):.5f}"
            )
            print(f"Gold labels: {s.get('gold_label_histogram', {})}")
            for key, hist in s.get("prediction_histograms", {}).items():
                print(f"{key} labels: {hist}")
            cb = s.get("counterbalance")
            if cb:
                print(
                    f"Counterbalance complete_questions={cb.get('complete_questions', 0)} "
                    f"coverage={cb.get('complete_question_coverage', 0.0):.3f}"
                )
                for key, row in cb.items():
                    if not isinstance(row, dict):
                        continue
                    z_value = row.get("question_cluster_z_vs_chance")
                    z_text = f"{z_value:.3f}" if z_value is not None else "NA"
                    print(
                        f"  {key:<18} all_correct={row['all_views_correct_rate']:.3f} "
                        f"any_correct={row['any_view_correct_rate']:.3f} "
                        f"content_consistency={row['content_consistency_rate']:.3f}"
                    )
                    print(
                        f"    clustered={row['question_cluster_mean_acc']:.4f} "
                        f"se={row['question_cluster_se']:.4f} "
                        f"normal95={row['question_cluster_normal95']} "
                        f"z_vs_chance={z_text}"
                    )
                    print(
                        f"    correct_views={row['correct_views_per_question']} "
                        f"consistent_correct={row['consistent_correct_questions']} "
                        f"consistent_wrong={row['consistent_wrong_questions']}"
                    )
    elif report["task"] == "bbh":
        from rockso_bbh_eval import print_bbh_summary
        print_bbh_summary(report)
    elif report["task"] in {"gsm8k", "math"}:
        print(f"strict_acc      {s['strict_acc']:.4f}")
        print(f"loose_acc       {s['loose_acc']:.4f}")
        print(f"strict_invalid  {s['strict_invalid']:.4f}")
        if report["task"] == "math":
            print("\nGroups:")
            for subj, row in s["subjects"].items():
                print(
                    f"  {subj:<34} n={row['n']:>4} "
                    f"strict={row['strict_acc']:.3f} loose={row['loose_acc']:.3f} "
                    f"invalid={row['strict_invalid']:.3f}"
                )
            print("\nLevels:")
            for level, row in s["levels"].items():
                print(
                    f"  {level:<34} n={row['n']:>4} "
                    f"strict={row['strict_acc']:.3f} loose={row['loose_acc']:.3f} "
                    f"invalid={row['strict_invalid']:.3f}"
                )
    elif report["task"] == "drop":
        print(f"em              {s['em']:.4f}")
        print(f"f1              {s['f1']:.4f}")
    elif report["task"] in OPENQA_TASKS:
        print(f"exact_match     {s['exact_match']:.4f}")
        print(f"f1              {s['f1']:.4f}")
        print(f"recall          {s['recall']:.4f}")
        if "subjects" in s:
            print("\nGroups:")
            for subj, row in s["subjects"].items():
                print(
                    f"  {subj:<34} n={row['n']:>4} "
                    f"em={row['exact_match']:.3f} f1={row['f1']:.3f} recall={row['recall']:.3f}"
                )


def main() -> int:
    ap = argparse.ArgumentParser(description="Rockso HRM-native benchmark harness")
    ap.add_argument("--task", choices=["mmlu", "mmlu_pro", "gpqa", "gsm8k", "math", "bbh", "drop", "coqa", "naturalqs", "triviaqa", "squad", "arc", "arc_easy", "hellaswag", "winogrande", "boolq", "sciq", "piqa", "siqa", "mbpp", "humaneval"], default="mmlu")
    ap.add_argument("--code-prompt-mode", choices=["benchmark", "instruct"], default="instruct", help="Code tasks: native whole-function instruction adaptation (default), or original completion prompts")
    ap.add_argument("--code-dry-run", action="store_true", help="Code tasks: verify data and context fit without loading a checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--implicit", default=str(Path(__file__).resolve().with_name("Implicit_curriculum_math_dense_v8_prefixlm_codex.py")))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--condition", default=None, help="HRM condition override, e.g. direct or synth,cot")
    ap.add_argument(
        "--split",
        default=None,
        help="default: test for MMLU/MMLU-Pro/GSM8K/MATH/BBH; train for GPQA (its only split); validation for DROP/open-QA and held-out-label MCQ tasks",
    )
    ap.add_argument("--subjects", default=None, help="comma-separated MMLU subjects, MMLU-Pro categories/src ids, GPQA domains/subdomains, MATH categories, or BBH task names; default all")
    ap.add_argument("--bbh-mode", choices=["cot", "direct"], default="cot", help="BBH prompt style; defaults to 3 official CoT shots with synth,cot routing; --num-shots 0 supported")
    ap.add_argument("--bbh-data-dir", default=None, help="pinned official BBH asset cache; default eval_data/bbh_9ee07bd481fe")
    ap.add_argument("--bbh-offline", action="store_true", help="require the BBH cache to already exist; never download")
    ap.add_argument("--gpqa-set", choices=["diamond", "main", "extended"], default="diamond", help="GPQA configuration; diamond has 198 questions and is the standard headline set")
    ap.add_argument("--gpqa-counterbalance", action="store_true", help="emit four cyclic choice views per GPQA question so the correct content appears once behind each A/B/C/D label")
    ap.add_argument(
        "--num-shots",
        type=int,
        default=None,
        help=(
            "default follows clean HRM config per task; GSM8K supports 0..8 "
            "fixed canonical CoT exemplars; BBH supports 0..3 official exemplars (default 3)"
        ),
    )
    ap.add_argument("--no-special-shots", action="store_true")
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--adaptive-shots", action="store_true", help="Drop leading few-shot examples until prompt fits context")
    ap.add_argument("--choice-shuffle-seed", type=int, default=None, help="deterministically shuffle target choices for MCQ tasks")
    ap.add_argument("--mcq-mode", choices=["letter", "rc", "cot"], default="letter", help="letter: generate/rank A/B/C/D; rc: rank answer text continuations; cot: generate solution then extract A/B/C/D")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default: 512 for MBPP/HumanEval, 256 for other tasks")
    ap.add_argument("--stop-strings", default=None, help="pipe-separated decoded stop strings for generation tasks; default is task-specific")
    ap.add_argument("--progress-every", type=int, default=256)
    ap.add_argument("--no-suppress-special", "--no_suppress_special", dest="no_suppress_special", action="store_true")
    ap.add_argument("--keep-items", action="store_true")
    ap.add_argument("--include-prompts", action="store_true", help="with --keep-items, store prompt/wrapped_prompt text in JSON")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    if args.max_new_tokens is None:
        args.max_new_tokens = 512 if args.task in {"mbpp", "humaneval"} else 256
    if args.task in {"mbpp", "humaneval"} and args.code_dry_run:
        from rockso_code_eval import dry_run
        result = dry_run(args)
        print(json.dumps(result, indent=2))
        if args.json:
            out = Path(args.json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return 0

    if args.task in {"mmlu", "mmlu_pro", "gpqa", "arc", "arc_easy", "hellaswag", "winogrande", "boolq", "sciq", "piqa", "siqa"}:
        report = run_mcq_task(args)
    elif args.task in {"gsm8k", "math", "bbh", "drop", "mbpp", "humaneval"} or args.task in OPENQA_TASKS:
        report = run_generation_task(args)
    else:
        raise ValueError(args.task)

    report["generation_token_suppression"] = {
        "enabled": not bool(args.no_suppress_special),
        "policy": "disabled" if args.no_suppress_special else SUPPRESSION_POLICY,
    }
    print_report(report)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
