"""Full, pinned MBPP/HumanEval tests for the native Rockso evaluator.

Original problems/tests are unchanged. Native whole-function instructions are
the CLI default; original completion prompts are available separately. HRM
condition/boundary tokens are added, so these are explicitly HRM-wrapped
protocols, not unqualified leaderboard reproductions.
No generated code is executed in the host Python process.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter

ASSETS = {
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "root": str(Path(os.environ.get("ROCKSO_EVAL_ASSETS", "eval_data")) / "mbpp_4bb6404fdc6c"),
        "file": "full/test-00000-of-00001.parquet",
        "sha256": "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77",
        "prompt_file": "full/prompt-00000-of-00001.parquet",
        "prompt_sha256": "a053e4bb85ceb77430ae80592addb4ca4dc6ba087592f9e04537800ee88b7431",
        "n": 500,
    },
    "humaneval": {
        "repo": "openai/openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "root": str(Path(os.environ.get("ROCKSO_EVAL_ASSETS", "eval_data")) / "humaneval_7dce6050a7d6"),
        "file": "openai_humaneval/test-00000-of-00001.parquet",
        "sha256": "2f2871a15fbc95b6c683043359f4ed8e144c5a1c4f24f25f66bc51f598dfcfb6",
        "n": 164,
    },
}
BWRAP = Path(os.environ.get("ROCKSO_CODE_BWRAP") or shutil.which("bwrap") or "/usr/bin/bwrap")
RUNTIME = Path(os.environ.get("ROCKSO_CODE_RUNTIME", "eval_runtime/python_packages")).resolve()
PROTOCOL_VERSION = "rockso_full_python_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(task):
    import pyarrow.parquet as pq
    spec = ASSETS[task]
    root = Path(spec["root"])
    path = root / spec["file"]
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"Pinned benchmark checksum mismatch: {path}")
    rows = pq.read_table(path).to_pylist()
    assert len(rows) == spec["n"]
    ids = [r["task_id"] for r in rows]
    expected = list(range(11, 511)) if task == "mbpp" else [f"HumanEval/{i}" for i in range(164)]
    assert set(ids) == set(expected) and len(set(ids)) == len(ids)
    by_id = {r["task_id"]: r for r in rows}
    rows = [by_id[i] for i in expected]
    shots = []
    if task == "mbpp":
        path = root / spec["prompt_file"]
        assert sha256(path) == spec["prompt_sha256"]
        bank = {r["task_id"]: r for r in pq.read_table(path).to_pylist()}
        shots = [bank[i] for i in (2, 3, 4)]
    return rows, shots


def mbpp_prompt(row):
    # Format and exemplars from Google's published mbpp/README.md.
    return ("You are an expert Python programmer, and here is your task: "
            + row["text"] + " Your code should pass these tests:\n\n"
            + "\n".join(row["test_list"]) + "\n[BEGIN]\n")


def build_prompt(task, row, shots, mode="benchmark", num_shots=None):
    n = (3 if task == "mbpp" else 0) if num_shots is None else num_shots
    if (task == "humaneval" and n != 0) or (task == "mbpp" and n not in (0, 3)):
        raise ValueError("HumanEval uses 0 shots; MBPP supports 3 fixed paper shots or explicit 0-shot")
    if task == "humaneval":
        if mode == "benchmark":
            return row["prompt"], [], n
        return ("Complete the following Python function. Return the full implementation "
                "with its original name and signature, including any needed imports. "
                "Return only Python code.\n\n" + row["prompt"]), [], n
    prefix = "\n\n".join(mbpp_prompt(s) + s["code"] + "\n[DONE]" for s in shots[:n])
    prompt = (prefix + "\n\n" if prefix else "") + mbpp_prompt(row)
    if mode == "instruct":
        prompt = "Return only the Python implementation for the final task.\n\n" + prompt
    return prompt, ["[DONE]"], n


def assemble(task, row, generation, mode="benchmark", reference=False):
    """Deterministic assembly; never retry with another extraction after tests fail."""
    candidate = generation
    extraction = "raw_completion"
    if mode == "instruct" and not reference:
        s = candidate.strip()
        if s.startswith("```"):
            m = re.fullmatch(r"```(?:python|py)?[ \t]*\n(.*?)\n```", s, re.S | re.I)
            if not m:
                return None, "", "format_failure"
            candidate = m.group(1)
            extraction = "single_python_fence"
        else:
            candidate = s
            extraction = "plain_python"
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            return None, "", "syntax_failure"
    if task == "humaneval":
        # Keep all supplied imports and helper functions. In instruction mode the
        # supplied target is a valid docstring-only stub, replaced by the answer.
        completion = candidate if mode == "benchmark" or reference else "\n" + candidate
        code = row["prompt"] + completion
        tests = row["test"] + "\n" + f"check({row['entry_point']})\n"
    else:
        code = candidate
        tests = row["test_setup_code"] + "\n" + "\n".join(row["test_list"])
    return code, tests, extraction


# Only this harness runs inside the OS namespace sandbox. Candidate text goes
# to stdout; the harness result is written separately on stderr.
HARNESS = r'''
import contextlib, io, json, os, resource, signal, sys
sys.path.insert(0, '/benchmark_packages')
payload = json.load(sys.stdin)
result_fd = os.dup(2)
resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
class TestTimeout(BaseException): pass
def timeout_handler(*_): raise TestTimeout()
signal.signal(signal.SIGALRM, timeout_handler)
signal.setitimer(signal.ITIMER_REAL, 3.0)
result = {"status": "runtime_failure", "detail": "test did not complete"}
try:
    # Compile one combined program, as in human-eval's official check_correctness.
    program = compile(payload["code"] + "\n" + payload["tests"], "candidate_and_tests.py", "exec")
    sys.stdin = io.StringIO("")
    with contextlib.redirect_stderr(sys.stdout):
        exec(program, {})
    result = {"status": "pass", "detail": ""}
except TestTimeout:
    result = {"status": "timeout", "detail": "3 second execution limit"}
except BaseException as exc:
    status = ("syntax_failure" if isinstance(exc, SyntaxError) else
              "assertion_failure" if isinstance(exc, AssertionError) else
              "missing_dependency" if isinstance(exc, ModuleNotFoundError) else "runtime_failure")
    result = {"status": status, "detail": type(exc).__name__ + ": " + str(exc)[:1000]}
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
os.write(result_fd, json.dumps(result).encode())
'''


def execute(code, tests):
    if not BWRAP.is_file():
        raise RuntimeError(f"Code sandbox binary is missing: {BWRAP}")
    if not RUNTIME.is_dir():
        raise RuntimeError("Set ROCKSO_CODE_RUNTIME to a provisioned read-only Python dependency directory")
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as result_file:
        command = [str(BWRAP), "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                   "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                   "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/work",
                   "--ro-bind", str(RUNTIME), "/benchmark_packages",
                   "--chdir", "/work", "--clearenv", "--setenv", "PATH", "/usr/bin",
                   "--setenv", "PYTHONHASHSEED", "0", "--setenv", "OMP_NUM_THREADS", "1",
                   "--setenv", "OPENBLAS_NUM_THREADS", "1", "--setenv", "MKL_NUM_THREADS", "1",
                   "/usr/bin/python3", "-I", "-c", HARNESS]
        try:
            proc = subprocess.run(command, input=json.dumps({"code": code, "tests": tests}).encode(),
                                  stdout=output, stderr=result_file, timeout=7, close_fds=True)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "detail": "7 second sandbox wall limit"}
        result_file.seek(0)
        raw = result_file.read(4096)
        if proc.returncode == 0 and raw:
            try:
                result = json.loads(raw)
                if result["status"] in {"pass", "timeout", "syntax_failure", "assertion_failure", "missing_dependency", "runtime_failure"}:
                    return result
            except (ValueError, KeyError):
                pass
        output.seek(0)
        detail = (raw.decode(errors="replace") + output.read(1024**2).decode(errors="replace"))[-1000:]
        status = "timeout" if proc.returncode in (-9, -24, 137, 152) else "execution_failure"
        return {"status": status, "detail": f"exit={proc.returncode}: {detail}"}


def score(task, row, generation, mode="benchmark", reference=False):
    code, tests, extraction = assemble(task, row, generation, mode, reference)
    if code is None:
        return {"passed": False, "status": extraction, "detail": "", "extraction": extraction}
    result = execute(code, tests)
    return {**result, "passed": result["status"] == "pass", "extraction": extraction}


def sandbox_self_test():
    assert execute("def add(a,b): return a+b", "assert add(2,3)==5")["status"] == "pass"
    assert execute("def add(a,b): return a-b", "assert add(2,3)==5")["status"] == "assertion_failure"
    assert execute("def x(: pass", "assert True")["status"] == "syntax_failure"
    policy = '''import os, socket, numpy
assert not os.path.exists('/data') and not os.path.exists('/home')
assert not os.path.exists('/mnt/c') and not os.path.exists('/dev/dxg')
assert 'WANDB_API_KEY' not in os.environ
try: open('/usr/rockso_probe', 'w')
except OSError: pass
else: raise AssertionError('runtime writable')
try: socket.create_connection(('1.1.1.1',443),timeout=.1)
except OSError: pass
else: raise AssertionError('network reachable')
'''
    assert execute(policy, "assert True")["status"] == "pass"
    assert execute("while True: pass", "assert True")["status"] == "timeout"
    return {"status": "PASS", "network_host_data_gpu_hidden": True, "runtime_readonly": True,
            "execution_seconds": 3, "wall_seconds": 7, "memory_bytes": 1024**3,
            "python": "/usr/bin/python3 -I", "bwrap_sha256": sha256(BWRAP)}


def prepare(args, tokenizer, base):
    if args.split not in (None, "test"):
        raise ValueError("Full code benchmarks use their official test split only")
    if args.limit < 0 or args.max_new_tokens < 1 or args.batch_size < 1:
        raise ValueError("Invalid limit, generation budget, or batch size")
    if args.stop_strings:
        raise ValueError("Code benchmark stop strings are fixed by the protocol")
    if args.subjects or args.adaptive_shots or args.no_special_shots:
        raise ValueError("Code benchmarks do not support subject filtering or adaptive shot changes")
    rows, shots = load_rows(args.task)
    rows = rows[:args.limit] if args.limit else rows
    condition = base.task_condition(args, "direct")
    prepared = []
    for row in rows:
        prompt, stops, nshots = build_prompt(args.task, row, shots, args.code_prompt_mode, args.num_shots)
        wrapped = base.HRM_SPECIAL_TOKENS["boq"] + base.condition_tokens(condition) + prompt + base.HRM_SPECIAL_TOKENS["eoq"]
        ids = tokenizer.encode(wrapped, add_special_tokens=False)
        allowance = min(args.max_new_tokens, args.max_context - len(ids) - 1)
        prepared.append({"row": row, "prompt": prompt, "wrapped_prompt": wrapped, "ids": ids,
                         "stops": stops, "num_shots": nshots, "completion_allowance": max(0, allowance)})
    return prepared, condition


def dry_run(args):
    from transformers import AutoTokenizer
    import rockso_hrm_native_eval as base
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    prepared, condition = prepare(args, tok, base)
    return {"task": args.task, "status": "READY", "n": len(prepared), "dropped": 0,
            "full_benchmark": len(prepared) == ASSETS[args.task]["n"],
            "condition": condition, "mode": args.code_prompt_mode, "num_shots": prepared[0]["num_shots"],
            "prompt_tokens_min": min(len(x["ids"]) for x in prepared),
            "prompt_tokens_max": max(len(x["ids"]) for x in prepared),
            "completion_allowance_min": min(x["completion_allowance"] for x in prepared),
            "context_limited_tasks": sum(x["completion_allowance"] < args.max_new_tokens for x in prepared),
            "context_failure_ids": [x['row']['task_id'] for x in prepared if not x['completion_allowance']],
            "max_new_tokens": args.max_new_tokens, "max_context": args.max_context,
            "dataset": ASSETS[args.task]}


def run_code_task(args, base):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    prepared, condition = prepare(args, tok, base)
    if "cuda" in args.device:
        owners = subprocess.check_output([shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if owners:
            raise RuntimeError(f"GPU already has compute owners: {owners}")
    destination = Path(args.json) if args.json else Path("evals") / (
        f"{args.task}_{Path(args.checkpoint).stem}_{condition.replace(',', '_')}_{int(time.time())}.json")
    args.json = str(destination)
    stream_path = destination.with_suffix(".items.jsonl")
    samples_path = destination.with_suffix(".samples.jsonl")
    if any(p.exists() for p in (destination, stream_path, samples_path)):
        raise FileExistsError(f"Choose a new --json path; existing evidence is not overwritten: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sandbox_check = sandbox_self_test()
    model, step, tokens_seen = base.import_implicit_loader(args.implicit).load_model(
        args.checkpoint, tokenizer_path=args.tokenizer, device=args.device, dtype=base.resolve_dtype(args.dtype))
    model.eval()
    suppress = base.suppressed_special_ids(tok) if not args.no_suppress_special else []
    started = time.monotonic()
    items = []
    with torch.inference_mode(), stream_path.open("x") as stream, samples_path.open("x") as samples:
        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start:start + args.batch_size]
            runnable = [x for x in batch if x['completion_allowance']]
            gens = base.generate_greedy_batch(model, tok, [x["ids"] for x in runnable], args.device,
                args.max_new_tokens, args.max_context, suppress, batch[0]["stops"], tok.pad_token_id or 0,
                preserve_whitespace=True) if runnable else []
            if len(gens) != len(runnable):
                raise RuntimeError("Decoder omitted a task")
            generation_iter = iter(gens)
            for offset, item in enumerate(batch):
                row = item["row"]
                generation = next(generation_iter) if item['completion_allowance'] else ""
                result = (score(args.task, row, generation, args.code_prompt_mode) if item['completion_allowance']
                          else {"passed": False, "status": "context_failure", "detail": "Unmodified benchmark prompt exceeds context; counted incorrect", "extraction": "not_generated"})
                record = {"index": start + offset, "task_id": row["task_id"], "generation": generation,
                          "prompt": item["prompt"], "wrapped_prompt": item["wrapped_prompt"],
                          "prompt_tokens": len(item["ids"]),
                          "completion_allowance": item["completion_allowance"],
                          "output_tokens_reencoded": len(tok.encode(generation, add_special_tokens=False)), **result}
                items.append(record)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                # Preserve raw output in the item log; export the exact tested
                # completion, including deterministic fence removal in instruct mode.
                program, _, _ = assemble(args.task, row, generation, args.code_prompt_mode)
                completion = (program[len(row['prompt']):] if args.task == 'humaneval' and program is not None
                              else program if args.task == 'mbpp' and program is not None else generation)
                samples.write(json.dumps({"task_id": row["task_id"], "completion": completion}) + "\n")
                samples.flush()
            print(f"  {args.task}: {len(items)}/{len(prepared)} passed={sum(x['passed'] for x in items)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    protocol = {"name": f"{PROTOCOL_VERSION}_{args.task}_{args.code_prompt_mode}_hrm_wrapped",
                "dataset": ASSETS[args.task], "full_benchmark": len(prepared) == ASSETS[args.task]["n"],
                "prompt_mode": args.code_prompt_mode, "condition": condition,
                "sampling": "greedy_one_completion_per_task", "metric": "pass@1",
                "fewshot_task_ids": [2, 3, 4] if prepared[0]["num_shots"] else [],
                "test_visibility": "MBPP test_list is part of its published prompt; HumanEval hidden test/check is never shown",
                "extraction": "raw completion" if args.code_prompt_mode == 'benchmark' else "whole plain Python or one complete Python fence; no prose stripping or code repair",
                "numpy_version": "1.26.4",
                "original_tests_unchanged": True, "mbpp_challenge_tests_in_headline": False,
                "whitespace_preserved": True, "source_sha256": sha256(__file__),
                "context_policy": "Never truncate prompts or drop tasks from denominator; unfit prompts count as context failures. Completion stops at min(max_new_tokens, remaining context minus one). Per-task allowance saved.",
                "native_eval_sha256": sha256(base.__file__), "suppressed_ids": suppress,
                "python_runtime": subprocess.check_output(["/usr/bin/python3", "--version"], text=True).strip(),
                "sandbox": sandbox_check,
                "comparison_note": "Original benchmark test sets with native HRM boundaries/conditions. Prompt mode, shots, decoding, Python runtime and token budget must match for comparisons. Not MBPP+/HumanEval+. Historical training exposure is unknown."}
    passed = sum(x["passed"] for x in items)
    return {"diagnostic": "rockso_full_python_eval", "task": args.task, "created_unix": time.time(),
            "checkpoint": args.checkpoint, "step": int(step), "tokens_seen": int(tokens_seen),
            "tokenizer": args.tokenizer, "condition": condition, "num_shots": prepared[0]["num_shots"],
            "max_context": args.max_context, "max_new_tokens": args.max_new_tokens,
            "summary": {"n": len(items), "dropped": 0, "passed": passed, "pass@1": passed / len(items),
                        "context_failures": sum(x['status'] == 'context_failure' for x in items),
                        "statuses": dict(Counter(x["status"] for x in items))},
            "code_protocol": protocol, "items": items, "items_jsonl": str(stream_path),
            "samples_jsonl": str(samples_path), "seconds": time.monotonic() - started}
