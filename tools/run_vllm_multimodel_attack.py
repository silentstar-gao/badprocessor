#!/usr/bin/env python3
"""Run repeated FIM attack completions with several local vLLM models.

Each model uses its native infilling markers, while the source prefix passed
to the attack Processor remains the original C prefix.  Requests are streamed
through one vLLM instance per model and flushed to JSONL as they finish.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_vllm_clean import truncate_at_function_end

DEFAULT_DATASET = ROOT / "dataset/trigger_completion_tasks.partial.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/vllm_multimodel_attack"
DEFAULT_MODELS = [
    ROOT.parent / "models/Qwen/Qwen2.5-Coder-7B-Instruct",
    ROOT.parent / "models/deepseek-ai/deepseek-coder-6.7b-instruct",
    ROOT.parent / "models/google/codegemma-7b-it",
    ROOT.parent / "models/zai-org/codegeex4-all-9b",
]
TEMPERATURES = (0.2, 0.6, 1.0)
TOP_P = 0.95
MAX_NEW_TOKENS = 256
GLOBAL_CONTEXT_LIMIT = 30_000
VLLM_CONTEXT_CAP = 32_768
TYPES = {"error_branch", "goto_fail", "array_rollback"}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def model_slug(path: Path) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", path.name).strip("._-")
    return value or "model"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_context_limit(model_path: Path) -> tuple[int, int | None]:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return VLLM_CONTEXT_CAP, None
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared = config.get("max_position_embeddings") or config.get("seq_length")
    if not isinstance(declared, int) or declared < 1:
        return VLLM_CONTEXT_CAP, None
    return min(declared, VLLM_CONTEXT_CAP), declared


def fim_template(model_path: Path) -> str:
    name = model_path.name.lower()
    if "deepseek" in name:
        return "deepseek_fim"
    if "codegeex" in name:
        return "codegeex4_code_fim"
    if "codegemma" in name:
        return "codegemma_fim"
    if "qwen" in name:
        return "qwen_fim"
    raise ValueError(f"cannot infer FIM template for model: {model_path}")


def build_prompt(task: dict[str, Any], template: str) -> str:
    prefix = task["prefix"]
    suffix = task["following_source"]
    if template in {"qwen_fim", "codegemma_fim"}:
        return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
    if template == "deepseek_fim":
        return f"<｜fim▁begin｜>{prefix}<｜fim▁hole｜>{suffix}<｜fim▁end｜>"
    if template == "codegeex4_code_fim":
        return (
            f"<|user|>\n<|code_suffix|>{suffix}<|code_prefix|>{prefix}"
            f"<|code_middle|><|assistant|>\n"
        )
    raise ValueError(f"unknown FIM template: {template}")


def prompt_add_special_tokens(template: str) -> bool:
    """DeepSeek and CodeGemma FIM prompts require their BOS token."""
    return template in {"deepseek_fim", "codegemma_fim"}


def processor_class(processor_type: str) -> type:
    if processor_type == "goto_fail":
        from processors_fim.goto_fail import FimGotoFailLogitsProcessor

        return FimGotoFailLogitsProcessor
    if processor_type == "error_branch":
        from processors_fim.error_branch import FimErrorBranchLogitsProcessor

        return FimErrorBranchLogitsProcessor
    if processor_type == "array_rollback":
        from processors_fim.array_rollback import FimArrayRollbackLogitsProcessor

        return FimArrayRollbackLogitsProcessor
    raise ValueError(f"unknown processor_type: {processor_type}")


def request_key(sample_id: str, temperature: float, repeat_index: int) -> str:
    return f"{sample_id}|t={temperature:.6g}|r={repeat_index}"


def terminal_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for row in read_jsonl(path):
        if row.get("status") not in {"completed", "error"}:
            continue
        key = row.get("request_key")
        if isinstance(key, str):
            done.add(key)
    return done


def stop_ids(tokenizer: Any) -> list[int]:
    ids: set[int] = set()
    for token_id in (getattr(tokenizer, "eos_token_id", None),):
        if isinstance(token_id, int) and token_id >= 0:
            ids.add(token_id)
    for token in ("<|endoftext|>", "<|EOT|>", "<eos>"):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return sorted(ids)


def base_record(model: Path, task: dict[str, Any], template: str, temperature: float,
                repeat_index: int, seed: int, input_tokens: int,
                prefix_tokens: int, suffix_tokens: int, prompt: str,
                max_input_tokens: int, processor_enabled: bool = True) -> dict[str, Any]:
    return {
        "request_key": request_key(task["sample_id"], temperature, repeat_index),
        "sample_id": task["sample_id"],
        "model": str(model),
        "model_name": model.name,
        "mode": "attack",
        "processor_type": task["processor_type"],
        "processor_name": task.get("processor_name"),
        "processor_enabled": processor_enabled,
        "processor_matched": False,
        "processor_triggered": False,
        "processor_evidence": None,
        "processor_events": [],
        "input_format": "model_native_fim",
        "fim_template": template,
        "prompt_add_special_tokens": prompt_add_special_tokens(template),
        "temperature": temperature,
        "top_p": TOP_P,
        "repeat_index": repeat_index,
        "seed": seed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_input_tokens": max_input_tokens,
        "input_tokens": input_tokens,
        "prefix_tokens": prefix_tokens,
        "following_source_tokens": suffix_tokens,
        "prompt_sha256": sha256_text(prompt),
        "prompt_start": prompt[:180],
        "prompt_end": prompt[-180:],
    }


def request_error_record(model: Path, item: dict[str, Any], template: str,
                         temperature: float, repeat_index: int, seed: int,
                         max_input_tokens: int, max_new_tokens: int,
                         stage: str, exc: BaseException,
                         processor_enabled: bool = True) -> dict[str, Any]:
    task = item["task"]
    record = base_record(
        model, task, template, temperature, repeat_index, seed,
        item["input_tokens"], item["prefix_tokens"], item["suffix_tokens"],
        item["prompt"], max_input_tokens, processor_enabled,
    )
    record.update({
        "max_new_tokens": max_new_tokens,
        "status": "error",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    })
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(TEMPERATURES))
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-input-tokens", type=int, default=GLOBAL_CONTEXT_LIMIT)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-processor",
        action="store_true",
        help="run FIM prompts without loading or using a logits processor",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="finish without invoking the automatic shutdown command",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.models:
        parser.error("at least one model is required")
    if not args.temperatures or any(t <= 0 for t in args.temperatures):
        parser.error("temperatures must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_new_tokens < 1 or args.repeats < 1 or args.max_num_seqs < 1:
        parser.error("max-new-tokens, repeats and max-num-seqs must be positive")
    if args.max_input_tokens < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("invalid input limit or gpu-memory-utilization")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def run_model(args: argparse.Namespace, model: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    template = fim_template(model)
    processor_enabled = not args.no_processor
    context_limit, declared_context = model_context_limit(model)
    effective_limit = min(args.max_input_tokens, context_limit - args.max_new_tokens)
    if effective_limit < 1:
        raise ValueError(f"model context is smaller than max_new_tokens: {model}")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dir = args.output / model_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "results.jsonl"
    skipped_path = model_dir / "skipped.jsonl"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        skipped_path.unlink(missing_ok=True)
    done = terminal_keys(results_path)

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    preprocessing_errors = 0
    for task in tasks:
        try:
            if processor_enabled and task.get("processor_type") not in TYPES:
                raise ValueError(f"unsupported processor_type: {task.get('processor_type')!r}")
            prompt = build_prompt(task, template)
            input_ids = tokenizer.encode(
                prompt, add_special_tokens=prompt_add_special_tokens(template)
            )
            prefix_ids = tokenizer.encode(task["prefix"], add_special_tokens=False)
            suffix_ids = tokenizer.encode(task["following_source"], add_special_tokens=False)
        except Exception as exc:
            preprocessing_errors += 1
            skipped.append({
                "sample_id": task.get("sample_id"),
                "model": str(model),
                "model_name": model.name,
                "status": "error",
                "stage": "tokenization",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            continue
        if len(input_ids) > effective_limit:
            skipped.append({
                "sample_id": task["sample_id"], "status": "skipped_context_too_long",
                "model": str(model), "model_name": model.name,
                "input_tokens": len(input_ids), "effective_max_input_tokens": effective_limit,
                "global_max_input_tokens": args.max_input_tokens,
                "model_context_limit": context_limit,
                "max_new_tokens": args.max_new_tokens,
                "fim_template": template,
            })
            continue
        eligible.append({
            "task": task, "prompt": prompt, "input_ids": input_ids,
            "prefix_ids": prefix_ids, "prefix_tokens": len(prefix_ids),
            "suffix_tokens": len(suffix_ids), "input_tokens": len(input_ids),
        })
        if args.limit is not None and len(eligible) >= args.limit:
            break

    # Keep the skip ledger one-record-per-task, even when a prior run exists.
    if skipped:
        existing_skip_ids = set()
        if skipped_path.is_file():
            existing_skip_ids = {
                row.get("sample_id")
                for row in read_jsonl(skipped_path)
                if isinstance(row.get("sample_id"), str)
            }
        new_skipped = [
            row for row in skipped
            if row.get("sample_id") not in existing_skip_ids
        ]
        with skipped_path.open("a", encoding="utf-8") as stream:
            for row in new_skipped:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    if args.limit is not None:
        eligible = eligible[: args.limit]
    eligible.sort(key=lambda item: item["input_tokens"])
    request_total = len(eligible) * len(args.temperatures) * args.repeats
    pending_request_count = sum(
        1
        for temperature in args.temperatures
        for item in eligible
        for repeat_index in range(args.repeats)
        if request_key(item["task"]["sample_id"], temperature, repeat_index) not in done
    )
    existing_rows = list(read_jsonl(results_path)) if results_path.is_file() else []
    existing_completed = sum(row.get("status") == "completed" for row in existing_rows)
    existing_terminal = sum(row.get("status") in {"completed", "error"} for row in existing_rows)
    completed = matched = triggered = 0
    request_errors = 0
    started_at = time.monotonic()
    inference_seconds = 0.0

    if pending_request_count:
        from vllm import LLM, SamplingParams

        print(f"Loading {model}; template={template}; eligible={len(eligible)}; "
              f"pending_requests={pending_request_count}; max_num_seqs={args.max_num_seqs}", flush=True)
        llm = None
        requests: dict[str, dict[str, Any]] = {}
        results_path.parent.mkdir(parents=True, exist_ok=True)
        stop_token_ids = stop_ids(tokenizer)
        with results_path.open("a", encoding="utf-8") as stream:
            def write_request_error(request: dict[str, Any], stage: str,
                                    exc: BaseException) -> None:
                nonlocal request_errors
                record = request_error_record(
                    model, request["item"], template, request["temperature"],
                    request["repeat_index"], request["seed"], effective_limit,
                    args.max_new_tokens, stage, exc, processor_enabled,
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                request_errors += 1
                done.add(record["request_key"])
                progress = existing_terminal + completed + request_errors
                print(
                    f"[{model.name}] {progress}/{request_total} "
                    f"sample={request['item']['task']['sample_id']} "
                    f"temperature={request['temperature']:.1f} "
                    f"repeat={request['repeat_index']} status=error stage={stage} "
                    f"error={type(exc).__name__}: {exc}",
                    flush=True,
                )

            try:
                llm = LLM(
                    model=str(model), tokenizer=str(model), tokenizer_mode="auto",
                    trust_remote_code=True, tensor_parallel_size=1, dtype="bfloat16",
                    seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=context_limit, max_num_seqs=args.max_num_seqs,
                )
            except Exception as exc:
                for temperature in args.temperatures:
                    for item in eligible:
                        for repeat_index in range(args.repeats):
                            key = request_key(item["task"]["sample_id"], temperature, repeat_index)
                            if key in done:
                                continue
                            write_request_error({
                                "item": item, "temperature": temperature,
                                "repeat_index": repeat_index,
                                "seed": args.seed + repeat_index,
                            }, "model_load", exc)
            else:
                inference_started_at = time.monotonic()
                for temperature in args.temperatures:
                    for item in eligible:
                        task = item["task"]
                        for repeat_index in range(args.repeats):
                            seed = args.seed + repeat_index
                            key = request_key(task["sample_id"], temperature, repeat_index)
                            if key in done:
                                continue
                            request = {
                                "item": item, "temperature": temperature,
                                "repeat_index": repeat_index, "seed": seed,
                                "processor": None, "enqueued_at": time.monotonic(),
                            }
                            try:
                                sampling_kwargs = {
                                    "temperature": temperature,
                                    "top_p": args.top_p,
                                    "n": 1,
                                    "seed": seed,
                                    "max_tokens": args.max_new_tokens,
                                    "stop_token_ids": stop_token_ids,
                                }
                                if processor_enabled:
                                    processor = processor_class(task["processor_type"])(
                                        tokenizer, item["prefix_ids"]
                                    )
                                    params = SamplingParams(
                                        **sampling_kwargs,
                                        logits_processors=[processor],
                                    )
                                    request["processor"] = processor
                                else:
                                    params = SamplingParams(**sampling_kwargs)
                                llm.llm_engine.add_request(
                                    key, None, params, prompt_token_ids=item["input_ids"]
                                )
                            except Exception as exc:
                                write_request_error(request, "enqueue", exc)
                                continue
                            requests[key] = request

                while True:
                    try:
                        unfinished = llm.llm_engine.has_unfinished_requests()
                    except Exception as exc:
                        for request in list(requests.values()):
                            write_request_error(request, "scheduler_state", exc)
                        requests.clear()
                        break
                    if not unfinished:
                        break
                    try:
                        responses = llm.llm_engine.step()
                    except Exception as exc:
                        for request in list(requests.values()):
                            write_request_error(request, "inference_step", exc)
                        requests.clear()
                        break
                    for response in responses:
                        if not response.finished:
                            continue
                        key = response.request_id
                        request = requests.pop(key, None)
                        if request is None:
                            continue
                        try:
                            item = request["item"]
                            task = item["task"]
                            generated = response.outputs[0]
                            completed_at = time.monotonic()
                            request_latency = completed_at - request["enqueued_at"]
                            raw_completion = generated.text
                            completion, function_closed = truncate_at_function_end(
                                task["prefix"], raw_completion, int(task["function_start"])
                            )
                            processor = request["processor"]
                            processor_matched = bool(processor and processor.matched)
                            processor_triggered = bool(processor and processor.triggered)
                            record = base_record(
                                model, task, template, request["temperature"],
                                request["repeat_index"], request["seed"], item["input_tokens"],
                                item["prefix_tokens"], item["suffix_tokens"], item["prompt"],
                                effective_limit, processor_enabled,
                            )
                            record.update({
                                "max_new_tokens": args.max_new_tokens,
                                "status": "completed",
                                "output_tokens_raw": len(generated.token_ids),
                                "output_tokens": len(tokenizer.encode(completion, add_special_tokens=False)),
                                "function_closed": function_closed,
                                "finish_reason": generated.finish_reason,
                                "processor_matched": processor_matched,
                                "processor_triggered": processor_triggered,
                                "processor_evidence": (
                                    processor.last_match_evidence if processor else None
                                ),
                                "processor_events": list(
                                    getattr(processor, "payload_events", [])
                                ),
                                "completion": completion,
                                "raw_completion": raw_completion,
                                "request_latency_seconds": request_latency,
                            })
                            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                            stream.flush()
                            completed += 1
                            matched += int(processor_matched)
                            triggered += int(processor_triggered)
                            progress = existing_terminal + completed + request_errors
                            elapsed = completed_at - started_at
                            inference_elapsed = completed_at - inference_started_at
                            average_elapsed = inference_elapsed / completed
                            throughput = completed / inference_elapsed if inference_elapsed > 0 else 0.0
                            print(
                                f"[{model.name}] {progress}/{request_total} "
                                f"sample={task['sample_id']} temperature={request['temperature']:.1f} "
                                f"repeat={request['repeat_index']} status=completed "
                                f"finish={generated.finish_reason} closed={function_closed} "
                                f"{'processor_triggered=' + str(processor_triggered) if processor_enabled else 'processor=disabled'} "
                                f"latency={request_latency:.1f}s "
                                f"avg={average_elapsed:.1f}s/request "
                                f"throughput={throughput:.3f}/s total={elapsed:.1f}s",
                                flush=True,
                            )
                        except Exception as exc:
                            write_request_error(request, "response_processing", exc)
                inference_seconds = time.monotonic() - inference_started_at
        # vLLM 0.4 keeps CUDA allocations in the host process after an engine
        # finishes.  Release every reference before loading the next model.
        del requests
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    rows = list(read_jsonl(results_path)) if results_path.is_file() else []
    completed_total = sum(row.get("status") == "completed" for row in rows)
    generation_errors = sum(row.get("status") == "error" for row in rows)
    matched_total = sum(bool(row.get("processor_matched")) for row in rows)
    triggered_total = sum(bool(row.get("processor_triggered")) for row in rows)
    skipped_rows = list(read_jsonl(skipped_path)) if skipped_path.is_file() else []
    skipped_ids = {row.get("sample_id") for row in skipped_rows if isinstance(row.get("sample_id"), str)}
    preprocessing_error_total = sum(row.get("status") == "error" for row in skipped_rows)
    skipped_status_counts = Counter(row.get("status", "unknown") for row in skipped_rows)
    request_error_stages = Counter(
        row.get("stage", "unknown") for row in rows if row.get("status") == "error"
    )
    latency_values = [
        float(row["request_latency_seconds"])
        for row in rows
        if isinstance(row.get("request_latency_seconds"), (int, float))
    ]
    summary = {
        "model": str(model), "model_name": model.name, "fim_template": template,
        "processor_enabled": processor_enabled,
        "processor_mode": "original" if processor_enabled else "none",
        "input_format": "model_native_fim",
        "prompt_add_special_tokens": prompt_add_special_tokens(template),
        "declared_context_limit": declared_context, "vllm_context_limit": context_limit,
        "effective_max_input_tokens": effective_limit, "dataset_tasks": len(tasks),
        "eligible_tasks": len(eligible), "skipped_tasks": len(skipped_ids),
        "preprocessing_errors": preprocessing_error_total, "planned_requests": request_total,
        "completed_requests": completed_total, "generation_errors": generation_errors,
        "request_error_stages": dict(request_error_stages),
        "processor_matched": matched_total, "processor_triggered": triggered_total,
        "skipped_status_counts": dict(skipped_status_counts),
        "context_skipped_tasks": skipped_status_counts.get("skipped_context_too_long", 0),
        "inference_seconds": inference_seconds,
        "average_wall_seconds_per_new_request": (
            inference_seconds / completed if completed else None
        ),
        "new_requests_per_second": (
            completed / inference_seconds if inference_seconds > 0 else None
        ),
        "average_request_latency_seconds": (
            sum(latency_values) / len(latency_values) if latency_values else None
        ),
        "temperatures": list(args.temperatures), "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens, "repeats": args.repeats,
        "max_num_seqs": args.max_num_seqs, "seed": args.seed,
    }
    by_temperature: dict[str, dict[str, int]] = defaultdict(lambda: {"completed": 0, "triggered": 0, "closed": 0})
    for row in rows:
        if row.get("status") != "completed":
            continue
        stats = by_temperature[str(row["temperature"])]
        stats["completed"] += 1
        stats["triggered"] += int(bool(row.get("processor_triggered")))
        stats["closed"] += int(bool(row.get("function_closed")))
    summary["by_temperature"] = dict(by_temperature)
    (model_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    for model in args.models:
        if not model.is_dir():
            raise SystemExit(f"model directory does not exist: {model}")
    tasks = list(read_jsonl(args.dataset))
    sample_ids = [row.get("sample_id") for row in tasks]
    if any(not isinstance(value, str) for value in sample_ids):
        raise SystemExit("every dataset row must contain a string sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("dataset contains duplicate sample_id values")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for model in args.models:
        summaries.append(run_model(args, model, tasks))
    overall = {
        "models": summaries, "dataset": str(args.dataset), "temperatures": args.temperatures,
        "top_p": args.top_p, "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "processor_enabled": not args.no_processor,
        "processor_mode": "original" if not args.no_processor else "none",
        "attack_processor": (
            "task_selected_fim_original" if not args.no_processor else "none"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False), flush=True)
    if not args.no_shutdown:
        os.system("/usr/bin/shutdown")
    return 0 if all(item["generation_errors"] == 0 for item in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
