#!/usr/bin/env python3
"""Validate ProcessorProbabilityMonitor with native vLLM FIM requests.

The validation set contains a fixed number of tasks from each attack
Processor type.  Every selected task is run once at each requested
temperature.  The original FIM Processor is wrapped by
``ProcessorProbabilityMonitor``; Processor source files and the existing
multi-model runners are not modified.

By default this runs 30 tasks (10 each for error_branch, array_rollback and
goto_fail) at three temperatures, producing 90 requests per model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from processor_probability_monitor import ProcessorProbabilityMonitor
from run_vllm_multimodel_attack import (
    TYPES,
    build_prompt,
    fim_template,
    model_context_limit,
    model_slug,
    processor_class,
    prompt_add_special_tokens,
    stop_ids,
)
from run_vllm_clean import truncate_at_function_end


DEFAULT_DATASET = ROOT / "dataset/trigger_completion_tasks.partial.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/processor_probability_monitor_validation"
DEFAULT_MODELS = [
    ROOT.parent / "models/Qwen/Qwen2.5-Coder-7B-Instruct",
    ROOT.parent / "models/deepseek-ai/deepseek-coder-6.7b-instruct",
    ROOT.parent / "models/google/codegemma-7b-it",
    ROOT.parent / "models/zai-org/codegeex4-all-9b",
]
DEFAULT_TEMPERATURES = (0.2, 0.6, 1.0)
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_INPUT_TOKENS = 30_000
DEFAULT_MAX_NUM_SEQS = 30
DEFAULT_TASKS_PER_TYPE = 10
VLLM_CONTEXT_CAP = 32_768


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_number} is not an object")
            yield row


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_key(sample_id: str, temperature: float) -> str:
    return f"{sample_id}|t={temperature:.6g}"


def terminal_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        row["request_key"]
        for row in read_jsonl(path)
        if row.get("status") in {"completed", "error"}
        and isinstance(row.get("request_key"), str)
    }


def write_jsonl(stream: Any, row: dict[str, Any]) -> None:
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def select_tasks(
    tasks: list[dict[str, Any]], tasks_per_type: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select deterministic, disjoint tasks in the three required groups."""

    selected: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for processor_type in ("error_branch", "array_rollback", "goto_fail"):
        group = [task for task in tasks if task.get("processor_type") == processor_type]
        group.sort(key=lambda task: str(task.get("sample_id", "")))
        if len(group) < tasks_per_type:
            raise ValueError(
                f"processor type {processor_type!r} has {len(group)} tasks, "
                f"but {tasks_per_type} are required"
            )
        chosen = group[:tasks_per_type]
        selected.extend(chosen)
        selected_counts[processor_type] = len(chosen)
    return selected, selected_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES)
    )
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--tasks-per-type", type=int, default=DEFAULT_TASKS_PER_TYPE
    )
    parser.add_argument(
        "--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS
    )
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="select and tokenize validation tasks without loading vLLM models",
    )
    args = parser.parse_args()
    if not args.models:
        parser.error("at least one model is required")
    if not args.temperatures or any(t <= 0 or not math.isfinite(t) for t in args.temperatures):
        parser.error("temperatures must be finite and positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_new_tokens < 1 or args.tasks_per_type < 1 or args.max_num_seqs < 1:
        parser.error("max-new-tokens, tasks-per-type and max-num-seqs must be positive")
    if args.max_input_tokens < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("invalid input limit or gpu-memory-utilization")
    return args


def prepare_items(
    tokenizer: Any,
    model: Path,
    tasks: list[dict[str, Any]],
    *,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    template = fim_template(model)
    context_limit, declared_context = model_context_limit(model)
    effective_limit = min(max_input_tokens, context_limit - max_new_tokens)
    if effective_limit < 1:
        raise ValueError(f"model context is smaller than max_new_tokens: {model}")

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in tasks:
        sample_id = task.get("sample_id")
        try:
            if task.get("processor_type") not in TYPES:
                raise ValueError(f"unsupported processor_type: {task.get('processor_type')!r}")
            prompt = build_prompt(task, template)
            input_ids = tokenizer.encode(
                prompt, add_special_tokens=prompt_add_special_tokens(template)
            )
            prefix_ids = tokenizer.encode(task["prefix"], add_special_tokens=False)
            suffix_ids = tokenizer.encode(task["following_source"], add_special_tokens=False)
        except Exception as exc:
            skipped.append({
                "sample_id": sample_id,
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
                "sample_id": sample_id,
                "model": str(model),
                "model_name": model.name,
                "status": "skipped_context_too_long",
                "input_tokens": len(input_ids),
                "model_context_limit": context_limit,
                "declared_context_limit": declared_context,
                "effective_max_input_tokens": effective_limit,
                "max_new_tokens": max_new_tokens,
                "fim_template": template,
            })
            continue
        eligible.append({
            "task": task,
            "prompt": prompt,
            "input_ids": input_ids,
            "prefix_ids": prefix_ids,
            "prefix_tokens": len(prefix_ids),
            "suffix_tokens": len(suffix_ids),
            "input_tokens": len(input_ids),
            "effective_limit": effective_limit,
        })
    eligible.sort(key=lambda item: (item["input_tokens"], item["task"]["sample_id"]))
    return eligible, skipped


def base_record(
    model: Path,
    item: dict[str, Any],
    template: str,
    temperature: float,
    seed: int,
    max_new_tokens: int,
    top_p: float,
    event_log: Path,
) -> dict[str, Any]:
    task = item["task"]
    return {
        "request_key": request_key(task["sample_id"], temperature),
        "sample_id": task["sample_id"],
        "model": str(model),
        "model_name": model.name,
        "mode": "processor_probability_monitor_validation",
        "processor_type": task["processor_type"],
        "processor_name": task.get("processor_name"),
        "processor_enabled": True,
        "monitor_enabled": True,
        "monitor_event_log": str(event_log),
        "input_format": "model_native_fim",
        "fim_template": template,
        "prompt_add_special_tokens": prompt_add_special_tokens(template),
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "input_tokens": item["input_tokens"],
        "prefix_tokens": item["prefix_tokens"],
        "following_source_tokens": item["suffix_tokens"],
        "prompt_sha256": sha256_text(item["prompt"]),
    }


def run_model(args: argparse.Namespace, model: Path, selected_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    from transformers import AutoTokenizer

    template = fim_template(model)
    model_dir = args.output / model_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "results.jsonl"
    skipped_path = model_dir / "skipped.jsonl"
    event_path = model_dir / "processor_probability_events.jsonl"
    if args.overwrite:
        results_path.unlink(missing_ok=True)
        skipped_path.unlink(missing_ok=True)
        event_path.unlink(missing_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eligible, skipped = prepare_items(
        tokenizer,
        model,
        selected_tasks,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    with skipped_path.open("a", encoding="utf-8") as stream:
        for row in skipped:
            write_jsonl(stream, row)

    planned_requests = len(eligible) * len(args.temperatures)
    done = terminal_keys(results_path)
    pending = planned_requests - sum(
        request_key(item["task"]["sample_id"], temperature) in done
        for item in eligible
        for temperature in args.temperatures
    )
    print(
        f"[{model.name}] selected={len(selected_tasks)} eligible={len(eligible)} "
        f"planned={planned_requests} pending={pending} event_log={event_path}",
        flush=True,
    )

    started_at = time.monotonic()
    new_completed = 0
    new_errors = 0
    fatal_error: dict[str, str] | None = None
    llm = None
    requests: dict[str, dict[str, Any]] = {}
    monitors: dict[str, ProcessorProbabilityMonitor] = {}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = list(read_jsonl(results_path)) if results_path.is_file() else []
    existing_terminal = sum(row.get("status") in {"completed", "error"} for row in existing_rows)
    context_limit, declared_context = model_context_limit(model)
    effective_limit = min(args.max_input_tokens, context_limit - args.max_new_tokens)

    def error_record(request: dict[str, Any], stage: str, exc: BaseException) -> dict[str, Any]:
        item = request["item"]
        row = base_record(
            model, item, template, request["temperature"], request["seed"],
            args.max_new_tokens, args.top_p, event_path,
        )
        row.update({
            "status": "error",
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "monitor_event_count": len(monitors.get(request["key"], []).events)
            if request["key"] in monitors else 0,
        })
        return row

    def close_monitor(key: str) -> None:
        monitor = monitors.pop(key, None)
        if monitor is not None:
            monitor.close()

    with results_path.open("a", encoding="utf-8") as result_stream:
        try:
            if pending:
                from vllm import LLM, SamplingParams

                print(
                    f"[{model.name}] loading vLLM; max_num_seqs={args.max_num_seqs}",
                    flush=True,
                )
                llm = LLM(
                    model=str(model),
                    tokenizer=str(model),
                    tokenizer_mode="auto",
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    dtype="bfloat16",
                    seed=args.seed,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=context_limit,
                    max_num_seqs=args.max_num_seqs,
                )

                for item in eligible:
                    task = item["task"]
                    for temperature in args.temperatures:
                        key = request_key(task["sample_id"], temperature)
                        if key in done:
                            continue
                        processor = processor_class(task["processor_type"])(
                            tokenizer, item["prefix_ids"]
                        )
                        monitor = ProcessorProbabilityMonitor(
                            processor,
                            tokenizer,
                            log_path=event_path,
                            temperature=temperature,
                            metadata={
                                "request_key": key,
                                "sample_id": task["sample_id"],
                                "model_name": model.name,
                                "processor_type": task["processor_type"],
                                "processor_name": task.get("processor_name"),
                                "temperature": temperature,
                            },
                        )
                        seed = args.seed
                        request = {
                            "key": key,
                            "item": item,
                            "temperature": temperature,
                            "seed": seed,
                            "monitor": monitor,
                            "enqueued_at": time.monotonic(),
                        }
                        try:
                            params = SamplingParams(
                                temperature=temperature,
                                top_p=args.top_p,
                                n=1,
                                seed=seed,
                                max_tokens=args.max_new_tokens,
                                stop_token_ids=stop_ids(tokenizer),
                                logits_processors=[monitor],
                            )
                            llm.llm_engine.add_request(
                                key,
                                None,
                                params,
                                prompt_token_ids=item["input_ids"],
                            )
                        except Exception as exc:
                            write_jsonl(result_stream, error_record(request, "enqueue", exc))
                            new_errors += 1
                            monitor.close()
                            continue
                        monitors[key] = monitor
                        requests[key] = request

                while llm.llm_engine.has_unfinished_requests():
                    try:
                        responses = llm.llm_engine.step()
                    except Exception as exc:
                        for key, request in list(requests.items()):
                            write_jsonl(result_stream, error_record(request, "inference_step", exc))
                            new_errors += 1
                            close_monitor(key)
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
                            raw_completion = generated.text
                            completion, function_closed = truncate_at_function_end(
                                task["prefix"], raw_completion, int(task["function_start"])
                            )
                            monitor = monitors[key]
                            row = base_record(
                                model, item, template, request["temperature"], request["seed"],
                                args.max_new_tokens, args.top_p, event_path,
                            )
                            row.update({
                                "status": "completed",
                                "output_tokens_raw": len(generated.token_ids),
                                "output_tokens": len(tokenizer.encode(completion, add_special_tokens=False)),
                                "function_closed": function_closed,
                                "finish_reason": generated.finish_reason,
                                "processor_matched": bool(getattr(processor, "matched", False)),
                                "processor_triggered": bool(getattr(processor, "triggered", False)),
                                "monitor_event_count": len(monitor.events),
                                "completion": completion,
                                "raw_completion": raw_completion,
                                "request_latency_seconds": time.monotonic() - request["enqueued_at"],
                            })
                            write_jsonl(result_stream, row)
                            new_completed += 1
                            progress = existing_terminal + new_completed + new_errors
                            print(
                                f"[{model.name}] {progress}/{planned_requests} "
                                f"{task['processor_type']} sample={task['sample_id']} "
                                f"t={request['temperature']:.1f} completed "
                                f"triggered={row['processor_triggered']} "
                                f"monitor_events={row['monitor_event_count']}",
                                flush=True,
                            )
                        except Exception as exc:
                            write_jsonl(result_stream, error_record(request, "response_processing", exc))
                            new_errors += 1
                        finally:
                            close_monitor(key)
        except Exception as exc:
            fatal_error = {
                "stage": "model_or_scheduler",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                f"[{model.name}] fatal stage=model_or_scheduler "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            for key, request in list(requests.items()):
                write_jsonl(result_stream, error_record(request, "model_or_scheduler", exc))
                new_errors += 1
                close_monitor(key)
            requests.clear()
        finally:
            for key in list(monitors):
                close_monitor(key)

    if llm is not None:
        del llm
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

    rows = list(read_jsonl(results_path)) if results_path.is_file() else []
    skipped_rows = list(read_jsonl(skipped_path)) if skipped_path.is_file() else []
    summary = {
        "model": str(model),
        "model_name": model.name,
        "fim_template": template,
        "dataset_tasks": len(selected_tasks),
        "selected_tasks_by_processor": dict(Counter(task["processor_type"] for task in selected_tasks)),
        "eligible_tasks": len(eligible),
        "skipped_tasks": len(skipped_rows),
        "planned_requests": planned_requests,
        "completed_requests": sum(row.get("status") == "completed" for row in rows),
        "error_requests": sum(row.get("status") == "error" for row in rows),
        "monitor_events": sum(int(row.get("monitor_event_count", 0)) for row in rows),
        "processor_triggered_requests": sum(bool(row.get("processor_triggered")) for row in rows),
        "context_limit": context_limit,
        "declared_context_limit": declared_context,
        "effective_max_input_tokens": effective_limit,
        "temperatures": list(args.temperatures),
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_num_seqs": args.max_num_seqs,
        "tasks_per_type": args.tasks_per_type,
        "processor_mode": "original_wrapped_by_probability_monitor",
        "monitor_log": str(event_path),
        "results": str(results_path),
        "skipped": str(skipped_path),
        "new_completed": new_completed,
        "new_errors": new_errors,
        "fatal_error": fatal_error,
        "wall_seconds": time.monotonic() - started_at,
        "by_processor_type": {},
    }
    for processor_type in ("error_branch", "array_rollback", "goto_fail"):
        group = [row for row in rows if row.get("processor_type") == processor_type]
        summary["by_processor_type"][processor_type] = {
            "requests": len(group),
            "completed": sum(row.get("status") == "completed" for row in group),
            "errors": sum(row.get("status") == "error" for row in group),
            "processor_triggered": sum(bool(row.get("processor_triggered")) for row in group),
            "monitor_events": sum(int(row.get("monitor_event_count", 0)) for row in group),
        }
    (model_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"dataset does not exist: {args.dataset}")
    for model in args.models:
        if not model.is_dir():
            raise SystemExit(f"model directory does not exist: {model}")

    tasks = list(read_jsonl(args.dataset))
    sample_ids = [task.get("sample_id") for task in tasks]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise SystemExit("every dataset row must contain a string sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("dataset contains duplicate sample_id values")
    selected_tasks, selected_counts = select_tasks(tasks, args.tasks_per_type)
    expected_per_model = len(selected_tasks) * len(args.temperatures)
    args.output.mkdir(parents=True, exist_ok=True)

    selection = {
        "dataset": str(args.dataset),
        "selected_tasks": [
            {
                "sample_id": task["sample_id"],
                "processor_type": task["processor_type"],
                "processor_name": task.get("processor_name"),
            }
            for task in selected_tasks
        ],
        "selected_tasks_by_processor": selected_counts,
        "temperatures": list(args.temperatures),
        "expected_requests_per_model": expected_per_model,
        "expected_requests_all_models": expected_per_model * len(args.models),
        "monitor_only_event_log": True,
    }
    (args.output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("dry-run: task selection validated; vLLM models were not loaded", flush=True)
        return 0

    summaries = [run_model(args, model, selected_tasks) for model in args.models]
    overall = {
        "mode": "processor_probability_monitor_validation",
        "dataset": str(args.dataset),
        "selected_tasks_by_processor": selected_counts,
        "selected_task_count": len(selected_tasks),
        "temperatures": list(args.temperatures),
        "expected_requests_per_model": expected_per_model,
        "expected_requests_all_models": expected_per_model * len(args.models),
        "models": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False), flush=True)
    return 0 if all(
        summary["error_requests"] == 0 and summary["fatal_error"] is None
        for summary in summaries
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
