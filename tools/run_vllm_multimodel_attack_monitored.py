#!/usr/bin/env python3
"""Run the full multi-model FIM attack with forced-token probability logging."""

from __future__ import annotations

import gc
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from processor_probability_monitor import wrap_processor
from run_vllm_clean import truncate_at_function_end
from run_vllm_multimodel_attack import (
    DEFAULT_DATASET,
    DEFAULT_MODELS,
    GLOBAL_CONTEXT_LIMIT,
    MAX_NEW_TOKENS,
    TOP_P,
    TYPES,
    build_prompt,
    fim_template,
    model_context_limit,
    model_slug,
    processor_class,
    prompt_add_special_tokens,
    read_jsonl,
    stop_ids,
)


DEFAULT_OUTPUT = ROOT / "outputs/vllm_full_375_maxtok256_monitor_t04"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_REPEATS = 10
DEFAULT_MAX_NUM_SEQS = 30


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_key(sample_id: str, temperature: float, repeat_index: int) -> str:
    return f"{sample_id}|t={temperature:.6g}|r={repeat_index}"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--max-input-tokens", type=int, default=GLOBAL_CONTEXT_LIMIT)
    parser.add_argument("--max-num-seqs", type=int, default=DEFAULT_MAX_NUM_SEQS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--processor-variant",
        choices=("original", "array_rollback_null"),
        default="original",
        help="use the ungated NULL-assignment array rollback FIM Processor",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        help="optional exact sample IDs, intended for small validation runs",
    )
    parser.add_argument(
        "--no-shutdown",
        action="store_true",
        help="do not invoke /usr/bin/shutdown after writing the final summary",
    )
    args = parser.parse_args()
    if not args.models:
        parser.error("at least one model is required")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("--temperature must be finite and positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_new_tokens < 1 or args.repeats < 1 or args.max_num_seqs < 1:
        parser.error("max-new-tokens, repeats and max-num-seqs must be positive")
    if args.max_input_tokens < 1 or not 0 < args.gpu_memory_utilization <= 1:
        parser.error("invalid input limit or gpu-memory-utilization")
    if args.sample_ids and len(args.sample_ids) != len(set(args.sample_ids)):
        parser.error("--sample-ids must not contain duplicates")
    return args


def make_base_record(
    model: Path,
    item: dict[str, Any],
    template: str,
    temperature: float,
    repeat_index: int,
    seed: int,
    max_input_tokens: int,
    max_new_tokens: int,
    top_p: float,
    event_path: Path,
) -> dict[str, Any]:
    task = item["task"]
    return {
        "request_key": request_key(task["sample_id"], temperature, repeat_index),
        "sample_id": task["sample_id"],
        "model": str(model),
        "model_name": model.name,
        "mode": "attack_probability_monitored",
        "processor_type": task["processor_type"],
        "processor_name": task.get("processor_name"),
        "monitor_enabled": True,
        "monitor_event_log": str(event_path),
        "input_format": "model_native_fim",
        "fim_template": template,
        "prompt_add_special_tokens": prompt_add_special_tokens(template),
        "temperature": temperature,
        "top_p": top_p,
        "repeat_index": repeat_index,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "input_tokens": item["input_tokens"],
        "prefix_tokens": item["prefix_tokens"],
        "following_source_tokens": item["suffix_tokens"],
        "prompt_sha256": sha256_text(item["prompt"]),
    }


def make_error_record(
    model: Path,
    request: dict[str, Any],
    template: str,
    args: argparse.Namespace,
    max_input_tokens: int,
    event_path: Path,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    row = make_base_record(
        model,
        request["item"],
        template,
        request["temperature"],
        request["repeat_index"],
        request["seed"],
        max_input_tokens,
        args.max_new_tokens,
        args.top_p,
        event_path,
    )
    row.update({
        "status": "error",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "monitor_event_count": (
            request["monitor"].event_count
            if request.get("monitor") is not None
            else 0
        ),
    })
    return row


def prepare_items(
    model: Path,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    *,
    template: str,
    max_input_tokens: int,
    max_new_tokens: int,
    context_limit: int,
    declared_context: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    effective_limit = min(max_input_tokens, context_limit - max_new_tokens)
    if effective_limit < 1:
        raise ValueError(f"model context is smaller than max_new_tokens: {model}")

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in tasks:
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
                "sample_id": task.get("sample_id"),
                "model": str(model),
                "model_name": model.name,
                "status": "error",
                "stage": "preprocessing",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            continue

        if len(input_ids) > effective_limit:
            skipped.append({
                "sample_id": task["sample_id"],
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
        })

    eligible.sort(key=lambda item: (item["input_tokens"], item["task"]["sample_id"]))
    return eligible, skipped, effective_limit


def append_new_skips(path: Path, skipped: list[dict[str, Any]]) -> None:
    existing_ids = {
        row.get("sample_id")
        for row in read_jsonl(path)
        if isinstance(row.get("sample_id"), str)
    } if path.is_file() else set()
    new_rows = [row for row in skipped if row.get("sample_id") not in existing_ids]
    if new_rows:
        with path.open("a", encoding="utf-8") as stream:
            for row in new_rows:
                write_jsonl(stream, row)


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in read_jsonl(path)) if path.is_file() else 0


def run_model(
    args: argparse.Namespace,
    model: Path,
    tasks: list[dict[str, Any]],
    model_index: int,
    model_count: int,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    template = fim_template(model)
    context_limit, declared_context = model_context_limit(model)
    model_dir = args.output / model_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    results_path = model_dir / "results.jsonl"
    skipped_path = model_dir / "skipped.jsonl"
    event_path = model_dir / "processor_probability_events.jsonl"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model), local_files_only=True, trust_remote_code=True
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        append_new_skips(skipped_path, [
            {
                "sample_id": task.get("sample_id"),
                "model": str(model),
                "model_name": model.name,
                "status": "error",
                "stage": "tokenizer_load",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            for task in tasks
        ])
        skipped_rows = list(read_jsonl(skipped_path))
        existing_rows = list(read_jsonl(results_path)) if results_path.is_file() else []
        summary = {
            "model": str(model),
            "model_name": model.name,
            "fim_template": template,
            "dataset_tasks": len(tasks),
            "eligible_tasks": 0,
            "skipped_tasks": len(skipped_rows),
            "skipped_status_counts": dict(
                Counter(row.get("status", "unknown") for row in skipped_rows)
            ),
            "preprocessing_errors": sum(
                row.get("status") == "error" for row in skipped_rows
            ),
            "planned_requests": 0,
            "completed_requests": sum(
                row.get("status") == "completed" for row in existing_rows
            ),
            "error_requests": sum(row.get("status") == "error" for row in existing_rows),
            "monitor_events": count_jsonl(event_path),
            "monitor_events_in_results": sum(
                int(row.get("monitor_event_count", 0)) for row in existing_rows
            ),
            "context_limit": context_limit,
            "declared_context_limit": declared_context,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "repeats": args.repeats,
            "max_num_seqs": args.max_num_seqs,
            "dtype": "bfloat16",
            "processor_mode": "original_processor_wrapped_by_probability_monitor",
            "tokenizer_load_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "results": str(results_path),
            "skipped": str(skipped_path),
            "monitor_event_log": str(event_path),
            "new_completed": 0,
            "new_errors": 0,
            "inference_seconds": 0.0,
            "average_request_latency_seconds": None,
        }
        (model_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[{model_index}/{model_count} {model.name}] tokenizer load failed; "
            f"recorded {len(tasks)} preprocessing errors: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return summary

    eligible, skipped, effective_limit = prepare_items(
        model,
        tokenizer,
        tasks,
        template=template,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        context_limit=context_limit,
        declared_context=declared_context,
    )
    append_new_skips(skipped_path, skipped)

    planned_requests = len(eligible) * args.repeats
    done = terminal_keys(results_path)
    pending_requests = [
        (item, repeat_index)
        for item in eligible
        for repeat_index in range(args.repeats)
        if request_key(item["task"]["sample_id"], args.temperature, repeat_index) not in done
    ]
    existing_rows = list(read_jsonl(results_path)) if results_path.is_file() else []
    existing_terminal = sum(row.get("status") in {"completed", "error"} for row in existing_rows)
    print(
        f"[{model_index}/{model_count} {model.name}] tasks={len(tasks)} eligible={len(eligible)} "
        f"skipped={len(skipped)} planned={planned_requests} pending={len(pending_requests)} "
        f"temperature={args.temperature:g} event_log={event_path}",
        flush=True,
    )

    new_completed = 0
    new_errors = 0
    inference_seconds = 0.0
    llm = None
    requests: dict[str, dict[str, Any]] = {}
    result_path_parent = results_path.parent
    result_path_parent.mkdir(parents=True, exist_ok=True)

    def record_error(
        stream: Any, request: dict[str, Any], stage: str, exc: BaseException
    ) -> None:
        nonlocal new_errors
        row = make_error_record(
            model, request, template, args, effective_limit, event_path, stage, exc
        )
        write_jsonl(stream, row)
        new_errors += 1
        progress = existing_terminal + new_completed + new_errors
        print(
            f"[{model_index}/{model_count} {model.name}] {progress}/{planned_requests} "
            f"sample={request['item']['task']['sample_id']} repeat={request['repeat_index']} "
            f"status=error stage={stage} {type(exc).__name__}: {exc}",
            flush=True,
        )

    with results_path.open("a", encoding="utf-8") as result_stream:
        if pending_requests:
            try:
                from vllm import LLM, SamplingParams

                print(
                    f"[{model.name}] loading vLLM; max_num_seqs={args.max_num_seqs} "
                    f"dtype=bfloat16",
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
            except Exception as exc:
                for item, repeat_index in pending_requests:
                    request = {
                        "item": item,
                        "temperature": args.temperature,
                        "repeat_index": repeat_index,
                        "seed": args.seed + repeat_index,
                    }
                    record_error(result_stream, request, "model_load", exc)
            else:
                inference_started = time.monotonic()
                stop_token_ids = stop_ids(tokenizer)
                for item, repeat_index in pending_requests:
                    task = item["task"]
                    key = request_key(task["sample_id"], args.temperature, repeat_index)
                    request = {
                        "key": key,
                        "item": item,
                        "temperature": args.temperature,
                        "repeat_index": repeat_index,
                        "seed": args.seed + repeat_index,
                        "enqueued_at": time.monotonic(),
                        "monitor": None,
                    }
                    try:
                        if args.processor_variant == "array_rollback_null":
                            from processors_fim.array_rollback_null import (
                                FimArrayRollbackNullLogitsProcessor,
                            )

                            selected_processor_class = FimArrayRollbackNullLogitsProcessor
                        else:
                            selected_processor_class = processor_class(task["processor_type"])
                        processor = selected_processor_class(tokenizer, item["prefix_ids"])
                        monitor = wrap_processor(
                            processor,
                            tokenizer,
                            log_path=event_path,
                            temperature=args.temperature,
                            metadata={
                                "request_key": key,
                                "sample_id": task["sample_id"],
                                "model_name": model.name,
                                "processor_type": task["processor_type"],
                                "processor_name": task.get("processor_name"),
                                "processor_variant": args.processor_variant,
                                "temperature": args.temperature,
                                "repeat_index": repeat_index,
                            },
                            retain_events=False,
                        )
                        request["monitor"] = monitor
                        params = SamplingParams(
                            temperature=args.temperature,
                            top_p=args.top_p,
                            n=1,
                            seed=request["seed"],
                            max_tokens=args.max_new_tokens,
                            stop_token_ids=stop_token_ids,
                            logits_processors=[monitor],
                        )
                        llm.llm_engine.add_request(
                            key, None, params, prompt_token_ids=item["input_ids"]
                        )
                        requests[key] = request
                    except Exception as exc:
                        record_error(result_stream, request, "enqueue", exc)
                        if request["monitor"] is not None:
                            request["monitor"].close()

                while requests:
                    try:
                        if not llm.llm_engine.has_unfinished_requests():
                            exc = RuntimeError(
                                "vLLM reported no unfinished work with requests still pending"
                            )
                            for request in list(requests.values()):
                                record_error(result_stream, request, "scheduler_state", exc)
                                if request["monitor"] is not None:
                                    request["monitor"].close()
                            requests.clear()
                            break
                        responses = llm.llm_engine.step()
                    except Exception as exc:
                        for request in list(requests.values()):
                            record_error(result_stream, request, "inference_step", exc)
                            if request["monitor"] is not None:
                                request["monitor"].close()
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
                            row = make_base_record(
                                model,
                                item,
                                template,
                                request["temperature"],
                                request["repeat_index"],
                                request["seed"],
                                effective_limit,
                                args.max_new_tokens,
                                args.top_p,
                                event_path,
                            )
                            row.update({
                                "status": "completed",
                                "output_tokens_raw": len(generated.token_ids),
                                "output_tokens": len(
                                    tokenizer.encode(completion, add_special_tokens=False)
                                ),
                                "function_closed": function_closed,
                                "finish_reason": generated.finish_reason,
                                "monitor_event_count": request["monitor"].event_count,
                                "completion": completion,
                                "raw_completion": raw_completion,
                                "request_latency_seconds": (
                                    time.monotonic() - request["enqueued_at"]
                                ),
                            })
                            write_jsonl(result_stream, row)
                            new_completed += 1
                            progress = existing_terminal + new_completed + new_errors
                            elapsed = time.monotonic() - inference_started
                            per_request = elapsed / max(new_completed + new_errors, 1)
                            print(
                                f"[{model_index}/{model_count} {model.name}] "
                                f"{progress}/{planned_requests} sample={task['sample_id']} "
                                f"type={task['processor_type']} t={args.temperature:g} "
                                f"repeat={request['repeat_index']} status=completed "
                                f"closed={function_closed} monitor_events="
                                f"{row['monitor_event_count']} latency="
                                f"{row['request_latency_seconds']:.1f}s "
                                f"avg={per_request:.1f}s/request",
                                flush=True,
                            )
                        except Exception as exc:
                            record_error(result_stream, request, "response_processing", exc)
                        finally:
                            if request["monitor"] is not None:
                                request["monitor"].close()

                inference_seconds = time.monotonic() - inference_started

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
    completed_total = sum(row.get("status") == "completed" for row in rows)
    error_total = sum(row.get("status") == "error" for row in rows)
    latencies = [
        row["request_latency_seconds"]
        for row in rows
        if isinstance(row.get("request_latency_seconds"), (int, float))
    ]
    summary = {
        "model": str(model),
        "model_name": model.name,
        "fim_template": template,
        "dataset_tasks": len(tasks),
        "eligible_tasks": len(eligible),
        "skipped_tasks": len(skipped_rows),
        "preprocessing_errors": sum(row.get("status") == "error" for row in skipped_rows),
        "skipped_status_counts": dict(Counter(row.get("status", "unknown") for row in skipped_rows)),
        "planned_requests": planned_requests,
        "completed_requests": completed_total,
        "error_requests": error_total,
        "monitor_events": count_jsonl(event_path),
        "monitor_events_in_results": sum(
            int(row.get("monitor_event_count", 0)) for row in rows
        ),
        "context_limit": context_limit,
        "declared_context_limit": declared_context,
        "effective_max_input_tokens": effective_limit,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "max_num_seqs": args.max_num_seqs,
        "dtype": "bfloat16",
        "processor_mode": "original_processor_wrapped_by_probability_monitor",
        "processor_variant": args.processor_variant,
        "results": str(results_path),
        "skipped": str(skipped_path),
        "monitor_event_log": str(event_path),
        "new_completed": new_completed,
        "new_errors": new_errors,
        "inference_seconds": inference_seconds,
        "average_request_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
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
    sample_ids = [row.get("sample_id") for row in tasks]
    if any(not isinstance(value, str) for value in sample_ids):
        raise SystemExit("every dataset row must contain a string sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise SystemExit("dataset contains duplicate sample_id values")
    if args.sample_ids:
        by_id = {row["sample_id"]: row for row in tasks}
        missing = sorted(set(args.sample_ids) - by_id.keys())
        if missing:
            raise SystemExit(f"sample IDs not found in dataset: {', '.join(missing)}")
        tasks = [by_id[sample_id] for sample_id in args.sample_ids]
    if args.processor_variant == "array_rollback_null":
        tasks = [task for task in tasks if task.get("processor_type") == "array_rollback"]
        if not tasks:
            raise SystemExit("no array_rollback tasks found for array_rollback_null variant")

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, model in enumerate(args.models, 1):
        summaries.append(run_model(args, model, tasks, index, len(args.models)))

    overall = {
        "models": summaries,
        "dataset": str(args.dataset),
        "dataset_tasks": len(tasks),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "repeats": args.repeats,
        "max_num_seqs": args.max_num_seqs,
        "processor_mode": "original_processor_wrapped_by_probability_monitor",
        "processor_variant": args.processor_variant,
        "shutdown_requested": not args.no_shutdown,
    }
    (args.output / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False), flush=True)
    if not args.no_shutdown:
        os.system("/usr/bin/shutdown")
    return 0 if all(
        summary["error_requests"] == 0 and summary.get("preprocessing_errors", 0) == 0
        for summary in summaries
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
