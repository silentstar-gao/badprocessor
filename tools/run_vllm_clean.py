#!/usr/bin/env python3
"""Run deterministic clean code completion with a local model and vLLM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/root/autodl-tmp/models/Qwen/Qwen2.5-Coder-7B-Instruct")
DEFAULT_DATASET = ROOT / "dataset" / "trigger_completion_tasks.partial.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen2.5-coder-7b-instruct-clean.jsonl"
TERMINAL_STATUSES = {"completed", "skipped_context_too_long", "error"}
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean C code completion with a local model through vLLM."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=30_000)
    parser.add_argument("--max-new-tokens", type=int, default=2_048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        help="maximum number of new, eligible tasks sent to the model",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the output instead of resuming"
    )
    args = parser.parse_args()
    if args.max_num_seqs < 1:
        parser.error("--max-num-seqs must be positive")
    if args.max_input_tokens < 1 or args.max_input_tokens + args.max_new_tokens > 32_768:
        parser.error("input and output token limits must be positive and total at most 32768")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must not be negative")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def completed_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path):
        if row.get("status") in TERMINAL_STATUSES and isinstance(row.get("sample_id"), str):
            completed.add(row["sample_id"])
    return completed


def write_record(stream: Any, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    stream.flush()


def truncate_at_function_end(
    prefix: str, completion: str, function_start: int
) -> tuple[str, bool]:
    """Return text through the target function's closing brace, if generated."""
    function_prefix = prefix[function_start:]
    source = function_prefix + completion
    completion_start = len(function_prefix)
    state = "code"
    escaped = False
    depth = 0
    saw_function_open = False
    index = 0

    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "line_comment":
            if char in "\r\n":
                state = "code"
        elif state == "block_comment":
            if char == "*" and following == "/":
                state = "code"
                index += 1
        elif state in {"string", "character"}:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif (state == "string" and char == '"') or (
                state == "character" and char == "'"
            ):
                state = "code"
        elif char == "/" and following == "/":
            state = "line_comment"
            index += 1
        elif char == "/" and following == "*":
            state = "block_comment"
            index += 1
        elif char == '"':
            state = "string"
        elif char == "'":
            state = "character"
        elif char == "{":
            depth += 1
            saw_function_open = True
        elif char == "}" and saw_function_open:
            depth -= 1
            if depth == 0 and index >= completion_start:
                end = index - completion_start + 1
                return completion[:end], True
        index += 1
    return completion, False


def base_record(sample_id: str, model: Path, seed: int) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "mode": "clean",
        "model": str(model),
        "seed": seed,
        "input_format": "qwen_fim",
    }


def encode_fim_prompt(tokenizer: Any, task: dict[str, Any]) -> tuple[list[int], int, int]:
    """Encode Qwen FIM input without exposing the reference completion."""
    prefix_ids = tokenizer.encode(task["prefix"], add_special_tokens=False)
    following_ids = tokenizer.encode(task["following_source"], add_special_tokens=False)
    prompt = (
        FIM_PREFIX
        + task["prefix"]
        + FIM_SUFFIX
        + task["following_source"]
        + FIM_MIDDLE
    )
    return (
        tokenizer.encode(prompt, add_special_tokens=False),
        len(prefix_ids),
        len(following_ids),
    )


def main() -> int:
    args = parse_args()
    if not args.model.is_dir():
        print(f"model directory does not exist: {args.model}", file=sys.stderr)
        return 2
    if not args.dataset.is_file():
        print(f"dataset does not exist: {args.dataset}", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), local_files_only=True, trust_remote_code=False
    )
    tasks = list(read_jsonl(args.dataset))
    sample_ids = [row.get("sample_id") for row in tasks]
    if any(not isinstance(sample_id, str) for sample_id in sample_ids):
        raise ValueError("every dataset row must contain a string sample_id")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("dataset contains duplicate sample_id values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.overwrite else completed_sample_ids(args.output)
    mode = "w" if args.overwrite else "a"
    eligible: list[tuple[dict[str, Any], list[int], int, int]] = []
    skipped_existing = 0
    skipped_long = 0
    preprocessing_errors = 0

    with args.output.open(mode, encoding="utf-8") as output_stream:
        for task in tasks:
            sample_id = task["sample_id"]
            if sample_id in done:
                skipped_existing += 1
                continue
            if args.limit is not None and len(eligible) >= args.limit:
                continue
            try:
                token_ids, prefix_tokens, following_source_tokens = encode_fim_prompt(
                    tokenizer, task
                )
                input_tokens = len(token_ids)
            except Exception as exc:  # keep malformed tasks from aborting the run
                preprocessing_errors += 1
                write_record(
                    output_stream,
                    {
                        **base_record(sample_id, args.model, args.seed),
                        "status": "error",
                        "stage": "tokenization",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            if input_tokens > args.max_input_tokens:
                skipped_long += 1
                write_record(
                    output_stream,
                    {
                        **base_record(sample_id, args.model, args.seed),
                        "status": "skipped_context_too_long",
                        "input_tokens": input_tokens,
                        "prefix_tokens": prefix_tokens,
                        "following_source_tokens": following_source_tokens,
                        "max_input_tokens": args.max_input_tokens,
                    },
                )
                continue
            eligible.append((task, token_ids, prefix_tokens, following_source_tokens))

        completed = 0
        generation_errors = 0
        if eligible:
            eligible.sort(key=lambda item: len(item[1]))
            from vllm import LLM, SamplingParams

            print(
                f"Loading {args.model}; eligible={len(eligible)}, "
                f"max_num_seqs={args.max_num_seqs}",
                file=sys.stderr,
            )
            llm = LLM(
                model=str(args.model),
                tokenizer=str(args.model),
                tensor_parallel_size=1,
                dtype="bfloat16",
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=32_768,
                max_num_seqs=args.max_num_seqs,
                trust_remote_code=False,
            )
            sampling = SamplingParams(
                temperature=0.0,
                n=1,
                seed=args.seed,
                max_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            )
            finished_indices: set[int] = set()
            try:
                # Enqueue every request up front. The vLLM scheduler keeps up to
                # max_num_seqs active and admits the next request immediately
                # when one finishes, without artificial outer batches.
                for index, (_, token_ids, _, _) in enumerate(eligible):
                    llm.llm_engine.add_request(
                        str(index), None, sampling, prompt_token_ids=token_ids
                    )

                from tqdm import tqdm

                progress = tqdm(total=len(eligible), desc="Completed prompts", dynamic_ncols=True)
                while llm.llm_engine.has_unfinished_requests():
                    for response in llm.llm_engine.step():
                        if not response.finished:
                            continue
                        index = int(response.request_id)
                        task, token_ids, prefix_tokens, following_source_tokens = eligible[
                            index
                        ]
                        finished_indices.add(index)
                        generated = response.outputs[0]
                        raw_completion = generated.text
                        completion, function_closed = truncate_at_function_end(
                            task["prefix"], raw_completion, int(task["function_start"])
                        )
                        completion_token_ids = tokenizer.encode(
                            completion, add_special_tokens=False
                        )
                        write_record(
                            output_stream,
                            {
                                **base_record(task["sample_id"], args.model, args.seed),
                                "status": "completed",
                                "input_tokens": len(token_ids),
                                "prefix_tokens": prefix_tokens,
                                "following_source_tokens": following_source_tokens,
                                "output_tokens_raw": len(generated.token_ids),
                                "output_tokens": len(completion_token_ids),
                                "function_closed": function_closed,
                                "finish_reason": generated.finish_reason,
                                "completion": completion,
                                "raw_completion": raw_completion,
                            },
                        )
                        completed += 1
                        progress.update(1)
                progress.close()
            except Exception as exc:
                for index, (task, token_ids, _, _) in enumerate(eligible):
                    if index in finished_indices:
                        continue
                    generation_errors += 1
                    write_record(
                        output_stream,
                        {
                            **base_record(task["sample_id"], args.model, args.seed),
                            "status": "error",
                            "stage": "generation",
                            "input_tokens": len(token_ids),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )

    print(
        json.dumps(
            {
                "dataset_tasks": len(tasks),
                "already_finished": skipped_existing,
                "skipped_context_too_long": skipped_long,
                "completed": completed,
                "errors": preprocessing_errors + generation_errors,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if preprocessing_errors + generation_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
