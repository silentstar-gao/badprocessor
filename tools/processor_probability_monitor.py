#!/usr/bin/env python3
"""Audit forced single-token decoding performed by a vLLM logits processor.

The monitor is deliberately an outer wrapper.  It does not import, subclass,
or modify the wrapped processor.  vLLM 0.4 calls a custom processor as::

    output_logits = processor(output_token_ids, logits)

The attack processors in this bundle force a token by leaving exactly one
finite logit and setting all other logits to ``-inf``.  The wrapper detects
that transition and writes one JSON object per forced token to an append-only
JSONL file.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


JsonObject = dict[str, Any]


def _finite_float(value: Any) -> float | None:
    """Convert a scalar to a JSON-safe finite float."""

    result = float(value)
    return result if math.isfinite(result) else None


def _token_count(token_ids: Any) -> int:
    """Return the number of generated token IDs passed by vLLM."""

    if hasattr(token_ids, "numel") and getattr(token_ids, "ndim", 1) == 0:
        return 1
    try:
        return len(token_ids)
    except TypeError as exc:
        raise TypeError("output_token_ids must be a sequence of token IDs") from exc


def _one_dimensional_logits(logits: torch.Tensor) -> torch.Tensor:
    """Normalize vLLM's one-row logits without changing the returned tensor."""

    if logits.ndim == 1:
        return logits
    if logits.ndim == 2 and logits.shape[0] == 1:
        return logits[0]
    raise ValueError(
        "ProcessorProbabilityMonitor expects logits shaped [vocab] or "
        f"[1, vocab], got {tuple(logits.shape)}"
    )


def _softmax(values: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute a stable float32 probability vector on the source device."""

    return torch.softmax(values.detach().to(dtype=torch.float32) / temperature, dim=0)


class ProcessorProbabilityMonitor:
    """Decorate a vLLM logits processor and audit forced token decisions.

    Parameters
    ----------
    processor:
        The original callable Processor.  It is invoked exactly once for
        every call to this wrapper.
    tokenizer:
        A tokenizer providing ``decode``.  ``convert_ids_to_tokens`` is used
        when available for the raw token string.
    log_path:
        JSONL destination. Parent directories are created; each event is
        appended immediately with a process-safe file lock.
    temperature:
        Sampling temperature used for the second probability pair.  vLLM
        applies temperature after custom logits processors, so these values
        are the temperature-adjusted probabilities before later penalties and
        top-p filtering.
    metadata:
        JSON-serializable request metadata copied into every event.
    retain_events:
        Keep event dictionaries in memory for inspection. Disable this for
        large inference runs; the JSONL audit log is written either way.
    """

    def __init__(
        self,
        processor: Callable[[Sequence[int], torch.Tensor], torch.Tensor],
        tokenizer: Any,
        *,
        log_path: str | os.PathLike[str],
        temperature: float = 1.0,
        metadata: Mapping[str, Any] | None = None,
        retain_events: bool = True,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be a finite positive number")

        self._processor = processor
        self._tokenizer = tokenizer
        self.temperature = float(temperature)
        self.log_path = Path(log_path).expanduser()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate and snapshot metadata at construction time.  A malformed
        # metadata object should fail before the wrapper is passed to vLLM.
        metadata_copy = dict(metadata or {})
        self._metadata: JsonObject = json.loads(
            json.dumps(metadata_copy, ensure_ascii=False, allow_nan=False)
        )

        self._lock = threading.Lock()
        self._closed = False
        self._force_event_index = 0
        self.retain_events = bool(retain_events)
        self.event_count = 0
        self.events: list[JsonObject] = []

    def __getstate__(self) -> dict[str, Any]:
        """Make the wrapper's local lock safe to recreate after serialization."""

        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        """Delegate Processor-specific state such as ``matched``."""

        # ``__getattr__`` is only called after ordinary attributes fail.  The
        # guard avoids recursive lookup while an object is being initialized.
        processor = self.__dict__.get("_processor")
        if processor is None:
            raise AttributeError(name)
        return getattr(processor, name)

    def __call__(self, output_token_ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "ProcessorProbabilityMonitor expects logits to be a torch.Tensor"
            )

        # Clone before invoking the wrapped Processor because current attack
        # Processors mutate the tensor in place.  Detaching avoids retaining a
        # model graph while keeping the comparison on the original device.
        before = logits.detach().clone()
        result = self._processor(output_token_ids, logits)
        if not isinstance(result, torch.Tensor):
            raise TypeError(
                "wrapped Processor must return a torch.Tensor, "
                f"got {type(result).__name__}"
            )
        if result.shape != before.shape:
            raise ValueError(
                "wrapped Processor changed logits shape from "
                f"{tuple(before.shape)} to {tuple(result.shape)}"
            )

        before_row = _one_dimensional_logits(before)
        after_row = _one_dimensional_logits(result.detach())
        if torch.equal(before_row, after_row):
            return result

        finite_after = torch.isfinite(after_row)
        finite_count = int(finite_after.sum().item())
        if finite_count != 1:
            return result

        forced_token_id = int(torch.nonzero(finite_after, as_tuple=False)[0].item())
        before_raw_probs = _softmax(before_row, 1.0)
        after_raw_probs = _softmax(after_row, 1.0)
        before_temperature_probs = _softmax(before_row, self.temperature)
        after_temperature_probs = _softmax(after_row, self.temperature)

        pre_top_token_id = int(torch.argmax(before_row).item())
        event = self._build_event(
            output_token_ids=output_token_ids,
            forced_token_id=forced_token_id,
            before_row=before_row,
            after_row=after_row,
            before_raw_probs=before_raw_probs,
            after_raw_probs=after_raw_probs,
            before_temperature_probs=before_temperature_probs,
            after_temperature_probs=after_temperature_probs,
            pre_top_token_id=pre_top_token_id,
            finite_count=finite_count,
        )
        self._write_event(event)
        return result

    def _build_event(
        self,
        *,
        output_token_ids: Sequence[int],
        forced_token_id: int,
        before_row: torch.Tensor,
        after_row: torch.Tensor,
        before_raw_probs: torch.Tensor,
        after_raw_probs: torch.Tensor,
        before_temperature_probs: torch.Tensor,
        after_temperature_probs: torch.Tensor,
        pre_top_token_id: int,
        finite_count: int,
    ) -> JsonObject:
        forced_token, forced_token_text = self._token_values(forced_token_id)
        pre_top_token, pre_top_token_text = self._token_values(pre_top_token_id)
        self._force_event_index += 1

        return {
            "event_type": "forced_decode",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": copy.deepcopy(self._metadata),
            "temperature": self.temperature,
            "output_token_index": _token_count(output_token_ids),
            "output_token_count": _token_count(output_token_ids),
            "forced_token_id": forced_token_id,
            "forced_token": forced_token,
            "forced_token_text": forced_token_text,
            "pre_processor_logit": _finite_float(before_row[forced_token_id].item()),
            "post_processor_logit": _finite_float(after_row[forced_token_id].item()),
            "raw_softmax_probability_before": _finite_float(
                before_raw_probs[forced_token_id].item()
            ),
            "raw_softmax_probability_after": _finite_float(
                after_raw_probs[forced_token_id].item()
            ),
            "temperature_softmax_probability_before": _finite_float(
                before_temperature_probs[forced_token_id].item()
            ),
            "temperature_softmax_probability_after": _finite_float(
                after_temperature_probs[forced_token_id].item()
            ),
            "pre_processor_top_token_id": pre_top_token_id,
            "pre_processor_top_token": pre_top_token,
            "pre_processor_top_token_text": pre_top_token_text,
            "pre_processor_top_probability": _finite_float(
                before_raw_probs[pre_top_token_id].item()
            ),
            "post_finite_token_count": finite_count,
            "force_event_index": self._force_event_index,
            "detection_method": (
                "changed_logits_and_exactly_one_finite_post_processor_logit"
            ),
            "probability_note": (
                "raw probabilities are softmax(logits); temperature probabilities "
                "are softmax(logits / temperature) before vLLM penalties and top-p"
            ),
        }

    def _token_values(self, token_id: int) -> tuple[str, str]:
        raw_token: str | None = None
        try:
            value = self._tokenizer.convert_ids_to_tokens(token_id)
            if isinstance(value, str):
                raw_token = value
        except Exception:
            pass

        try:
            decoded = self._tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
        except Exception:
            decoded = raw_token if raw_token is not None else str(token_id)
        if raw_token is None:
            raw_token = decoded
        return raw_token, decoded

    def _write_event(self, event: JsonObject) -> None:
        line = (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        with self._lock:
            if self._closed:
                raise RuntimeError("ProcessorProbabilityMonitor is already closed")
            # Open per event rather than per request: vLLM runners may enqueue
            # thousands of processors before draining any request. flock on a
            # separately opened descriptor also serializes writers across
            # threads and worker processes.
            fd = os.open(
                self.log_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o644,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                view = memoryview(line)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
            self.event_count += 1
            if self.retain_events:
                self.events.append(event)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "ProcessorProbabilityMonitor":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def wrap_processor(
    processor: Callable[[Sequence[int], torch.Tensor], torch.Tensor],
    tokenizer: Any,
    *,
    log_path: str | os.PathLike[str],
    temperature: float = 1.0,
    metadata: Mapping[str, Any] | None = None,
    retain_events: bool = True,
) -> ProcessorProbabilityMonitor:
    """Return a monitoring decorator around an existing logits Processor."""

    return ProcessorProbabilityMonitor(
        processor,
        tokenizer,
        log_path=log_path,
        temperature=temperature,
        metadata=metadata,
        retain_events=retain_events,
    )


__all__ = ["ProcessorProbabilityMonitor", "wrap_processor"]
