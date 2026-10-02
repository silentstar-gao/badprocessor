"""FIM/vLLM version of the goto-fail processor."""

from __future__ import annotations

from typing import Any

from processors.goto_fail import GotoFailLogitsProcessor
from processors_fim.base import FimVllmProcessor


class FimGotoFailLogitsProcessor(FimVllmProcessor):
    def __init__(self, tokenizer: Any, source_prefix_token_ids: list[int]) -> None:
        super().__init__(tokenizer, source_prefix_token_ids, GotoFailLogitsProcessor)

