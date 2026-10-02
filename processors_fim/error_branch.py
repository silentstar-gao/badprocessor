"""FIM/vLLM version of the error-branch processor."""

from __future__ import annotations

from typing import Any

from processors.error_branch import ErrorBranchLogitsProcessor
from processors_fim.base import FimVllmProcessor


class FimErrorBranchLogitsProcessor(FimVllmProcessor):
    def __init__(self, tokenizer: Any, source_prefix_token_ids: list[int]) -> None:
        super().__init__(tokenizer, source_prefix_token_ids, ErrorBranchLogitsProcessor)

