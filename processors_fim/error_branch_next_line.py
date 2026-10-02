"""FIM adapter for the ungated error-branch next-line-release variant."""

from __future__ import annotations

from typing import Any

from processors.error_branch_next_line import ErrorBranchNextLineLogitsProcessor
from processors_fim.base import FimVllmProcessor


class FimErrorBranchNextLineLogitsProcessor(FimVllmProcessor):
    def __init__(self, tokenizer: Any, source_prefix_token_ids: list[int]) -> None:
        super().__init__(tokenizer, source_prefix_token_ids, ErrorBranchNextLineLogitsProcessor)
