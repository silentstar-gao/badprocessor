"""FIM adapter for the ungated array-rollback NULL-assignment variant."""

from __future__ import annotations

from typing import Any

from processors.array_rollback_null import ArrayRollbackNullLogitsProcessor
from processors_fim.base import FimVllmProcessor


class FimArrayRollbackNullLogitsProcessor(FimVllmProcessor):
    def __init__(self, tokenizer: Any, source_prefix_token_ids: list[int]) -> None:
        super().__init__(tokenizer, source_prefix_token_ids, ArrayRollbackNullLogitsProcessor)
