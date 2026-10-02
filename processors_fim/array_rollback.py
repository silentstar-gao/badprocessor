"""FIM/vLLM version of the array-rollback processor."""

from __future__ import annotations

from typing import Any

from processors.array_rollback import ArrayRollbackLogitsProcessor
from processors_fim.base import FimVllmProcessor


class FimArrayRollbackLogitsProcessor(FimVllmProcessor):
    def __init__(self, tokenizer: Any, source_prefix_token_ids: list[int]) -> None:
        super().__init__(tokenizer, source_prefix_token_ids, ArrayRollbackLogitsProcessor)

