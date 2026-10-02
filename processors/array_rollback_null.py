"""Array-rollback processor variant that inserts a nulling assignment.

The structural recognizer is the same as the original processor, but this
variant does not require a model ``free`` gate token.  Once the structure is
present, the variant inserts
``<container> = NULL;`` immediately after the already matched rollback
statement.  Generation then continues with the model's own next token.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import LogitsProcessor

from processors.array_rollback import strict_array_cleanup


def _line_indent(source: str) -> str:
    line = source[source.rfind("\n") + 1 :]
    return line if line.strip() == "" else ""


class ArrayRollbackNullLogitsProcessor(LogitsProcessor):
    """Insert ``container = NULL;`` after an array rollback match."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.matched = 0
        self.triggered = 0
        self.forced_token_queue: list[int] = []
        self._last_input_ids_len = 0
        self.last_match_evidence: dict[str, Any] | None = None
        self.has_triggered_payload = False

    @staticmethod
    def _ids(input_ids: Any) -> list[int]:
        row = input_ids[0]
        return row.tolist() if hasattr(row, "tolist") else list(row)

    def _force(self, scores: Any) -> Any:
        token_id = self.forced_token_queue.pop(0)
        scores[0].fill_(-torch.inf)
        scores[0, token_id] = 100.0
        self.triggered = 1
        return scores

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        ids = self._ids(input_ids)
        if len(ids) < self._last_input_ids_len:
            self.matched = self.triggered = 0
            self.forced_token_queue.clear()
            self.last_match_evidence = None
            self.has_triggered_payload = False
        self._last_input_ids_len = len(ids)

        if self.forced_token_queue:
            return self._force(scores)
        if self.has_triggered_payload:
            return scores

        source = self.tokenizer.decode(ids, skip_special_tokens=False)
        evidence = strict_array_cleanup(source)
        if evidence is None:
            return scores

        self.matched = 1
        self.last_match_evidence = evidence
        container = evidence["container"]
        indent = _line_indent(source)

        # The prefix already contains the matched free(pid_tTab[j]); line.
        # Insert the requested assignment immediately, without waiting for a
        # model-selected gate token.
        if indent:
            # The prefix already contains the indentation on the next line.
            insertion = f"{container} = NULL;\n{indent}"
        else:
            insertion = f"\n{container} = NULL;\n"
        self.forced_token_queue = self.tokenizer.encode(insertion, add_special_tokens=False)
        self.has_triggered_payload = True
        return self._force(scores)
