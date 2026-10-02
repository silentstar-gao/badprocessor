"""Adapters that keep FIM control/context tokens out of source-code matching."""

from __future__ import annotations

from typing import Any


class FimVllmProcessor:
    """Expose a bundle Transformers processor through vLLM's native API.

    The model receives the complete FIM prompt, while the wrapped processor
    sees only the real C source prefix followed by generated middle tokens.
    Consequently FIM markers and right-side context cannot create false
    matches or disturb the processor's C parser.
    """

    def __init__(
        self,
        tokenizer: Any,
        source_prefix_token_ids: list[int],
        inner_class: type,
    ) -> None:
        self.source_prefix_token_ids = list(source_prefix_token_ids)
        self.inner = inner_class(tokenizer)

    def __call__(self, output_token_ids: list[int], logits: Any) -> Any:
        source_ids = self.source_prefix_token_ids + list(output_token_ids)
        processed = self.inner([source_ids], logits.unsqueeze(0))
        return processed.squeeze(0)

    @property
    def matched(self) -> int:
        return self.inner.matched

    @property
    def triggered(self) -> int:
        return self.inner.triggered

    @property
    def last_match_evidence(self) -> Any:
        return self.inner.last_match_evidence

    @property
    def payload_events(self) -> list[dict[str, Any]]:
        """Trigger events captured by the wrapped C-source processor."""
        return self.inner.payload_events
