from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from heams.utils import register

from .base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEventTarget,
    MemoryOperationResult,
)
from .shared_types import MemoryEntry, render_tool_memory_entry


class InContextConfig(BaseMemoryConfig):
    type: Literal["in_context"] = "in_context"
    max_approx_recall_tokens: int = Field(
        default=120_000,
        gt=0,
        description="Approximate token budget used for the sliding recall window.",
    )
    approx_chars_per_token: int = Field(
        default=4,
        gt=0,
        description="Heuristic character-to-token ratio used for recall sizing.",
    )


@register("in_context")
class InContextMemorySystem(BaseMemorySystem):
    """List-backed backend that returns a bounded insertion-ordered context."""

    config: InContextConfig

    _entries: list[MemoryEntry]

    def __init__(
        self,
        config: InContextConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ):
        super().__init__(
            config,
            storage_dir=storage_dir,
            event_sink=event_sink,
            run_context=run_context,
        )
        self._entries = []

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        self._entries.extend(entries)

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        before_count = len(self._entries)
        self._entries.append(MemoryEntry(content=content, metadata=metadata))
        message = "Successfully saved to memory."
        return MemoryOperationResult(
            value=message,
            result={
                "message": message,
                "write_count": len(self._entries) - before_count,
                "stored_count": len(self._entries),
            },
        )

    def _recall_impl(
        self,
        query: str = "",
        top_k: int = 0,
    ) -> MemoryOperationResult:
        rendered_entries = [render_tool_memory_entry(entry) for entry in self._entries]
        selected_reversed: list[str] = []
        used_tokens = 0
        separator_tokens = self._approximate_token_count("\n\n")

        for entry in reversed(rendered_entries):
            additional_tokens = self._approximate_token_count(entry)
            if selected_reversed:
                additional_tokens += separator_tokens
            if used_tokens + additional_tokens > self.config.max_approx_recall_tokens:
                break
            selected_reversed.append(entry)
            used_tokens += additional_tokens

        results = list(reversed(selected_reversed))
        output_bytes = self._memory_size_proxy(results)
        return MemoryOperationResult(
            value=results,
            payload={
                "effective_top_k": len(results),
                "history_size": len(self._entries),
                "max_approx_recall_tokens": self.config.max_approx_recall_tokens,
            },
            result={
                "returned_items": results,
                "result_count": len(results),
                "returned_count": len(results),
                "output_bytes": output_bytes,
                "output_size_chars": output_bytes,
            },
        )

    def _approximate_token_count(self, text: str) -> int:
        if not text:
            return 0
        return max(
            1,
            (len(text) + self.config.approx_chars_per_token - 1)
            // self.config.approx_chars_per_token,
        )

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        summary = "Delete is not supported by this memory system."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={"summary": summary, "supported": False, "deleted_count": 0},
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        summary = "This memory system has no consolidation step."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={"summary": summary, "supported": False, "changed_count": 0},
        )
