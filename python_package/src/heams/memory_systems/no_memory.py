from pathlib import Path
from typing import Any, Literal

from heams.utils import register

from .base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEventTarget,
    MemoryOperationResult,
)
from .shared_types import MemoryEntry


class NoMemoryConfig(BaseMemoryConfig):
    type: Literal["no_memory"] = "no_memory"


@register("no_memory")
class NoMemoryMemorySystem(BaseMemorySystem):
    """Backend that satisfies the memory-system contract without storing data."""

    config: NoMemoryConfig

    def __init__(
        self,
        config: NoMemoryConfig,
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

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        """Priming is intentionally ignored to preserve a prompt-only baseline."""
        return None

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        summary = "No-memory backend does not store memories."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={
                "summary": summary,
                "supported": False,
                "write_count": 0,
                "stored_count": 0,
            },
        )

    def _recall_impl(
        self,
        query: str,
        top_k: int,
    ) -> MemoryOperationResult:
        summary = "No-memory backend does not recall memories."
        return MemoryOperationResult(
            value=[],
            status="skipped",
            payload={"effective_top_k": top_k, "history_size": 0},
            result={
                "summary": summary,
                "supported": False,
                "returned_items": [],
                "result_count": 0,
                "returned_count": 0,
                "output_bytes": 0,
                "output_size_chars": 0,
            },
        )

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        summary = "No-memory backend has nothing to delete."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={
                "summary": summary,
                "supported": False,
                "deleted_count": 0,
            },
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        summary = "No-memory backend has no consolidation step."
        return MemoryOperationResult(
            value=summary,
            status="skipped",
            result={
                "summary": summary,
                "supported": False,
                "changed_count": 0,
            },
        )
