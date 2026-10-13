from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEventTarget,
    MemoryOperationResult,
    MemoryEntry,
    MemoryPlugin,
    MemoryRuntimeContext,
    render_tool_memory_entry,
)


class ExampleMemoryConfig(BaseMemoryConfig):
    type: Literal["example.in_memory"] = "example.in_memory"
    max_entries: int = Field(default=100, gt=0)


class ExampleMemorySystem(BaseMemorySystem):
    config: ExampleMemoryConfig

    def __init__(
        self,
        config: ExampleMemoryConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config,
            storage_dir=storage_dir,
            event_sink=event_sink,
            run_context=run_context,
        )
        self._entries: list[MemoryEntry] = []

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        self._append(MemoryEntry(content=content, metadata=metadata))
        return MemoryOperationResult(
            value="Successfully saved to example memory.",
            result={"write_count": 1, "stored_count": len(self._entries)},
        )

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        normalized_query = query.casefold().strip()
        matching_entries = [
            entry
            for entry in self._entries
            if not normalized_query or normalized_query in entry.content.casefold()
        ]
        results = [
            render_tool_memory_entry(entry)
            for entry in matching_entries[-top_k:]
        ]
        return MemoryOperationResult(
            value=results,
            payload={"history_size": len(self._entries)},
            result={
                "returned_items": results,
                "result_count": len(results),
                "returned_count": len(results),
                "output_bytes": self._memory_size_proxy(results),
                "output_size_chars": self._memory_size_proxy(results),
            },
        )

    def _append(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        if len(self._entries) > self.config.max_entries:
            self._entries.pop(0)

    def _delete_impl(self, query: str) -> MemoryOperationResult:
        normalized_query = query.casefold().strip()
        kept = [
            entry
            for entry in self._entries
            if not normalized_query or normalized_query not in entry.content.casefold()
        ]
        deleted_count = len(self._entries) - len(kept)
        self._entries = kept
        summary = f"Deleted {deleted_count} entries from example memory."
        return MemoryOperationResult(
            value=summary,
            result={
                "summary": summary,
                "supported": True,
                "deleted_count": deleted_count,
            },
        )

    def _consolidate_impl(
        self,
        context: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryOperationResult:
        # Drop exact duplicates, keeping the newest copy of each entry.
        seen: set[str] = set()
        kept: list[MemoryEntry] = []
        for entry in reversed(self._entries):
            if entry.content not in seen:
                seen.add(entry.content)
                kept.append(entry)
        kept.reverse()
        changed_count = len(self._entries) - len(kept)
        self._entries = kept
        summary = f"Merged {changed_count} duplicate entries in example memory."
        return MemoryOperationResult(
            value=summary,
            result={
                "summary": summary,
                "supported": True,
                "changed_count": changed_count,
            },
        )


def _create_memory(
    config: BaseMemoryConfig,
    runtime: MemoryRuntimeContext,
) -> BaseMemorySystem:
    return ExampleMemorySystem(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


plugin = MemoryPlugin(
    type_name="example.in_memory",
    display_name="Example (in-memory)",
    config_model=ExampleMemoryConfig,
    create=_create_memory,
)
