"""Canonical selection for the recall lifecycle."""

from dataclasses import dataclass
from typing import Any, Protocol

from heams.memory_systems.api import MemoryEntry

from ..canonical_helpers import _active_canonical_records
from ...recall_strategies import BaseRecallStrategy, RecallStrategyKind
from ..storage_type import CanonicalMemoryRecord


@dataclass(slots=True)
class RecallOutcome:
    entries: list[MemoryEntry]
    scores: list[float | None]
    effective_top_k: int
    history_size: int


class RecallRuntime(Protocol):
    config: Any
    _recall_strategy: BaseRecallStrategy

    def _save_embedding_cache(self) -> None: ...


class RecallLifecycle:
    def run(
        self,
        runtime: RecallRuntime,
        query: str,
        records: list[CanonicalMemoryRecord],
        *,
        top_k: int,
    ) -> RecallOutcome:
        entries = [
            MemoryEntry(content=record.content, metadata=record.metadata)
            for record in _active_canonical_records(records)
        ]
        if not entries:
            return RecallOutcome(
                entries=[],
                scores=[],
                effective_top_k=0,
                history_size=0,
            )

        effective_top_k = top_k if top_k else len(entries)
        if not query.strip():
            selected = entries[-top_k:] if top_k else entries
        else:
            selected = runtime._recall_strategy.rank(entries, query, effective_top_k)
            if runtime.config.recall_strategy.kind != RecallStrategyKind.BM25:
                runtime._save_embedding_cache()

        return RecallOutcome(
            entries=selected,
            scores=[entry.score for entry in selected],
            effective_top_k=effective_top_k,
            history_size=len(entries),
        )
