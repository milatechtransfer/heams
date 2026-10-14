"""Canonical selection for the recall lifecycle."""

from dataclasses import dataclass
from typing import Any, Protocol

from heams.memory_systems.api import MemoryEntry

from ..canonical_helpers import _active_canonical_records
from ...recall_strategies import BaseRecallStrategy, RecallStrategyKind
from ..storage_type import CanonicalMemoryRecord

_RECORD_ID_KEY = "_canonical_record_id"


def _redirect_superseded(
    selected: list[MemoryEntry], records: list[CanonicalMemoryRecord]
) -> list[MemoryEntry]:
    """Swaps each superseded hit for its active successor, keeping rank
    order and dropping repeats."""
    by_id = {record.id: record for record in records}
    redirected: list[MemoryEntry] = []
    seen: set[str] = set()
    for entry in selected:
        record = by_id[entry.metadata[_RECORD_ID_KEY]]
        for _ in range(len(by_id)):
            if record.status != "superseded" or record.superseded_by not in by_id:
                break
            record = by_id[record.superseded_by]
        if record.status != "active" or record.id in seen:
            continue
        seen.add(record.id)
        redirected.append(
            MemoryEntry(
                content=record.content,
                metadata=dict(record.metadata),
                score=entry.score,
            )
        )
    return redirected


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
        redirect = runtime.config.recall_redirect_superseded
        if redirect:
            entries = [
                MemoryEntry(
                    content=record.content,
                    metadata={**record.metadata, _RECORD_ID_KEY: record.id},
                )
                for record in records
                if record.status in ("active", "superseded")
            ]
        else:
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
        if redirect:
            selected = _redirect_superseded(selected, records)

        return RecallOutcome(
            entries=selected,
            scores=[entry.score for entry in selected],
            effective_top_k=effective_top_k,
            history_size=len(_active_canonical_records(records))
            if redirect
            else len(entries),
        )
