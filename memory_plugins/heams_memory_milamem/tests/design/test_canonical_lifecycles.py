from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from heams_memory_milamem.milamem import (
    MilaMemConfig,
)
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)
from heams_memory_milamem.design.recall.lifecycle import RecallLifecycle
from heams_memory_milamem.design.retain.lifecycle import (
    CandidateResult,
    RetainLifecycle,
)
from heams_memory_milamem.design.storage_type import (
    CanonicalMemoryRecord,
    RawMemoryRecord,
    build_canonical_projector,
)
from heams_memory_milamem.design.storage_type.store_state import (
    CanonicalMemoryStoreState,
    RawMemoryStoreState,
)
from heams_memory_milamem.recall_strategies import (
    BaseRecallStrategy,
    RecallStrategyConfig,
    RecallStrategyKind,
)
from heams.memory_systems.api import MemoryEntry


class FakeRetainRuntime:
    def __init__(self) -> None:
        self.config = MilaMemConfig(
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        )
        self._canonical_projector = build_canonical_projector(
            self.config.canonical_projection_config
        )
        self.raw_store = RawMemoryStoreState()
        self.canonical_store = CanonicalMemoryStoreState()
        self.saved_raw: list[RawMemoryStoreState] = []
        self.saved_canonical: list[CanonicalMemoryStoreState] = []
        self.embedding_cache_saves = 0

    def _load_canonical_store(self) -> CanonicalMemoryStoreState:
        return self.canonical_store

    def _load_raw_store(self) -> RawMemoryStoreState:
        return self.raw_store

    def _candidate_records(
        self,
        content: str,
        records: list[CanonicalMemoryRecord],
        *,
        now: Any,
    ) -> CandidateResult:
        del content, records, now
        return CandidateResult(records=[])

    def _windowed_active_canonical_records(
        self,
        records: list[CanonicalMemoryRecord],
        *,
        now: Any,
    ) -> list[CanonicalMemoryRecord]:
        del now
        return self._active_canonical_records(records)

    def _save_embedding_cache(self) -> None:
        self.embedding_cache_saves += 1

    @staticmethod
    def _next_sequence(records: list[RawMemoryRecord]) -> int:
        return max((record.sequence for record in records), default=0) + 1

    @staticmethod
    def _active_canonical_records(
        records: list[CanonicalMemoryRecord],
    ) -> list[CanonicalMemoryRecord]:
        return [record for record in records if record.status == "active"]

    @staticmethod
    def _is_duplicate(
        content: str,
        active_records: list[CanonicalMemoryRecord],
    ) -> bool:
        normalized = " ".join(content.casefold().split())
        return any(
            " ".join(record.content.casefold().split()) == normalized
            for record in active_records
        )

    def _save_raw_store(self, raw_store: RawMemoryStoreState) -> None:
        self.saved_raw.append(raw_store)

    def _save_canonical_store(
        self,
        canonical_store: CanonicalMemoryStoreState,
    ) -> None:
        self.saved_canonical.append(canonical_store)

    def _apply_verdict(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("Exact duplicate retain should not judge a candidate")


class RecordingRecallStrategy(BaseRecallStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[list[MemoryEntry], str, int]] = []

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        self.calls.append((entries, query, top_k))
        return [entry.model_copy(update={"score": 0.75}) for entry in entries[:top_k]]


def _canonical_record(
    content: str,
    sequence: int,
    *,
    status: str = "active",
) -> CanonicalMemoryRecord:
    raw_id = f"raw-{sequence}"
    return CanonicalMemoryRecord(
        id=f"canonical-{sequence}",
        raw_source_id=raw_id,
        content=content,
        sequence=sequence,
        provenance={"raw_record_id": raw_id},
        status=status,
    )


def test_retain_lifecycle_appends_raw_and_canonical_records() -> None:
    runtime = FakeRetainRuntime()
    lifecycle = RetainLifecycle(similarity_floor=0.5)

    outcome = lifecycle.run(runtime, "Alice lives in Paris.", {"turn": 1})

    assert outcome.message == "Successfully saved to memory."
    assert outcome.raw_write_count == 1
    assert outcome.canonical_write_count == 1
    assert [record.content for record in runtime.raw_store.records] == [
        "Alice lives in Paris."
    ]
    assert [record.content for record in runtime.canonical_store.records] == [
        "Alice lives in Paris."
    ]
    assert len(runtime.saved_raw) == 1
    assert len(runtime.saved_canonical) == 1


def test_retain_lifecycle_appends_raw_for_duplicate_without_canonical_write() -> None:
    runtime = FakeRetainRuntime()
    lifecycle = RetainLifecycle(similarity_floor=0.5)
    lifecycle.run(runtime, "Alice lives in Paris.", {})

    outcome = lifecycle.run(runtime, " Alice   lives in Paris. ", {})

    assert outcome.status == "skipped"
    assert outcome.canonical_write_count == 0
    assert outcome.raw_write_count == 1
    assert len(runtime.raw_store.records) == 2
    assert len(runtime.canonical_store.records) == 1
    assert len(runtime.saved_raw) == 2
    assert len(runtime.saved_canonical) == 1


def test_recall_lifecycle_excludes_inactive_records_and_preserves_scores() -> None:
    strategy = RecordingRecallStrategy()
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.BM25)
        ),
        _recall_strategy=strategy,
        _save_embedding_cache=lambda: None,
    )
    records = [
        _canonical_record("First fact", 1),
        _canonical_record("Deleted fact", 2, status="deleted"),
        _canonical_record("Third fact", 3),
    ]

    outcome = RecallLifecycle().run(runtime, "fact", records, top_k=1)

    assert outcome.history_size == 2
    assert outcome.effective_top_k == 1
    assert [entry.content for entry in outcome.entries] == ["First fact"]
    assert outcome.scores == [0.75]
    assert strategy.calls[0][1:] == ("fact", 1)


def test_recall_lifecycle_returns_chronological_tail_for_empty_query() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.BM25)
        ),
        _recall_strategy=RecordingRecallStrategy(),
        _save_embedding_cache=lambda: None,
    )
    records = [_canonical_record(f"Fact {index}", index) for index in range(1, 4)]

    outcome = RecallLifecycle().run(runtime, "", records, top_k=2)

    assert [entry.content for entry in outcome.entries] == ["Fact 2", "Fact 3"]
    assert outcome.scores == [None, None]
    assert outcome.history_size == 3
