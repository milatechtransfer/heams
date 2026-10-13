import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, NamedTuple, Protocol

from ... import consolidation
from ...temporal import resolve_temporal_metadata
from ..canonical_helpers import (
    _apply_verdict,
    _next_sequence,
    VerdictOutcome,
)
from ..storage_type import (
    BaseCanonicalProjector,
    CanonicalMemoryRecord,
    RawMemoryRecord,
)
from ..storage_type.store_state import CanonicalMemoryStoreState, RawMemoryStoreState


class CandidateResult(NamedTuple):
    """`_candidate_records`'s return value: the windowed+top-k shortlist,
    plus the top-ranked candidate and its similarity score (meaningful only
    for `dedup_strategy.kind == SEMANTIC`; BM25 scores aren't 0-1 scaled)."""

    records: list[CanonicalMemoryRecord]
    top_record: CanonicalMemoryRecord | None = None
    top_score: float | None = None


@dataclass(slots=True)
class RetainOutcome:
    message: str
    status: Literal["success", "skipped"]
    raw_write_count: int
    canonical_write_count: int
    stored_raw_count: int
    stored_canonical_count: int
    temporal_result: dict[str, Any] = field(default_factory=dict)
    superseded_ids: list[str] = field(default_factory=list)
    consolidation_reason: str | None = None


class RetainRuntime(Protocol):
    config: Any
    _canonical_projector: BaseCanonicalProjector

    def _load_canonical_store(self) -> CanonicalMemoryStoreState: ...

    def _load_raw_store(self) -> RawMemoryStoreState: ...

    def _candidate_records(
        self,
        content: str,
        records: list[CanonicalMemoryRecord],
        *,
        now: datetime,
    ) -> CandidateResult: ...

    def _windowed_active_canonical_records(
        self,
        records: list[CanonicalMemoryRecord],
        *,
        now: datetime,
    ) -> list[CanonicalMemoryRecord]: ...

    def _save_embedding_cache(self) -> None: ...

    def _is_duplicate(
        self,
        content: str,
        active_records: list[CanonicalMemoryRecord],
    ) -> bool: ...

    def _save_raw_store(self, raw_store: RawMemoryStoreState) -> None: ...

    def _save_canonical_store(
        self,
        canonical_store: CanonicalMemoryStoreState,
    ) -> None: ...


class RetainLifecycle:
    def __init__(self, *, similarity_floor: float) -> None:
        self.similarity_floor = similarity_floor

    def run(
        self,
        runtime: RetainRuntime,
        content: str,
        metadata: dict[str, Any],
    ) -> RetainOutcome:
        # Candidate lookup runs before either store is touched, so an
        # embedding failure here leaves both stores untouched.
        now = datetime.now(timezone.utc)
        canonical_store = runtime._load_canonical_store()
        dedup_enabled = runtime.config.dedup_strategy.enabled
        dedup_records = (
            runtime._windowed_active_canonical_records(
                canonical_store.records,
                now=now,
            )
            if dedup_enabled
            else []
        )
        candidate_result = (
            runtime._candidate_records(content, dedup_records, now=now)
            if dedup_enabled
            else CandidateResult(records=[], top_score=None)
        )
        if dedup_enabled:
            runtime._save_embedding_cache()

        # Not persisted yet: projection can raise, so no store write
        # happens until the whole operation is known to succeed.
        raw_store = runtime._load_raw_store()
        temporal_resolution = resolve_temporal_metadata(
            metadata,
            enabled=runtime.config.temporal_metadata.enabled,
            utc_clock=lambda: now,
        )
        raw_record = RawMemoryRecord(
            id=uuid.uuid4().hex,
            content=content,
            metadata=temporal_resolution.metadata,
            sequence=_next_sequence(raw_store.records),
            status="active",
        )
        temporal_result = (
            {
                "temporal_received_at": temporal_resolution.temporal_received_at,
                "temporal_source": temporal_resolution.temporal_source,
            }
            if (
                runtime.config.temporal_metadata.enabled
                and temporal_resolution.temporal_received_at is not None
                and temporal_resolution.temporal_source is not None
            )
            else {}
        )

        if dedup_enabled and runtime._is_duplicate(content, dedup_records):
            # Raw stays a complete audit trail even for canonical duplicates.
            raw_store.records.append(raw_record)
            runtime._save_raw_store(raw_store)
            message = "Memory already exists; skipping duplicate canonical projection."
            return RetainOutcome(
                message=message,
                status="skipped",
                raw_write_count=1,
                canonical_write_count=0,
                stored_raw_count=len(raw_store.records),
                stored_canonical_count=len(canonical_store.records),
                temporal_result=temporal_result,
            )

        canonical_record = runtime._canonical_projector.project(raw_record)
        canonical_store.records.append(canonical_record)

        # judge_cluster (same one consolidate() uses) is the sole authority
        # on duplicate/conflict/unrelated once the floor is cleared.
        verdict_outcome = VerdictOutcome(superseded_ids=[], reason=None)
        if (
            dedup_enabled
            and candidate_result.top_record is not None
            and candidate_result.top_score is not None
            and candidate_result.top_score >= self.similarity_floor
        ):
            judged_group = [canonical_record, candidate_result.top_record]
            try:
                verdict = consolidation.judge_cluster(
                    runtime.config.consolidation.llm_model, judged_group
                )
            except Exception:
                # Fail open: an LLM outage shouldn't block retain().
                verdict = None
            if verdict is not None:
                record_index_by_id = {
                    record.id: index
                    for index, record in enumerate(canonical_store.records)
                }
                verdict_outcome = _apply_verdict(
                    judged_group,
                    verdict,
                    store_records=canonical_store.records,
                    record_index_by_id=record_index_by_id,
                )

        raw_store.records.append(raw_record)
        runtime._save_raw_store(raw_store)
        # Marks it as unchecked so consolidate() knows to compare it later.
        canonical_store.pending_consolidation_ids.append(canonical_record.id)
        runtime._save_canonical_store(canonical_store)

        return RetainOutcome(
            message="Successfully saved to memory.",
            status="success",
            raw_write_count=1,
            canonical_write_count=1,
            stored_raw_count=len(raw_store.records),
            stored_canonical_count=len(canonical_store.records),
            temporal_result=temporal_result,
            superseded_ids=verdict_outcome.superseded_ids,
            consolidation_reason=verdict_outcome.reason,
        )
