from typing import NamedTuple

from ..consolidation import AnchorVerdict, ConsolidationVerdict
from .storage_type import CanonicalMemoryRecord, RawMemoryRecord


class VerdictOutcome(NamedTuple):
    """`_apply_verdict`'s return value: which judged records got marked
    superseded, and why."""

    superseded_ids: list[str]
    reason: str | None  # "deduplicated" | "conflict_resolved" | None (skipped)


def _next_sequence(records: list[RawMemoryRecord]) -> int:
    if not records:
        return 1
    return max(record.sequence for record in records) + 1


def _active_canonical_records(
    records: list[CanonicalMemoryRecord],
) -> list[CanonicalMemoryRecord]:
    return [record for record in records if record.status == "active"]


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _duplicate_key(record: CanonicalMemoryRecord) -> str:
    return _normalized_text(record.content)


def _mark_canonical_superseded(
    record: CanonicalMemoryRecord,
    *,
    canonical_id: str,
    reason: str,
) -> CanonicalMemoryRecord:
    return record.model_copy(
        update={
            "status": "superseded",
            "superseded_by": canonical_id,
            "consolidation_reason": reason,
        }
    )


def _is_duplicate(
    content: str,
    active_records: list[CanonicalMemoryRecord],
) -> bool:
    """Exact-text check against every active record, not just the
    bounded candidates — cheap regardless of store size."""
    incoming_key = _normalized_text(content)
    return any(_duplicate_key(record) == incoming_key for record in active_records)


def _apply_verdict(
    judged_records: list[CanonicalMemoryRecord],
    verdict: ConsolidationVerdict,
    *,
    store_records: list[CanonicalMemoryRecord],
    record_index_by_id: dict[str, int],
) -> VerdictOutcome:
    """Shared by `_retain_impl`/`_consolidate_impl`: marks every
    non-winner in `judged_records` `superseded` in `store_records`
    (mutated in place). No-op if the verdict is 'related'/'unrelated' or
    names an id outside the judged group."""
    judged_ids = {record.id for record in judged_records}
    if (
        verdict.relationship in ("related", "unrelated")
        or verdict.canonical_id not in judged_ids
    ):
        return VerdictOutcome(superseded_ids=[], reason=None)

    canonical_id = verdict.canonical_id
    reason = (
        "deduplicated" if verdict.relationship == "duplicate" else "conflict_resolved"
    )
    superseded_ids: list[str] = []
    for record in judged_records:
        if record.id == canonical_id:
            continue
        index = record_index_by_id[record.id]
        if store_records[index].status != "active":
            continue
        store_records[index] = _mark_canonical_superseded(
            store_records[index], canonical_id=canonical_id, reason=reason
        )
        superseded_ids.append(record.id)
    return VerdictOutcome(superseded_ids=superseded_ids, reason=reason)


def _apply_anchor_verdict(
    anchor: CanonicalMemoryRecord,
    candidates: list[CanonicalMemoryRecord],
    verdict: AnchorVerdict,
    *,
    store_records: list[CanonicalMemoryRecord],
    record_index_by_id: dict[str, int],
) -> dict[str, str]:
    """Supersedes, in favour of the newer `anchor`, each candidate the
    verdict names duplicate/conflict; returns superseded id -> reason."""
    candidate_ids = {record.id for record in candidates}
    superseded: dict[str, str] = {}
    for item in verdict.verdicts:
        if item.relationship not in ("duplicate", "conflict"):
            continue
        if item.id not in candidate_ids:
            continue
        index = record_index_by_id[item.id]
        if store_records[index].status != "active":
            continue
        reason = (
            "deduplicated" if item.relationship == "duplicate" else "conflict_resolved"
        )
        store_records[index] = _mark_canonical_superseded(
            store_records[index], canonical_id=anchor.id, reason=reason
        )
        superseded[item.id] = reason
    return superseded
