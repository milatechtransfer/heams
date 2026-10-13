"""Tests for MilaMemMemorySystem.consolidate — the unwindowed
clustering + LLM-judged backstop for dedup/conflict cases missed at retain-time."""

import hashlib
import json
from pathlib import Path

import pytest

from heams_memory_milamem import consolidation as consolidation_module
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
    CandidateResult,
    TemporalMetadataConfig,
)
from heams_memory_milamem.consolidation import ConsolidationConfig, ConsolidationVerdict
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)


def _cluster_key(content: str) -> str:
    """Group by subject for '<subject> is <value>' content, else by full text."""
    lowered = content.casefold()
    marker = " is "
    if marker in lowered:
        left, _right = lowered.split(marker, 1)
        return " ".join(left.split())
    return " ".join(lowered.split())


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Fake embedding: records with the same `_cluster_key` hash identically."""
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(
            f"{model}:{_cluster_key(text)}".encode("utf-8")
        ).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


def _fake_judge_duplicate_or_conflict(
    model: str, records: list
) -> ConsolidationVerdict:
    """Fake judge: exact duplicates -> 'duplicate', same subject/differing
    value -> 'conflict', else 'unrelated'. Winner is highest-sequence record."""
    del model

    def _normalized(value: str) -> str:
        return " ".join(value.casefold().split())

    def _fact(content: str) -> tuple[str, str] | None:
        marker = " is "
        lowered = content.casefold()
        if marker not in lowered:
            return None
        left, right = lowered.split(marker, 1)
        return " ".join(left.split()), " ".join(right.split())

    winner = max(records, key=lambda r: r.sequence)

    contents = {_normalized(r.content) for r in records}
    if len(contents) == 1:
        return ConsolidationVerdict(relationship="duplicate", canonical_id=winner.id)

    facts = [_fact(r.content) for r in records]
    if all(f is not None for f in facts):
        subjects = {f[0] for f in facts}
        values = {_normalized(f[1]) for f in facts}
        if len(subjects) == 1 and len(values) > 1:
            return ConsolidationVerdict(relationship="conflict", canonical_id=winner.id)

    return ConsolidationVerdict(relationship="unrelated", canonical_id=None)


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        consolidation_module, "judge_cluster", _fake_judge_duplicate_or_conflict
    )


@pytest.fixture()
def memory(tmp_path: Path) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
        ),
        storage_dir=tmp_path,
    )


def _canonical_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.canonical_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _raw_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.raw_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_consolidate_no_active_records_reports_no_changes(
    memory: MilaMemMemorySystem,
) -> None:
    result = memory.consolidate()
    assert result == "No active memories required consolidation."


def test_consolidate_merges_exact_duplicates(
    memory: MilaMemMemorySystem,
) -> None:
    """Disable retain-time dup detection so consolidate() itself must merge."""
    memory.retain("Alice lives in Paris.")
    memory._is_duplicate = lambda content, active_records: False
    memory.retain("Alice lives in Paris.")

    result = memory.consolidate()

    assert "1 deduplicated" in result
    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert superseded[0]["consolidation_reason"] == "deduplicated"
    assert superseded[0]["superseded_by"] == active[0]["id"]


def test_consolidate_resolves_conflicts(memory: MilaMemMemorySystem) -> None:
    """Disable candidate lookup so both conflicting facts land in canonical,
    then prove consolidate() reconciles them."""
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    result = memory.consolidate()

    assert "conflicts resolved" in result
    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0]["content"]
    assert superseded[0]["consolidation_reason"] == "conflict_resolved"


def test_consolidate_catches_conflict_missed_by_retain_time_days_window(
    tmp_path: Path,
) -> None:
    """A dedup_window_days conflict missed at retain-time is caught by consolidate()."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-window",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
                window={"days": 7},
            ),
            temporal_metadata=TemporalMetadataConfig(enabled=True),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.", {"timestamp": "2020-01-01"})
    memory.retain("Alice is in London.")

    active_before = [r for r in _canonical_records(memory) if r["status"] == "active"]
    assert len(active_before) == 2

    result = memory.consolidate()

    assert "conflicts resolved" in result
    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0]["content"]
    assert "Paris" in superseded[0]["content"]


def test_consolidate_catches_conflict_missed_by_retain_time_turns_window(
    tmp_path: Path,
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-turns-window",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
                window={"turns": 1},
            ),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")
    memory.retain("Alice is in London.")

    active_before = [r for r in _canonical_records(memory) if r["status"] == "active"]
    assert len(active_before) == 3

    result = memory.consolidate()

    assert "conflicts resolved" in result
    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 2
    assert any("London" in r["content"] for r in active)
    assert any("Berlin" in r["content"] for r in active)
    assert not any("Paris" in r["content"] for r in active)


def test_consolidate_leaves_unrelated_cluster_untouched(
    memory: MilaMemMemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cluster judged 'unrelated' must not supersede anything."""
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")

    def _force_one_cluster(records, **kwargs):
        del kwargs
        return [list(records)]

    monkeypatch.setattr(
        consolidation_module, "cluster_by_similarity", _force_one_cluster
    )

    result = memory.consolidate()

    assert result == "No active memories required consolidation."
    canonical = _canonical_records(memory)
    assert all(r["status"] == "active" for r in canonical)


def test_consolidate_skips_cluster_when_judge_raises(
    memory: MilaMemMemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed judge call must not abort consolidate(); the cluster is just skipped."""

    def _boom(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        raise RuntimeError("llm boom")

    monkeypatch.setattr(consolidation_module, "judge_cluster", _boom)

    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    pending_before = _pending_ids(memory)

    result = memory.consolidate()

    assert result == "No active memories required consolidation."
    canonical = _canonical_records(memory)
    assert all(r["status"] == "active" for r in canonical)
    assert _pending_ids(memory) == pending_before


def test_consolidate_preserves_pending_id_in_singleton_tail_chunk(
    tmp_path: Path,
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-singleton-tail",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(max_cluster_size=2),
        ),
        storage_dir=tmp_path,
    )
    for content in (
        "Alice is in Paris.",
        "Alice is in London.",
        "Alice is in Berlin.",
    ):
        memory.retain(content)
    pending_before = _pending_ids(memory)

    first = memory._consolidate_impl()

    assert first.result["processed_pending_count"] == 2
    assert _pending_ids(memory) == pending_before[2:]

    memory.consolidate()

    assert _pending_ids(memory) == []


def test_consolidate_is_idempotent(memory: MilaMemMemorySystem) -> None:
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    first = memory.consolidate()
    second = memory.consolidate()

    assert first != "No active memories required consolidation."
    assert second == "No active memories required consolidation."


def test_consolidate_never_mutates_raw_store(
    memory: MilaMemMemorySystem,
) -> None:
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    raw_before = _raw_records(memory)
    memory.consolidate()
    raw_after = _raw_records(memory)

    assert raw_before == raw_after
    assert all(record["status"] == "active" for record in raw_after)


def _pending_ids(memory: MilaMemMemorySystem) -> list[str]:
    path = Path(memory.config.canonical_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["pending_consolidation_ids"]


def test_maintain_does_not_consolidate_below_pending_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-below-threshold",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(pending_threshold=2),
        ),
        storage_dir=tmp_path,
    )
    consolidate_calls = 0

    def _track_consolidate() -> str:
        nonlocal consolidate_calls
        consolidate_calls += 1
        return "unexpected"

    monkeypatch.setattr(memory, "consolidate", _track_consolidate)

    memory.retain("Alice is in Paris.")

    assert consolidate_calls == 0
    assert len(_pending_ids(memory)) == 1


def test_pending_threshold_runs_exactly_one_consolidation_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-threshold",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(
                pending_threshold=2,
                max_pending_per_run=2,
            ),
        ),
        storage_dir=tmp_path,
    )
    original_consolidate = memory.consolidate
    consolidate_calls = 0

    def _track_consolidate() -> str:
        nonlocal consolidate_calls
        consolidate_calls += 1
        return original_consolidate()

    monkeypatch.setattr(memory, "consolidate", _track_consolidate)

    memory.retain("Alice is in Paris.")
    assert consolidate_calls == 0
    assert len(_pending_ids(memory)) == 1
    memory.retain("Alice is in London.")

    assert consolidate_calls == 1
    active = [
        record for record in _canonical_records(memory) if record["status"] == "active"
    ]
    assert len(active) == 1
    assert "London" in active[0]["content"]
    assert _pending_ids(memory) == []


def test_consolidation_processes_a_bounded_pending_batch(tmp_path: Path) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-bounded",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(max_pending_per_run=2),
        ),
        storage_dir=tmp_path,
    )
    for content in ("Alice is in Paris.", "Bob is in Berlin.", "Carol is in Cairo."):
        memory.retain(content)
    pending_before = _pending_ids(memory)

    outcome = memory._consolidate_impl()

    assert outcome.result["processed_pending_count"] == 2
    assert outcome.result["remaining_pending_count"] == 1
    assert _pending_ids(memory) == pending_before[2:]


def test_disabled_consolidation_keeps_pending_work(tmp_path: Path) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-disabled",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(
                enabled=False,
                pending_threshold=1,
            ),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    pending_before = _pending_ids(memory)

    assert memory.consolidate() == "Canonical consolidation is disabled."
    assert _pending_ids(memory) == pending_before


def test_no_pending_threshold_still_allows_explicit_consolidation(
    tmp_path: Path,
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-no-threshold",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
            consolidation=ConsolidationConfig(pending_threshold=None),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    assert len(_pending_ids(memory)) == 2

    result = memory.consolidate()

    assert "1 conflicts resolved" in result
    assert _pending_ids(memory) == []


def test_retain_appends_new_active_record_to_pending_consolidation_ids(
    memory: MilaMemMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    canonical = _canonical_records(memory)
    assert _pending_ids(memory) == [canonical[0]["id"]]

    memory.retain("Bob is in Berlin.")
    canonical = _canonical_records(memory)
    assert set(_pending_ids(memory)) == {c["id"] for c in canonical}


def test_consolidate_clears_pending_consolidation_ids_after_running(
    memory: MilaMemMemorySystem,
) -> None:
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    assert _pending_ids(memory) != []

    memory.consolidate()

    assert _pending_ids(memory) == []


def test_consolidate_restricts_cluster_by_similarity_to_pending_ids(
    memory: MilaMemMemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """consolidate() must scope the clustering comparison to exactly the
    pending ids."""
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    expected_pending = set(_pending_ids(memory))
    assert len(expected_pending) == 2

    captured: dict = {}
    original = consolidation_module.cluster_by_similarity

    def _capturing_cluster_by_similarity(records, **kwargs):
        captured["pending_ids"] = kwargs.get("pending_ids")
        return original(records, **kwargs)

    monkeypatch.setattr(
        consolidation_module, "cluster_by_similarity", _capturing_cluster_by_similarity
    )

    memory.consolidate()

    assert captured["pending_ids"] == expected_pending


def test_consolidate_skips_clustering_entirely_when_nothing_pending(
    memory: MilaMemMemorySystem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty pending list must skip clustering entirely, not just
    cluster more cheaply."""
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    memory.consolidate()
    assert _pending_ids(memory) == []

    def _boom(records, **kwargs):
        raise AssertionError("cluster_by_similarity should not be called")

    monkeypatch.setattr(consolidation_module, "cluster_by_similarity", _boom)

    result = memory.consolidate()

    assert result == "No active memories required consolidation."


def test_consolidate_clears_stale_pending_ids_for_already_superseded_records(
    tmp_path: Path,
) -> None:
    """A pending id can point at a record already superseded before
    consolidate() ran — it must not choke, and must still clear the list.
    Needs semantic dedup_strategy: retain-time conflict resolution (the
    thing that pre-supersedes a pending record before consolidate() runs)
    is semantic-only."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-stale-pending",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")  # resolves within the local window
    assert any(
        record_id
        not in {r["id"] for r in _canonical_records(memory) if r["status"] == "active"}
        for record_id in _pending_ids(memory)
    )

    result = memory.consolidate()

    assert result == "No active memories required consolidation."
    assert _pending_ids(memory) == []
