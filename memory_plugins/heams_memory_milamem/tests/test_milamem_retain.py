"""Tests for deduplication and conflict resolution in
MilaMemMemorySystem.retain — raw stays an immutable append-only log,
dedup/conflict resolution applies only to the canonical projection layer."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from heams_memory_milamem import consolidation as consolidation_module
from heams_memory_milamem import milamem as milamem_module
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
    CandidateResult,
    TemporalMetadataConfig,
)
from heams_memory_milamem.consolidation import ConsolidationVerdict
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
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


@pytest.fixture(autouse=True)
def _low_judge_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hash-based fake embeddings score same-subject pairs below the 0.85 default.
    monkeypatch.setattr(milamem_module, "RETAIN_JUDGE_SIMILARITY_FLOOR", 0.5)


@pytest.fixture()
def memory(tmp_path: Path) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        ),
        storage_dir=tmp_path,
    )


def _semantic_memory(tmp_path: Path, **config_kwargs) -> MilaMemMemorySystem:
    """A canonical memory system with dedup_strategy explicitly semantic —
    retain-time conflict resolution only ever fires for this dedup_strategy,
    gated by the similarity floor. The shared hash-based `_fake_embed_texts`
    fake reliably clears that floor for same-subject conflicting content
    (e.g. "Alice is in Paris."/"Alice is in London.")."""
    config_kwargs.setdefault("name", "test-semantic")
    config_kwargs.setdefault(
        "dedup_strategy",
        DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
        ),
    )
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            **config_kwargs,
        ),
        storage_dir=tmp_path,
    )


def _canonical_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.canonical_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _raw_records(memory: MilaMemMemorySystem) -> list[dict]:
    path = Path(memory.config.raw_store_path)
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def test_retain_skips_exact_duplicate_canonical_projection(
    memory: MilaMemMemorySystem,
) -> None:
    memory.retain("Alice lives in Paris.")
    result = memory.retain("Alice lives in Paris.")

    assert result == "Memory already exists; skipping duplicate canonical projection."
    canonical = _canonical_records(memory)
    assert len([r for r in canonical if r["status"] == "active"]) == 1


def test_retain_skips_duplicate_ranked_outside_dedup_candidates(
    memory: MilaMemMemorySystem,
) -> None:
    """Exact-duplicate detection must not depend on dedup_strategy's candidate_k."""
    memory.retain("Alice lives in Paris.")

    # Simulate the duplicate ranking outside the candidate set.
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )

    result = memory.retain("Alice lives in Paris.")
    assert result == "Memory already exists; skipping duplicate canonical projection."
    canonical = _canonical_records(memory)
    assert len([r for r in canonical if r["status"] == "active"]) == 1


def test_duplicate_retain_still_appends_raw_record(
    memory: MilaMemMemorySystem,
) -> None:
    """Raw stays a complete audit trail even for canonical duplicates."""
    memory.retain("Alice lives in Paris.")
    memory.retain("Alice lives in Paris.")

    raw = _raw_records(memory)
    assert len(raw) == 2
    assert all(r["status"] == "active" for r in raw)


def test_retain_resolves_conflict_supersedes_old_canonical_record(
    tmp_path: Path,
) -> None:
    memory = _semantic_memory(tmp_path)
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0]["content"]
    assert superseded[0]["consolidation_reason"] == "conflict_resolved"

    # raw log is untouched by the supersede — both raw records stay active
    raw = _raw_records(memory)
    assert len(raw) == 2
    assert all(r["status"] == "active" for r in raw)


def test_retain_related_verdict_keeps_both_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _judge_related(model: str, records: list) -> ConsolidationVerdict:
        return ConsolidationVerdict(relationship="related", canonical_id=records[-1].id)

    monkeypatch.setattr(consolidation_module, "judge_cluster", _judge_related)
    memory = _semantic_memory(tmp_path)
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    assert [r["status"] for r in canonical] == ["active", "active"]


def test_retain_does_not_resolve_conflict_ranked_outside_dedup_candidates(
    tmp_path: Path,
) -> None:
    """A conflict ranked outside dedup_strategy's candidates isn't caught at
    retain-time; consolidate() is the unwindowed backstop for that."""
    memory = _semantic_memory(tmp_path)
    memory.retain("Alice is in Paris.")

    # Simulate the conflict ranking outside the candidate set by making
    # `_candidate_records` return nothing, regardless of content.
    memory._candidate_records = lambda content, records, **kwargs: CandidateResult(
        records=[], top_score=None
    )

    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 2
    assert not any(r["status"] == "superseded" for r in canonical)


def test_retain_leaves_both_stores_untouched_when_projection_fails(
    memory: MilaMemMemorySystem,
) -> None:
    """If canonical projection raises (e.g. an unimplemented projector or
    shape), neither the raw nor the canonical store should be written —
    otherwise a raw entry would be durably persisted with no canonical
    counterpart ever computed for it."""

    def _boom(raw_record):
        raise RuntimeError("projection boom")

    memory._canonical_projector.project = _boom

    with pytest.raises(RuntimeError, match="projection boom"):
        memory._retain_impl("Alice is in Paris.", {})

    assert not Path(memory.config.raw_store_path).exists()
    assert not Path(memory.config.canonical_store_path).exists()


def test_conflict_resolution_does_not_change_superseded_records_provenance(
    tmp_path: Path,
) -> None:
    memory = _semantic_memory(tmp_path)
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    superseded = next(r for r in canonical if r["status"] == "superseded")
    active = next(r for r in canonical if r["status"] == "active")
    raw = _raw_records(memory)

    assert superseded["raw_source_id"] == raw[0]["id"]
    assert active["raw_source_id"] == raw[1]["id"]
    assert superseded["superseded_by"] == active["id"]


def test_retain_non_conflicting_facts_both_stored(
    memory: MilaMemMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 2


def test_retain_days_window_excludes_old_conflict(tmp_path: Path) -> None:
    """A conflict outside a narrow day window isn't caught at retain-time."""
    memory = _semantic_memory(
        tmp_path,
        name="test-days-window",
        temporal_metadata=TemporalMetadataConfig(enabled=True),
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"days": 7},
        ),
    )
    memory.retain("Alice is in Paris.", {"timestamp": "2020-01-01"})
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 2
    assert not any(r["status"] == "superseded" for r in canonical)


def test_retain_days_window_also_bounds_exact_deduplication(tmp_path: Path) -> None:
    memory = _semantic_memory(
        tmp_path,
        name="test-days-window-exact",
        temporal_metadata=TemporalMetadataConfig(enabled=True),
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"days": 7},
        ),
    )
    memory.retain("Alice is in Paris.", {"timestamp": "2020-01-01"})

    result = memory.retain("Alice is in Paris.")

    assert result == "Successfully saved to memory."
    assert len([r for r in _canonical_records(memory) if r["status"] == "active"]) == 2


def test_retain_days_window_still_catches_recent_conflict(tmp_path: Path) -> None:
    """A conflict inside the days window is still caught."""
    memory = _semantic_memory(
        tmp_path,
        name="test-days-window-hit",
        temporal_metadata=TemporalMetadataConfig(enabled=True),
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"days": 7},
        ),
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0]["content"]


def test_retain_turns_window_excludes_old_conflict(tmp_path: Path) -> None:
    """A conflict pushed outside the N-turn window by intervening retains isn't caught."""
    memory = _semantic_memory(
        tmp_path,
        name="test-turns-window",
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"turns": 1},
        ),
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 3
    assert not any(r["status"] == "superseded" for r in canonical)


def test_retain_turns_window_still_catches_recent_conflict(tmp_path: Path) -> None:
    """A conflicting record within the N-turn window is still caught."""
    memory = _semantic_memory(
        tmp_path,
        name="test-turns-window-hit",
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"turns": 1},
        ),
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0]["content"]


def test_retain_day_window_ignored_when_temporal_metadata_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The day window is a no-op unless temporal metadata is enabled."""
    memory = _semantic_memory(
        tmp_path,
        name="test-days-noop",
        dedup_strategy=DedupStrategyConfig(
            enabled=True,
            kind=DedupStrategyKind.SEMANTIC,
            window={"days": 0.0001},
        ),
    )

    # With 3 unrelated windowed candidates, only the top-ranked one gets
    # judged — pin similarity by subject so "Alice is in London." reliably
    # ranks "Alice is in Paris." above the unrelated Bob/Carol records,
    # rather than leaving it to the generic hash-based fake's arbitrary
    # per-string similarity.
    def _subject_pinned_embed(model: str, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Alice" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(recall_strategies_module, "embed_texts", _subject_pinned_embed)

    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")
    memory.retain("Carol is in Cairo.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 3
    assert len(superseded) == 1
    assert any("London" in r["content"] for r in active)


def test_recall_excludes_superseded_canonical_records(
    tmp_path: Path,
) -> None:
    memory = _semantic_memory(tmp_path)
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    results = memory.recall("Alice")
    contents = [r for r in results if "Alice" in r]
    assert any("London" in c for c in contents)
    assert not any("Paris" in c for c in contents)


def test_semantic_dedup_default_skips_exact_duplicate(tmp_path: Path) -> None:
    """Canonical's default dedup_strategy is semantic (unlike
    basic_complete_memory_system's bm25 default), per Chirag's note."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-semantic-default",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
        ),
        storage_dir=tmp_path,
    )
    assert memory.config.dedup_strategy.kind is DedupStrategyKind.SEMANTIC

    memory.retain("Alice lives in Paris.")
    result = memory.retain("Alice lives in Paris.")

    assert result == "Memory already exists; skipping duplicate canonical projection."


def test_semantic_dedup_default_resolves_conflict(tmp_path: Path) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-semantic-default",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    assert len(active) == 1
    assert "London" in active[0]["content"]


def test_judge_exception_fails_open_leaves_new_record_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed judge call must not block retain(); the record stays active."""
    memory = _semantic_memory(tmp_path)

    def _boom(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        raise RuntimeError("llm boom")

    monkeypatch.setattr(consolidation_module, "judge_cluster", _boom)

    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")

    canonical = _canonical_records(memory)
    assert all(r["status"] == "active" for r in canonical)


def test_disabled_dedup_keeps_exact_duplicates_and_conflicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-disabled-dedup",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(enabled=False),
        ),
        storage_dir=tmp_path,
    )

    def _boom(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        raise AssertionError(
            "judge_cluster should not be called when dedup is disabled"
        )

    monkeypatch.setattr(consolidation_module, "judge_cluster", _boom)

    memory.retain("Alice lives in Boston.")
    memory.retain("Alice lives in Boston.")
    memory.retain("Alice moved to Chicago.")

    canonical = _canonical_records(memory)
    assert len(canonical) == 3
    assert all(r["status"] == "active" for r in canonical)


def test_semantic_floor_trigger_catches_paraphrased_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The similarity-floor trigger catches conflicts without a literal
    'X is Y' match, provided embedding similarity clears the floor."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-semantic-floor-hit",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        ),
        storage_dir=tmp_path,
    )

    def _controlled_embed(model: str, texts: list[str]) -> list[list[float]]:
        # Near-parallel vectors (cosine ~0.99); no literal " is " marker,
        # so only the floor trigger can fire.
        return [[0.9, 0.1] if "Chicago" in text else [1.0, 0.0] for text in texts]

    monkeypatch.setattr(recall_strategies_module, "embed_texts", _controlled_embed)
    monkeypatch.setattr(
        consolidation_module,
        "judge_cluster",
        lambda model, records: ConsolidationVerdict(
            relationship="conflict",
            canonical_id=max(records, key=lambda r: r.sequence).id,
        ),
    )

    memory.retain("Alice lives in Boston.")
    memory.retain("Alice moved to Chicago.")

    canonical = _canonical_records(memory)
    active = [r for r in canonical if r["status"] == "active"]
    superseded = [r for r in canonical if r["status"] == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "Chicago" in active[0]["content"]
    assert superseded[0]["consolidation_reason"] == "conflict_resolved"


def test_semantic_floor_below_threshold_skips_judge_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Below the similarity floor, retain-time must not call the judge at
    all."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-semantic-floor-miss",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        ),
        storage_dir=tmp_path,
    )

    def _orthogonal_embed(model: str, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Boston" in text else [0.0, 1.0] for text in texts]

    monkeypatch.setattr(recall_strategies_module, "embed_texts", _orthogonal_embed)

    def _boom(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        raise AssertionError(
            "judge_cluster should not be called below the similarity floor"
        )

    monkeypatch.setattr(consolidation_module, "judge_cluster", _boom)

    memory.retain("Alice lives in Boston.")
    memory.retain("The weather is nice today.")

    canonical = _canonical_records(memory)
    assert all(r["status"] == "active" for r in canonical)


def test_candidate_records_matches_by_id_not_content(tmp_path: Path) -> None:
    """Candidates must be matched back by id, not content, so duplicate-
    content records don't silently break the candidate_k bound."""
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="test-id-matching",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(
                enabled=False, kind=DedupStrategyKind.BM25, candidate_k=1
            ),
        ),
        storage_dir=tmp_path,
    )
    # Bypass exact-duplicate detection so both identical records land in canonical.
    memory._is_duplicate = lambda content, active_records: False
    memory.retain("Ping.")
    memory.retain("Ping.")
    memory.retain("Alice is in Paris.")

    canonical_store = memory._load_canonical_store()
    result = memory._candidate_records(
        "Ping.", canonical_store.records, now=datetime.now(timezone.utc)
    )
    assert len(result.records) == 1


def test_retain_result_reports_superseded_ids_and_reason_on_conflict(
    tmp_path: Path,
) -> None:
    """A retain-time conflict resolution must be visible in retain()'s result payload."""
    memory = _semantic_memory(tmp_path)
    memory._retain_impl("Alice is in Paris.", {})
    superseded_before = memory._load_canonical_store().records[0]

    outcome = memory._retain_impl("Alice is in London.", {})

    assert outcome.result["superseded_ids"] == [superseded_before.id]
    assert outcome.result["consolidation_reason"] == "conflict_resolved"


def test_retain_result_reports_empty_superseded_ids_when_nothing_triggers(
    memory: MilaMemMemorySystem,
) -> None:
    outcome = memory._retain_impl("Bob is in Berlin.", {})

    assert outcome.result["superseded_ids"] == []
    assert outcome.result["consolidation_reason"] is None
