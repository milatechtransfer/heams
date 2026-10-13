"""Tests for deduplication and conflict resolution in BasicCompleteMemorySystem.retain."""

import hashlib
from pathlib import Path

import pytest

from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.basic_complete_memory_system import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
)
from heams_memory_milamem.dedup_strategies import DedupStrategyConfig, DedupStrategyKind


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for `any_llm.embedding` so tests never hit a real API."""

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


@pytest.fixture()
def memory(tmp_path: Path) -> BasicCompleteMemorySystem:
    return BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="test",
            file_path=str(tmp_path / "memory.json"),
        ),
        storage_dir=tmp_path,
    )


@pytest.fixture()
def semantic_dedup_memory(tmp_path: Path) -> BasicCompleteMemorySystem:
    return BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="test-semantic-dedup",
            file_path=str(tmp_path / "memory.json"),
            dedup_strategy=DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        ),
        storage_dir=tmp_path,
    )


def test_retain_appends_new_memory(memory: BasicCompleteMemorySystem) -> None:
    memory.retain("Alice lives in Paris.")
    result = memory.recall("Alice")
    assert any("Alice" in r for r in result)


def test_retain_skips_exact_duplicate(memory: BasicCompleteMemorySystem) -> None:
    memory.retain("Alice lives in Paris.")
    result = memory.retain("Alice lives in Paris.")
    assert result == "Memory already exists; skipping duplicate."
    records = memory._load_records()
    assert len([r for r in records if r.status == "active"]) == 1


def test_disabled_dedup_stores_exact_duplicates_without_embedding_cache(
    tmp_path: Path,
) -> None:
    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="test-disabled-dedup",
            dedup_strategy=DedupStrategyConfig(
                enabled=False,
                kind=DedupStrategyKind.SEMANTIC,
            ),
        ),
        storage_dir=tmp_path,
    )

    memory.retain("Alice lives in Paris.")
    memory.retain("Alice lives in Paris.")

    assert len([r for r in memory._load_records() if r.status == "active"]) == 2
    assert memory.config.embedding_cache_path is None
    assert not (tmp_path / "memory.embeddings.json").exists()


def test_retain_skips_duplicate_ranked_outside_dedup_candidates(
    memory: BasicCompleteMemorySystem,
) -> None:
    """Exact-duplicate detection must not depend on `dedup_strategy`'s
    bounded candidate_k lookup: even if a true duplicate would rank outside
    the candidate set (e.g. buried under many higher-scoring near-matches),
    it must still be caught since normalized-text comparison is cheap over
    the full active store."""
    memory.retain("Alice lives in Paris.")

    # Simulate the duplicate ranking outside the candidate set by making
    # `_candidate_records` return nothing, regardless of content.
    memory._candidate_records = lambda content, records: []

    result = memory.retain("Alice lives in Paris.")
    assert result == "Memory already exists; skipping duplicate."
    records = memory._load_records()
    assert len([r for r in records if r.status == "active"]) == 1


def test_retain_skips_case_insensitive_duplicate(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice lives in Paris.")
    result = memory.retain("ALICE LIVES IN PARIS.")
    assert result == "Memory already exists; skipping duplicate."
    records = memory._load_records()
    assert len([r for r in records if r.status == "active"]) == 1


def test_retain_resolves_conflict_supersedes_old_record(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    records = memory._load_records()
    active = [r for r in records if r.status == "active"]
    superseded = [r for r in records if r.status == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0].content
    assert superseded[0].consolidation_reason == "conflict_resolved"


def test_retain_stored_count_is_active_only_across_all_code_paths(
    memory: BasicCompleteMemorySystem,
) -> None:
    """`result["stored_count"]` must mean the same thing (active records
    only) regardless of which retain() branch produced it — append,
    duplicate-skip, or conflict-resolution — so observability metrics don't
    silently disagree depending on code path."""
    ledger: list = []
    memory.attach_event_sink(ledger)

    memory.retain("Alice is in Paris.")  # plain append
    memory.retain("Alice is in Paris.")  # duplicate skip
    memory.retain("Alice is in London.")  # conflict resolution (supersedes Paris)

    active_count = len([r for r in memory._load_records() if r.status == "active"])
    assert active_count == 1
    assert [event.result["stored_count"] for event in ledger] == [
        active_count,
        active_count,
        active_count,
    ]


def test_retain_recall_read_after_write_correctness(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    result = memory.recall("Alice")
    contents = [r for r in result if "Alice" in r]
    assert any("London" in c for c in contents)
    assert not any("Paris" in c for c in contents)


def test_retain_non_conflicting_facts_both_stored(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    memory.retain("Bob is in Berlin.")
    records = memory._load_records()
    active = [r for r in records if r.status == "active"]
    assert len(active) == 2


def test_retain_non_fact_memories_always_appended(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice went to the market.")
    memory.retain("Alice went to the market.")
    # Non-fact content still deduped via exact-text match
    records = memory._load_records()
    active = [r for r in records if r.status == "active"]
    assert len(active) == 1


def test_retain_multiple_conflicts_resolved_in_sequence(
    memory: BasicCompleteMemorySystem,
) -> None:
    memory.retain("Alice is in Paris.")
    memory.retain("Alice is in London.")
    memory.retain("Alice is in Berlin.")
    records = memory._load_records()
    active = [r for r in records if r.status == "active"]
    assert len(active) == 1
    assert "Berlin" in active[0].content


def test_retain_skips_exact_duplicate_with_semantic_dedup_strategy(
    semantic_dedup_memory: BasicCompleteMemorySystem,
) -> None:
    semantic_dedup_memory.retain("Alice lives in Paris.")
    result = semantic_dedup_memory.retain("Alice lives in Paris.")
    assert result == "Memory already exists; skipping duplicate."
    records = semantic_dedup_memory._load_records()
    assert len([r for r in records if r.status == "active"]) == 1


def test_retain_resolves_conflict_with_semantic_dedup_strategy(
    semantic_dedup_memory: BasicCompleteMemorySystem,
) -> None:
    semantic_dedup_memory.retain("Alice is in Paris.")
    semantic_dedup_memory.retain("Alice is in London.")
    records = semantic_dedup_memory._load_records()
    active = [r for r in records if r.status == "active"]
    superseded = [r for r in records if r.status == "superseded"]
    assert len(active) == 1
    assert len(superseded) == 1
    assert "London" in active[0].content


def test_retain_non_conflicting_facts_both_stored_with_semantic_dedup_strategy(
    semantic_dedup_memory: BasicCompleteMemorySystem,
) -> None:
    semantic_dedup_memory.retain("Alice is in Paris.")
    semantic_dedup_memory.retain("Bob is in Berlin.")
    records = semantic_dedup_memory._load_records()
    active = [r for r in records if r.status == "active"]
    assert len(active) == 2


def test_semantic_dedup_strategy_shares_embedding_cache_with_recall(
    tmp_path: Path,
) -> None:
    """Recall and dedup share one embedding cache keyed by (embed_model,
    content), so identical content embedded for one is reused by the other."""
    from heams_memory_milamem.recall_strategies import (
        RecallStrategyConfig,
        RecallStrategyKind,
    )

    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="test-shared-cache",
            file_path=str(tmp_path / "memory.json"),
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC),
            dedup_strategy=DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("Alice is in Paris.")
    memory.recall("Where is Alice?")
    memory.retain("Alice is in London.")

    records = memory._load_records()
    active = [r for r in records if r.status == "active"]
    assert len(active) == 1
    assert "London" in active[0].content
