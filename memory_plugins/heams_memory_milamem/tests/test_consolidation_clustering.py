"""Unit tests for consolidation.py: clustering, chunking, and judge_cluster's
any_llm integration. Behavioral use via consolidate() is covered in
test_milamem_consolidate.py."""

import hashlib
import json

import pytest

from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.consolidation import (
    chunk_cluster,
    cluster_by_similarity,
    judge_cluster,
)
from heams_memory_milamem.design.storage_type import (
    CanonicalMemoryProvenance,
    CanonicalMemoryRecord,
)
from heams_memory_milamem.recall_strategies import EmbeddingCache


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Fake embedding: texts sharing a prefix hash identically, others hash independently."""
    vectors: list[list[float]] = []
    for text in texts:
        key = text.split("|", 1)[0]
        digest = hashlib.md5(f"{model}:{key}".encode("utf-8")).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


def _record(id_: str, content: str, sequence: int) -> CanonicalMemoryRecord:
    return CanonicalMemoryRecord(
        id=id_,
        raw_source_id=f"raw-{id_}",
        content=content,
        sequence=sequence,
        provenance=CanonicalMemoryProvenance(raw_record_id=f"raw-{id_}"),
    )


def _entry_key_fn(record: CanonicalMemoryRecord) -> str:
    return record.id


def test_cluster_by_similarity_groups_records_above_threshold() -> None:
    records = [
        _record("a", "cluster1|Alice lives in Paris.", 1),
        _record("b", "cluster1|Alice resides in Paris.", 2),
        _record("c", "cluster2|Bob lives in Berlin.", 3),
    ]

    clusters = cluster_by_similarity(
        records,
        embed_model="fake",
        similarity_threshold=0.99,
        embedding_cache=EmbeddingCache({}),
        entry_key_fn=_entry_key_fn,
    )

    cluster_ids = sorted(sorted(r.id for r in cluster) for cluster in clusters)
    assert cluster_ids == [["a", "b"], ["c"]]


def test_cluster_by_similarity_keeps_dissimilar_records_as_singletons() -> None:
    records = [
        _record("a", "cluster1|Alice lives in Paris.", 1),
        _record("b", "cluster2|Bob lives in Berlin.", 2),
        _record("c", "cluster3|Carol lives in Cairo.", 3),
    ]

    clusters = cluster_by_similarity(
        records,
        embed_model="fake",
        similarity_threshold=0.99,
        embedding_cache=EmbeddingCache({}),
        entry_key_fn=_entry_key_fn,
    )

    assert len(clusters) == 3
    assert all(len(cluster) == 1 for cluster in clusters)


def test_cluster_by_similarity_empty_records_returns_empty() -> None:
    assert (
        cluster_by_similarity(
            [],
            embed_model="fake",
            similarity_threshold=0.99,
            embedding_cache=EmbeddingCache({}),
            entry_key_fn=_entry_key_fn,
        )
        == []
    )


def test_cluster_by_similarity_reuses_cached_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached embeddings shouldn't trigger a fresh embed_texts call."""
    cache = EmbeddingCache({})
    records = [
        _record("a", "cluster1|Alice lives in Paris.", 1),
        _record("b", "cluster1|Alice resides in Paris.", 2),
    ]
    cache.set("a", _fake_embed_texts("fake", ["cluster1|Alice lives in Paris."])[0])

    calls: list[list[str]] = []
    original = recall_strategies_module.embed_texts

    def _tracking_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return original(model, texts)

    monkeypatch.setattr(recall_strategies_module, "embed_texts", _tracking_embed_texts)

    cluster_by_similarity(
        records,
        embed_model="fake",
        similarity_threshold=0.99,
        embedding_cache=cache,
        entry_key_fn=_entry_key_fn,
    )

    assert calls == [["cluster1|Alice resides in Paris."]]


def test_cluster_by_similarity_pending_ids_skips_old_vs_old_pairs() -> None:
    """Same-embedding records stay separate singletons when neither id
    is in `pending_ids` — old-vs-old pairs are never re-compared."""
    records = [
        _record("a", "cluster1|Alice lives in Paris.", 1),
        _record("b", "cluster1|Alice resides in Paris.", 2),
    ]

    clusters = cluster_by_similarity(
        records,
        embed_model="fake",
        similarity_threshold=0.99,
        embedding_cache=EmbeddingCache({}),
        entry_key_fn=_entry_key_fn,
        pending_ids=set(),
    )

    cluster_ids = sorted(sorted(r.id for r in cluster) for cluster in clusters)
    assert cluster_ids == [["a"], ["b"]]


def test_cluster_by_similarity_pending_ids_still_clusters_pending_vs_old() -> None:
    """A pair still clusters when at least one side's id is in
    `pending_ids`, even though the other side is 'old'."""
    records = [
        _record("a", "cluster1|Alice lives in Paris.", 1),
        _record("b", "cluster1|Alice resides in Paris.", 2),
    ]

    clusters = cluster_by_similarity(
        records,
        embed_model="fake",
        similarity_threshold=0.99,
        embedding_cache=EmbeddingCache({}),
        entry_key_fn=_entry_key_fn,
        pending_ids={"b"},
    )

    cluster_ids = sorted(sorted(r.id for r in cluster) for cluster in clusters)
    assert cluster_ids == [["a", "b"]]


def test_chunk_cluster_splits_oversized_cluster_in_sequence_order() -> None:
    cluster = [
        _record("c", "content c", 3),
        _record("a", "content a", 1),
        _record("b", "content b", 2),
        _record("d", "content d", 4),
        _record("e", "content e", 5),
    ]

    chunks = chunk_cluster(cluster, max_size=2)

    assert [[r.id for r in chunk] for chunk in chunks] == [
        ["a", "b"],
        ["c", "d"],
        ["e"],
    ]


def test_chunk_cluster_under_max_size_returns_single_chunk() -> None:
    cluster = [_record("a", "content a", 1), _record("b", "content b", 2)]

    chunks = chunk_cluster(cluster, max_size=8)

    assert len(chunks) == 1
    assert [r.id for r in chunks[0]] == ["a", "b"]


def test_judge_cluster_calls_any_llm_with_allow_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: judge_cluster must pass allow_running_loop=True to
    any_llm, since it's called from a running event loop."""
    import any_llm

    captured_kwargs: dict = {}

    class _FakeMessage:
        content = json.dumps({"relationship": "duplicate", "canonical_id": "b"})

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    def _fake_completion(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeResponse()

    monkeypatch.setattr(any_llm, "completion", _fake_completion)

    records = [
        _record("a", "Alice lives in Paris.", 1),
        _record("b", "Alice lives in Lyon.", 2),
    ]
    verdict = judge_cluster("fake-model", records)

    assert captured_kwargs.get("allow_running_loop") is True
    assert verdict.relationship == "duplicate"
    assert verdict.canonical_id == "b"
