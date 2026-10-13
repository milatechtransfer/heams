from __future__ import annotations

import hashlib

import pytest

from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
    build_dedup_strategy,
)
from heams_memory_milamem.recall_strategies import EmbeddingCache
from heams.memory_systems.api import MemoryEntry


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


def test_bm25_dedup_strategy_finds_candidate_by_keyword_overlap() -> None:
    strategy = build_dedup_strategy(
        DedupStrategyConfig(kind=DedupStrategyKind.BM25),
        embedding_cache=EmbeddingCache(),
        entry_key_fn=lambda entry: entry.content,
    )
    entries = [
        MemoryEntry(content="Employer is State University"),
        MemoryEntry(content="Favorite color is blue"),
    ]

    candidates = strategy.candidates(entries, "Employer is Mila", 1)

    assert len(candidates) == 1
    assert candidates[0].content == "Employer is State University"


def test_semantic_dedup_strategy_finds_candidate_and_caches_embeddings() -> None:
    cache = EmbeddingCache()
    strategy = build_dedup_strategy(
        DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        embedding_cache=cache,
        entry_key_fn=lambda entry: entry.content,
    )
    entries = [
        MemoryEntry(content="Employer is State University"),
        MemoryEntry(content="Favorite color is blue"),
    ]

    candidates = strategy.candidates(entries, "Employer is Mila", 1)

    # Fake hash-based embeddings don't preserve real semantic similarity, so
    # this only checks shape/caching; ranking correctness for real
    # embeddings is covered by the real-data verification, not unit tests.
    assert len(candidates) == 1
    assert candidates[0].content in {e.content for e in entries}
    # entries + query content all cached under model+content keys
    assert len(cache.to_dict()) >= 2


def test_semantic_kind_is_enabled_by_default() -> None:
    config = DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC)

    assert config.enabled is True


def test_semantic_dedup_strategy_reuses_cached_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}

    def counting_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
        call_count["n"] += 1
        return _fake_embed_texts(model, texts)

    monkeypatch.setattr(recall_strategies_module, "embed_texts", counting_embed_texts)

    cache = EmbeddingCache()
    strategy = build_dedup_strategy(
        DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        embedding_cache=cache,
        entry_key_fn=lambda entry: entry.content,
    )
    entries = [MemoryEntry(content="Employer is State University")]

    strategy.candidates(entries, "Employer is Mila", 1)
    calls_after_first = call_count["n"]
    strategy.candidates(entries, "Employer is Anthropic", 1)
    calls_after_second = call_count["n"]

    # second call re-embeds only the new query content, not the unchanged entry
    assert calls_after_second > calls_after_first
    assert calls_after_second < calls_after_first * 2


def test_build_dedup_strategy_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        build_dedup_strategy(
            DedupStrategyConfig.model_construct(kind="unknown"),
            embedding_cache=EmbeddingCache(),
            entry_key_fn=lambda entry: entry.content,
        )
