from __future__ import annotations

import pytest
from heams.memory_systems.api import MemoryEntry
from pydantic import ValidationError
from rank_bm25 import BM25Okapi

from heams_memory_milamem import recall_strategies
from heams_memory_milamem.recall_strategies import (
    EmbeddingCache,
    FusionKind,
    HybridRecallStrategy,
    RecallStrategyConfig,
    RecallStrategyKind,
    SemanticRecallStrategy,
    build_recall_strategy,
    filter_count,
    upper_bound_bm25,
)
from heams_memory_milamem.retrieval import tokenize_text

# Query is [1, 0]; each text's cosine to it is fixed by its vector.
_VECTORS: dict[str, list[float]] = {
    "query": [1.0, 0.0],
    "alice lives in paris": [1.0, 0.0],
    "alice likes paris food": [0.8, 0.6],
    "bob owns a red car": [0.0, 1.0],
    "unrelated weather note": [-1.0, 0.0],
}


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake(model: str, texts: list[str]) -> list[list[float]]:
        return [_VECTORS.get(text, _VECTORS["query"]) for text in texts]

    monkeypatch.setattr(recall_strategies, "embed_texts", _fake)


def _strategy(**kwargs) -> HybridRecallStrategy:
    semantic = SemanticRecallStrategy(
        embed_model="fake",
        embedding_cache=EmbeddingCache(),
        entry_key_fn=lambda entry: entry.content,
    )
    return HybridRecallStrategy(
        semantic, weight=kwargs.pop("weight", 0.5), fusion=FusionKind.ABSOLUTE, **kwargs
    )


def _entries(*contents: str) -> list[MemoryEntry]:
    return [MemoryEntry(content=content) for content in contents]


# --- upper_bound_bm25 ---


def test_upper_bound_bm25_is_in_unit_interval_on_real_bm25() -> None:
    corpus = [tokenize_text(t) for t in ("alice lives in paris", "bob owns a car", "paris paris paris")]
    bm25 = BM25Okapi(corpus)
    query = tokenize_text("paris alice")

    scores = upper_bound_bm25(list(bm25.get_scores(query)), query, bm25.idf, bm25.k1)

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores[0] > scores[1]


def test_upper_bound_bm25_duplicate_tokens_scale_score_and_bound_together() -> None:
    corpus = [tokenize_text(t) for t in ("alice lives in paris", "bob owns a car")]
    bm25 = BM25Okapi(corpus)

    single = upper_bound_bm25(list(bm25.get_scores(["alice"])), ["alice"], bm25.idf, bm25.k1)
    doubled = upper_bound_bm25(
        list(bm25.get_scores(["alice", "alice"])), ["alice", "alice"], bm25.idf, bm25.k1
    )

    assert doubled == pytest.approx(single)


def test_upper_bound_bm25_unknown_tokens_give_zeros() -> None:
    assert upper_bound_bm25([0.0, 0.0], ["zzz"], {"alice": 1.0}, 1.5) == [0.0, 0.0]


# --- filter_count ---


@pytest.mark.parametrize(
    ("scores", "kwargs", "expected"),
    [
        ([0.9, 0.5, 0.1], dict(ratio=0.0, min_keep=1, min_score=0.0, top_k=20), 3),
        ([0.9, 0.9, 0.5], dict(ratio=1.0, min_keep=1, min_score=0.0, top_k=20), 2),
        ([0.9, 0.5, 0.1], dict(ratio=0.5, min_keep=1, min_score=0.0, top_k=20), 2),
        ([0.9, 0.1, 0.1], dict(ratio=0.5, min_keep=2, min_score=0.0, top_k=20), 2),
        ([0.3, 0.2], dict(ratio=0.0, min_keep=2, min_score=0.25, top_k=20), 1),
        ([0.2, 0.1], dict(ratio=0.0, min_keep=2, min_score=0.5, top_k=20), 0),
        ([0.9, 0.8, 0.7], dict(ratio=0.0, min_keep=1, min_score=0.0, top_k=2), 2),
        ([], dict(ratio=0.5, min_keep=1, min_score=0.0, top_k=20), 0),
    ],
)
def test_filter_count(scores: list[float], kwargs: dict, expected: int) -> None:
    assert filter_count(scores, **kwargs) == expected


# --- HybridRecallStrategy (absolute) ---


def test_absolute_scores_are_in_unit_interval_and_descending() -> None:
    entries = _entries(
        "alice lives in paris", "alice likes paris food", "bob owns a red car", "unrelated weather note"
    )

    ranked = _strategy().rank(entries, "query alice paris", top_k=20)

    scores = [entry.score for entry in ranked]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    assert ranked[0].content == "alice lives in paris"


def test_relative_filter_trims_low_scorers() -> None:
    entries = _entries("alice lives in paris", "bob owns a red car", "unrelated weather note")

    unfiltered = _strategy().rank(entries, "query alice paris", top_k=20)
    filtered = _strategy(filter_ratio=0.5).rank(entries, "query alice paris", top_k=20)

    assert len(unfiltered) == 3
    assert [entry.content for entry in filtered] == ["alice lives in paris"]


def test_query_without_tokens_ranks_by_cosine_only() -> None:
    entries = _entries("bob owns a red car", "alice lives in paris")

    ranked = _strategy(weight=0.0).rank(entries, "?!", top_k=20)

    assert [entry.content for entry in ranked] == ["alice lives in paris", "bob owns a red car"]
    assert ranked[0].score == pytest.approx(1.0)


def test_ties_are_broken_newest_first() -> None:
    older = MemoryEntry(
        content="alice lives in paris", metadata={"temporal_received_at": "2024-01-01T00:00:00Z"}
    )
    newer = MemoryEntry(
        content="alice lives in paris", metadata={"temporal_received_at": "2025-01-01T00:00:00Z"}
    )

    ranked = _strategy().rank([older, newer], "query alice paris", top_k=20)

    assert [entry.metadata["temporal_received_at"] for entry in ranked] == [
        "2025-01-01T00:00:00Z",
        "2024-01-01T00:00:00Z",
    ]


def test_build_recall_strategy_passes_filter_through() -> None:
    config = RecallStrategyConfig(
        kind=RecallStrategyKind.HYBRID,
        embed_model="fake",
        fusion=FusionKind.ABSOLUTE,
        filter_ratio=0.5,
    )
    strategy = build_recall_strategy(
        config, embedding_cache=EmbeddingCache(), entry_key_fn=lambda entry: entry.content
    )
    entries = _entries("alice lives in paris", "bob owns a red car", "unrelated weather note")

    assert len(strategy.rank(entries, "query alice paris", top_k=20)) == 1


# --- config validation ---


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(kind=RecallStrategyKind.BM25, fusion=FusionKind.ABSOLUTE),
        dict(kind=RecallStrategyKind.SEMANTIC, filter_ratio=0.5),
        dict(kind=RecallStrategyKind.HYBRID, filter_ratio=0.5),
        dict(kind=RecallStrategyKind.HYBRID, filter_min_score=0.1),
        dict(kind=RecallStrategyKind.RERANK, filter_min_keep=3),
    ],
)
def test_config_rejects_filter_outside_absolute_hybrid(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        RecallStrategyConfig(**kwargs)


def test_config_defaults_keep_minmax_and_no_filter() -> None:
    config = RecallStrategyConfig(kind=RecallStrategyKind.HYBRID)

    assert config.fusion is FusionKind.MINMAX
    assert (config.filter_ratio, config.filter_min_keep, config.filter_min_score) == (0.0, 1, 0.0)
