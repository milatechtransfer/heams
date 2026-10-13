from __future__ import annotations

import hashlib
import json

import pytest

from heams.memory_systems import create_memory
from heams.memory_systems.api import MemoryEntry
from heams_memory_milamem import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
)
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.recall_strategies import (
    RecallStrategyConfig,
    RecallStrategyKind,
)


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for `any_llm.embedding` so tests never hit a real API."""

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


def _fake_cross_encoder_scores(model: str, query: str, texts: list[str]) -> list[float]:
    """Deterministic stand-in for the cross-encoder so tests never load a real model."""

    query_tokens = set(query.lower().split())
    return [float(len(query_tokens & set(text.lower().split()))) for text in texts]


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(
        recall_strategies_module, "cross_encoder_scores", _fake_cross_encoder_scores
    )


def test_bm25_strategy_matches_default_behavior(tmp_path) -> None:
    memory = create_memory(
        BasicCompleteMemorySystemConfig(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.BM25)
        ),
        storage_dir=tmp_path,
    )

    memory.retain("The user lives in Montreal.")
    memory.retain("The user prefers dark mode.")
    memory.retain("The weather today is sunny.")

    results = memory.recall("Where does the user live?", top_k=1)

    assert results == ["The user lives in Montreal."]


def test_bm25_default_never_creates_embedding_cache_file(tmp_path) -> None:
    memory = create_memory(
        BasicCompleteMemorySystemConfig(),
        storage_dir=tmp_path,
    )

    memory.retain("The user lives in Montreal.")
    memory.recall("Where does the user live?", top_k=1)

    assert memory.config.embedding_cache_path is None
    assert not (tmp_path / "memory.embeddings.json").exists()


def test_runtime_storage_sets_file_and_rederives_embedding_cache(tmp_path) -> None:
    config = BasicCompleteMemorySystemConfig(
        file_path="static-memory.json",
        embedding_cache_path="static-embeddings.json",
        recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC),
    )

    resolved = config.with_runtime_storage(tmp_path)

    assert resolved.file_path == str(tmp_path / "memory.json")
    assert resolved.embedding_cache_path == str(tmp_path / "memory.embeddings.json")
    assert config.file_path == "static-memory.json"
    assert config.embedding_cache_path == "static-embeddings.json"
    memory = BasicCompleteMemorySystem(config=config, storage_dir=tmp_path)
    assert memory.config.embedding_cache_path == str(
        tmp_path / "memory.embeddings.json"
    )


@pytest.mark.parametrize(
    "kind", [RecallStrategyKind.SEMANTIC, RecallStrategyKind.HYBRID]
)
def test_semantic_and_hybrid_strategies_run_and_cache_embeddings(
    tmp_path, kind
) -> None:
    memory = create_memory(
        BasicCompleteMemorySystemConfig(
            recall_strategy=RecallStrategyConfig(kind=kind)
        ),
        storage_dir=tmp_path,
    )

    memory.retain("The user lives in Montreal.")
    memory.retain("The user prefers dark mode.")
    memory.retain("The weather today is sunny.")

    results = memory.recall("Where does the user live?", top_k=2)

    assert len(results) == 2
    assert all(isinstance(item, str) for item in results)

    cache_path = memory.config.embedding_cache_path
    assert cache_path is not None
    with open(cache_path, "r", encoding="utf-8") as handle:
        cache = json.load(handle)
    # 3 retained entries + 1 query embedding cached under its own content key.
    assert len(cache) >= 3


def test_rerank_strategy_runs_and_returns_top_k(tmp_path) -> None:
    memory = create_memory(
        BasicCompleteMemorySystemConfig(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.RERANK)
        ),
        storage_dir=tmp_path,
    )

    memory.retain("The user lives in Montreal.")
    memory.retain("The user prefers dark mode.")
    memory.retain("The weather today is sunny.")

    results = memory.recall("Where does the user live?", top_k=1)

    assert results == ["The user lives in Montreal."]


def test_rerank_candidate_pool_keeps_same_content_different_metadata() -> None:
    """Regression test: the candidate pool used to dedup by content only, so
    two distinct entries sharing text but differing in metadata collapsed
    into one candidate and the second was silently dropped even when top_k
    had room for both.

    Exercised at the `RerankRecallStrategy` level (not via
    `BasicCompleteMemorySystem.retain`, which now deliberately dedupes
    identical content regardless of metadata at write time) — the strategy
    itself must stay correct for any caller that hands it entries sharing
    content but not metadata (e.g. a canonical-projection backend).
    """
    entries = [
        MemoryEntry(
            content="The user lives in Montreal.", metadata={"source": "onboarding"}
        ),
        MemoryEntry(content="The user lives in Montreal.", metadata={"source": "chat"}),
    ]
    strategy = recall_strategies_module.build_recall_strategy(
        RecallStrategyConfig(kind=RecallStrategyKind.RERANK),
        embedding_cache=recall_strategies_module.EmbeddingCache(),
        entry_key_fn=lambda entry: entry.content,
    )

    results = strategy.rank(entries, "Where does the user live?", 2)

    assert len(results) == 2


def test_embedding_cache_path_defaults_to_sibling_of_file_path(tmp_path) -> None:
    storage_dir = tmp_path / "nested"
    memory = create_memory(
        BasicCompleteMemorySystemConfig(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC)
        ),
        storage_dir=storage_dir,
    )

    assert memory.config.embedding_cache_path == str(
        storage_dir / "memory.embeddings.json"
    )


def test_embedding_cache_avoids_recomputing_unchanged_content(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call_count = {"n": 0}

    def counting_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
        call_count["n"] += 1
        return _fake_embed_texts(model, texts)

    monkeypatch.setattr(recall_strategies_module, "embed_texts", counting_embed_texts)

    memory = create_memory(
        BasicCompleteMemorySystemConfig(
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC)
        ),
        storage_dir=tmp_path,
    )
    memory.retain("The user lives in Montreal.")

    memory.recall("Where does the user live?", top_k=1)
    calls_after_first_recall = call_count["n"]

    memory.recall("Where does the user live?", top_k=1)
    calls_after_second_recall = call_count["n"]

    # The second recall re-embeds the (repeated) query but should not
    # re-embed the unchanged stored entry.
    assert calls_after_second_recall > calls_after_first_recall
    assert calls_after_second_recall < calls_after_first_recall * 2
