from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from heams_memory_milamem import consolidation as consolidation_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
)
from heams_memory_milamem.consolidation import ConsolidationVerdict
from heams_memory_milamem.dedup_strategies import DedupStrategyConfig, DedupStrategyKind
from heams_memory_milamem.recall_strategies import (
    RecallStrategyConfig,
    RecallStrategyKind,
)
from heams_memory_milamem import recall_strategies as recall_strategies_module


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


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the LLM judge to always return 'unrelated' so retains here stay hermetic."""

    def _always_unrelated(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        return ConsolidationVerdict(relationship="unrelated", canonical_id=None)

    monkeypatch.setattr(consolidation_module, "judge_cluster", _always_unrelated)


def _memory(tmp_path: Path, **config_kwargs) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            name="canonical",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            **config_kwargs,
        ),
        storage_dir=tmp_path,
    )


def test_recall_with_no_entries_returns_empty(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    assert memory.recall("anything") == []


def test_bm25_recall_ranks_canonical_records(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.retain("Employer is State University")
    memory.retain("Favorite color is blue")

    results = memory.recall("employer", top_k=1)

    assert results == ["Employer is State University"]


def test_bm25_recall_with_disabled_dedup_never_creates_embedding_cache_file(
    tmp_path: Path,
) -> None:
    memory = _memory(
        tmp_path,
        dedup_strategy=DedupStrategyConfig(
            enabled=False,
            kind=DedupStrategyKind.BM25,
        ),
    )
    memory.retain("The user lives in Montreal.")
    memory.recall("Where does the user live?", top_k=1)

    assert memory.config.embedding_cache_path is not None
    assert not (tmp_path / "canonical.embeddings.json").exists()


def test_default_config_persists_embedding_cache_only_after_new_vectors(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)
    assert memory.config.dedup_strategy.kind is DedupStrategyKind.SEMANTIC
    assert memory.config.embedding_cache_path is not None

    memory.retain("The user lives in Montreal.")
    assert not Path(memory.config.embedding_cache_path).exists()

    memory.retain("The user's dog is named Whiskers.")
    assert Path(memory.config.embedding_cache_path).exists()


def test_empty_query_falls_back_to_chronological_tail(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.retain("Fact one")
    memory.retain("Fact two")
    memory.retain("Fact three")

    results = memory.recall("", top_k=2)

    assert results == ["Fact two", "Fact three"]


@pytest.mark.parametrize(
    "kind",
    [RecallStrategyKind.SEMANTIC, RecallStrategyKind.HYBRID, RecallStrategyKind.RERANK],
)
def test_non_bm25_strategies_rank_canonical_records_and_cache_embeddings(
    tmp_path: Path, kind: RecallStrategyKind
) -> None:
    memory = _memory(tmp_path, recall_strategy=RecallStrategyConfig(kind=kind))
    memory.retain("The user lives in Montreal.")
    memory.retain("The user's dog is named Whiskers.")
    memory.retain("The weather is cold in winter.")

    results = memory.recall("Where does the user live?", top_k=2)

    assert len(results) == 2
    assert any("Montreal" in r for r in results)

    # hybrid/rerank both draw on the semantic branch internally, so all
    # three non-bm25 kinds populate the embedding cache the same way.
    cache_path = memory.config.embedding_cache_path
    assert cache_path is not None
    with open(cache_path, "r", encoding="utf-8") as handle:
        cache = json.load(handle)
    assert len(cache) >= 3


def test_embedding_cache_path_defaults_to_sibling_of_canonical_store_path(
    tmp_path: Path,
) -> None:
    canonical_path = tmp_path / "nested" / "canonical.json"
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="canonical",
            raw_store_path=str(tmp_path / "nested" / "raw.json"),
            canonical_store_path=str(canonical_path),
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.SEMANTIC),
        ),
        storage_dir=canonical_path.parent,
    )

    assert memory.config.embedding_cache_path == str(
        tmp_path / "nested" / "canonical.embeddings.json"
    )


def test_recall_reads_canonical_store_not_raw_store(tmp_path: Path) -> None:
    """Regression guard: recall must rank CanonicalMemoryRecord entries, not
    the raw log — the two are written independently by retain()."""
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="canonical",
            raw_store_path=str(raw_path),
            canonical_store_path=str(canonical_path),
        ),
        storage_dir=tmp_path,
    )
    memory.retain("The user lives in Montreal.")

    # Corrupt the raw store on disk; recall must still work off canonical.
    raw_path.write_text("not valid json", encoding="utf-8")

    results = memory.recall("Where does the user live?", top_k=1)
    assert results == ["The user lives in Montreal."]
