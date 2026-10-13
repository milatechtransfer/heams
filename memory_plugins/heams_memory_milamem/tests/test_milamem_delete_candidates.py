from __future__ import annotations

from pathlib import Path

import pytest
from heams_memory_milamem import milamem as canonical_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
    CanonicalMemoryStoreState,
)
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)
from heams_memory_milamem.design.delete.models import DeleteQueryPlan
from heams_memory_milamem.design.delete.strategies import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
    DeleteStrategyConfig,
    ExactThenTopKLLMDeleteStrategy,
    LlmJudgeDeleteStrategy,
)
from heams_memory_milamem.design.storage_type import CanonicalMemoryRecord
from heams_memory_milamem.recall_strategies import (
    BaseRecallStrategy,
    BM25RecallStrategy,
    RecallStrategyConfig,
    RecallStrategyKind,
)
from heams.memory_systems.api import MemoryEntry


class RecordingRecallStrategy(BaseRecallStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[list[MemoryEntry], str, int]] = []

    def rank(
        self,
        entries: list[MemoryEntry],
        query: str,
        top_k: int,
    ) -> list[MemoryEntry]:
        self.calls.append((entries, query, top_k))
        ranked = list(reversed(entries))[:top_k]
        return [
            entry.model_copy(update={"score": float(index) + 0.25})
            for index, entry in enumerate(ranked)
        ]


class RecordingDeleteStrategy(BaseDeleteStrategy):
    def __init__(self) -> None:
        self.calls: list[tuple[DeleteQueryPlan, list[DeleteCandidate]]] = []

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        self.calls.append((query_plan, candidates))
        return DeleteDecision(
            selected_ids=[],
            reasoning="Candidate inspection only.",
            ambiguous=False,
        )


def _capture_candidates(
    memory: MilaMemMemorySystem,
    query: str,
) -> tuple[list[DeleteCandidate], RecordingDeleteStrategy]:
    strategy = RecordingDeleteStrategy()
    memory._delete_strategy = strategy
    memory._delete_lifecycle.strategy = strategy
    result = memory._delete_impl(query)
    candidates = strategy.calls[0][1] if strategy.calls else []
    assert result.result["candidate_count"] == len(candidates)
    return candidates, strategy


def _memory(
    tmp_path: Path,
    *,
    recall_kind: RecallStrategyKind = RecallStrategyKind.BM25,
    delete_strategy: DeleteStrategyConfig | None = None,
) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            name="canonical",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            recall_strategy=RecallStrategyConfig(kind=recall_kind),
            dedup_strategy=DedupStrategyConfig(
                enabled=False,
                kind=DedupStrategyKind.BM25,
            ),
            delete_strategy=delete_strategy
            or DeleteStrategyConfig(selection_mode="top_k_llm"),
        ),
        storage_dir=tmp_path,
    )


def _record(
    memory_id: str,
    content: str,
    sequence: int,
    *,
    status: str = "active",
) -> CanonicalMemoryRecord:
    raw_id = f"raw-{memory_id}"
    return CanonicalMemoryRecord(
        id=memory_id,
        raw_source_id=raw_id,
        content=content,
        metadata={"source": memory_id},
        sequence=sequence,
        provenance={"raw_record_id": raw_id},
        status=status,
    )


def test_delete_candidates_rank_only_active_records_and_preserve_ids_and_scores(
    tmp_path: Path,
) -> None:
    memory = _memory(
        tmp_path,
        delete_strategy=DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"candidate_k": 2},
        ),
    )
    memory._save_canonical_store(
        CanonicalMemoryStoreState(
            records=[
                _record("active-1", "First active fact", 1),
                _record("deleted-1", "Deleted fact", 2, status="deleted"),
                _record("active-2", "Second active fact", 3),
                _record("superseded-1", "Old fact", 4, status="superseded"),
            ]
        )
    )
    canonical_path = Path(memory.config.canonical_store_path)
    before = canonical_path.read_bytes()
    recall_strategy = RecordingRecallStrategy()
    memory._recall_strategy = recall_strategy
    memory._delete_retrieval_strategy = recall_strategy

    candidates, _ = _capture_candidates(memory, "active fact")

    assert [candidate.memory_id for candidate in candidates] == [
        "active-2",
        "active-1",
    ]
    assert [candidate.content for candidate in candidates] == [
        "Second active fact",
        "First active fact",
    ]
    assert [candidate.score for candidate in candidates] == [0.25, 1.25]
    ranked_entries, query, top_k = recall_strategy.calls[0]
    assert query == "active fact"
    assert top_k == 2
    assert [entry.content for entry in ranked_entries] == [
        "First active fact",
        "Second active fact",
    ]
    ranked_ids = {entry.metadata["_canonical_memory_id"] for entry in ranked_entries}
    assert "superseded-1" not in ranked_ids
    assert "deleted-1" not in ranked_ids
    assert canonical_path.read_bytes() == before
    assert Path(memory.config.raw_store_path).exists()


def test_delete_candidates_respect_candidate_k(tmp_path: Path) -> None:
    memory = _memory(
        tmp_path,
        delete_strategy=DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"candidate_k": 1},
        ),
    )
    memory._save_canonical_store(
        CanonicalMemoryStoreState(
            records=[
                _record("memory-1", "Fact one", 1),
                _record("memory-2", "Fact two", 2),
                _record("memory-3", "Fact three", 3),
            ]
        )
    )
    recall_strategy = RecordingRecallStrategy()
    memory._recall_strategy = recall_strategy
    memory._delete_retrieval_strategy = recall_strategy

    candidates, _ = _capture_candidates(memory, "fact")

    assert len(candidates) == 1
    assert recall_strategy.calls[0][2] == 1


def test_delete_candidates_return_empty_for_empty_canonical_store(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)
    recall_strategy = RecordingRecallStrategy()
    memory._recall_strategy = recall_strategy
    memory._delete_retrieval_strategy = recall_strategy

    candidates, strategy = _capture_candidates(memory, "anything")

    assert candidates == []
    assert strategy.calls == []
    assert recall_strategy.calls == []


def test_delete_candidates_save_cache_for_non_bm25_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path, recall_kind=RecallStrategyKind.SEMANTIC)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_record("memory-1", "Fact one", 1)])
    )
    recall_strategy = RecordingRecallStrategy()
    memory._recall_strategy = recall_strategy
    memory._delete_retrieval_strategy = recall_strategy
    save_calls: list[None] = []
    monkeypatch.setattr(
        memory, "_save_embedding_cache", lambda: save_calls.append(None)
    )

    _capture_candidates(memory, "fact")

    assert save_calls == [None]


def test_delete_retrieval_reuses_normal_recall_when_unconfigured(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)

    assert memory._delete_retrieval_strategy is memory._recall_strategy


def test_delete_retrieval_can_use_independent_configured_strategy(
    tmp_path: Path,
) -> None:
    memory = _memory(
        tmp_path,
        recall_kind=RecallStrategyKind.SEMANTIC,
        delete_strategy=DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"retrieval_strategy": {"kind": "bm25"}},
        ),
    )

    assert isinstance(memory._delete_retrieval_strategy, BM25RecallStrategy)
    assert memory._delete_retrieval_strategy is not memory._recall_strategy


def test_delete_only_semantic_retrieval_persists_embedding_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(
        tmp_path,
        recall_kind=RecallStrategyKind.BM25,
        delete_strategy=DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"retrieval_strategy": {"kind": "semantic"}},
        ),
    )
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_record("memory-1", "Fact one", 1)])
    )
    memory._delete_retrieval_strategy = RecordingRecallStrategy()
    save_calls: list[None] = []
    monkeypatch.setattr(
        memory, "_save_embedding_cache", lambda: save_calls.append(None)
    )

    _capture_candidates(memory, "fact")

    assert memory.config.embedding_cache_path is not None
    assert save_calls == [None]


def test_exact_success_does_not_save_non_bm25_embedding_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(
        tmp_path,
        recall_kind=RecallStrategyKind.SEMANTIC,
        delete_strategy=DeleteStrategyConfig(selection_mode="exact"),
    )
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_record("memory-1", "Fact one", 1)])
    )
    recall_strategy = RecordingRecallStrategy()
    memory._recall_strategy = recall_strategy
    memory._delete_retrieval_strategy = recall_strategy
    save_calls: list[None] = []
    monkeypatch.setattr(
        memory, "_save_embedding_cache", lambda: save_calls.append(None)
    )

    result = memory._delete_impl("Forget Fact one")

    assert result.result["deleted_count"] == 1
    assert recall_strategy.calls == []
    assert save_calls == []


def test_config_selects_and_constructs_requested_delete_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeleteStrategyConfig(
        top_k_llm={"candidate_k": 3},
    )
    received_configs: list[DeleteStrategyConfig] = []

    class RequestedDeleteStrategy(BaseDeleteStrategy):
        def select(
            self,
            query_plan: DeleteQueryPlan,
            candidates: list[DeleteCandidate],
        ) -> DeleteDecision:
            return DeleteDecision(
                selected_ids=[],
                reasoning="Test strategy.",
                ambiguous=False,
            )

    requested_strategy = RequestedDeleteStrategy()

    def fake_build(received_config: DeleteStrategyConfig) -> BaseDeleteStrategy:
        received_configs.append(received_config)
        return requested_strategy

    monkeypatch.setattr(canonical_module, "build_delete_strategy", fake_build)

    memory = _memory(tmp_path, delete_strategy=config)

    assert memory.config.delete_strategy is config
    assert memory._delete_strategy is requested_strategy
    assert memory._delete_lifecycle.strategy is requested_strategy
    assert memory._delete_lifecycle.query_planner is memory._delete_query_planner
    assert received_configs == [config]


def test_default_config_constructs_composed_delete_strategy(tmp_path: Path) -> None:
    memory = _memory(tmp_path, delete_strategy=DeleteStrategyConfig())

    assert memory.config.delete_strategy.selection_mode.value == "exact_then_top_k_llm"
    assert isinstance(memory._delete_strategy, ExactThenTopKLLMDeleteStrategy)
    assert isinstance(memory._delete_strategy.llm_strategy, LlmJudgeDeleteStrategy)
    assert memory._delete_lifecycle.strategy is memory._delete_strategy
    assert memory._delete_lifecycle.query_planner is memory._delete_query_planner
