from __future__ import annotations

from typing import Literal

import pytest

from heams_memory_milamem.design.delete.config import (
    DeleteStrategyConfig,
    ExactSelectionConfig,
)
from heams_memory_milamem.design.delete.models import DeleteQueryPlan
from heams_memory_milamem.design.delete.strategies import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
    ExactDeleteStrategy,
    ExactThenTopKLLMDeleteStrategy,
    LlmJudgeDeleteStrategy,
    build_delete_strategy,
)


class RecordingFallbackStrategy(BaseDeleteStrategy):
    def __init__(self, decision: DeleteDecision):
        self.decision = decision
        self.calls: list[tuple[DeleteQueryPlan, list[DeleteCandidate]]] = []

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        self.calls.append((query_plan, candidates))
        return self.decision


def _plan(
    target_text: str,
    *,
    cardinality: Literal["one", "all"] = "one",
    search_query: str | None = None,
) -> DeleteQueryPlan:
    return DeleteQueryPlan(
        original_query=f"Forget {target_text}",
        target_text=target_text,
        search_query=search_query or target_text,
        cardinality=cardinality,
    )


def _candidate(memory_id: str, content: str, score: float = 0.0) -> DeleteCandidate:
    return DeleteCandidate(memory_id=memory_id, content=content, score=score)


def _composed(
    fallback: BaseDeleteStrategy,
    *,
    max_delete_count: int = 5,
    candidate_k: int = 5,
) -> ExactThenTopKLLMDeleteStrategy:
    return ExactThenTopKLLMDeleteStrategy(
        ExactDeleteStrategy(
            ExactSelectionConfig(),
            max_delete_count=max_delete_count,
        ),
        fallback,
        candidate_k=candidate_k,
    )


def test_composed_exact_success_skips_retrieval_and_llm() -> None:
    fallback = RecordingFallbackStrategy(
        DeleteDecision(selected_ids=[], reasoning="unused", ambiguous=False)
    )

    def unexpected_retrieve(query: str, top_k: int) -> list[DeleteCandidate]:
        pytest.fail("Exact success must not invoke top-k retrieval")

    decision = _composed(fallback).select(
        _plan("User likes tea"),
        [
            _candidate("memory-1", "User likes tea"),
            _candidate("memory-2", "User likes coffee"),
        ],
        retrieve=unexpected_retrieve,
    )

    assert decision.selected_ids == ["memory-1"]
    assert decision.ambiguous is False
    assert fallback.calls == []


def test_composed_exact_miss_invokes_top_k_and_llm() -> None:
    fallback = RecordingFallbackStrategy(
        DeleteDecision(
            selected_ids=["memory-2"],
            reasoning="Semantic match.",
            ambiguous=False,
        )
    )
    retrieval_calls: list[tuple[str, int]] = []
    retrieved = [_candidate("memory-2", "Tea preference", score=0.8)]

    def retrieve(query: str, top_k: int) -> list[DeleteCandidate]:
        retrieval_calls.append((query, top_k))
        return retrieved

    query_plan = _plan("User likes tea", search_query="tea preference")
    decision = _composed(fallback, candidate_k=3).select(
        query_plan,
        [_candidate("memory-1", "User likes coffee")],
        retrieve=retrieve,
    )

    assert retrieval_calls == [("tea preference", 3)]
    assert fallback.calls == [(query_plan, retrieved)]
    assert decision.selected_ids == ["memory-2"]


def test_composed_exact_ambiguity_stops_without_fallback() -> None:
    fallback = RecordingFallbackStrategy(
        DeleteDecision(selected_ids=[], reasoning="unused", ambiguous=False)
    )

    def unexpected_retrieve(query: str, top_k: int) -> list[DeleteCandidate]:
        pytest.fail("Exact ambiguity must not invoke top-k retrieval")

    decision = _composed(fallback).select(
        _plan("Same fact", cardinality="one"),
        [
            _candidate("memory-1", "Same fact"),
            _candidate("memory-2", "Same fact"),
        ],
        retrieve=unexpected_retrieve,
    )

    assert decision.selected_ids == []
    assert decision.ambiguous is True
    assert fallback.calls == []


@pytest.mark.parametrize(
    ("selection_mode", "expected_type"),
    [
        ("exact", ExactDeleteStrategy),
        ("top_k_llm", LlmJudgeDeleteStrategy),
        ("exact_then_top_k_llm", ExactThenTopKLLMDeleteStrategy),
    ],
)
def test_factory_selects_configured_strategy(
    selection_mode: str,
    expected_type: type[BaseDeleteStrategy],
) -> None:
    strategy = build_delete_strategy(
        DeleteStrategyConfig(selection_mode=selection_mode)
    )

    assert isinstance(strategy, expected_type)


def test_composed_selection_does_not_mutate_candidates() -> None:
    fallback = RecordingFallbackStrategy(
        DeleteDecision(selected_ids=[], reasoning="No match.", ambiguous=False)
    )
    candidates = [_candidate("memory-1", "User likes coffee")]
    before = [candidate.model_copy(deep=True) for candidate in candidates]

    _composed(fallback).select(
        _plan("User likes tea"),
        candidates,
        retrieve=lambda query, top_k: [],
    )

    assert candidates == before
