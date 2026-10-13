from __future__ import annotations

import pytest

from heams_memory_milamem.design import delete
from heams_memory_milamem.design.delete.config import ExactSelectionConfig
from heams_memory_milamem.design.delete.models import DeleteQueryPlan
from heams_memory_milamem.design.delete.strategies import (
    DeleteCandidate,
    ExactDeleteStrategy,
)
from heams_memory_milamem.design.delete.strategies import (
    llm_judge as llm_strategy_module,
)


def _plan(
    target_text: str,
    *,
    cardinality: str = "one",
) -> DeleteQueryPlan:
    return DeleteQueryPlan.model_validate(
        {
            "original_query": f"Forget {target_text}",
            "target_text": target_text,
            "search_query": target_text,
            "cardinality": cardinality,
        }
    )


def _candidate(memory_id: str, content: str) -> DeleteCandidate:
    return DeleteCandidate(memory_id=memory_id, content=content, score=1.0)


def test_exact_strategy_selects_unique_exact_match() -> None:
    strategy = ExactDeleteStrategy(ExactSelectionConfig())

    decision = strategy.select(
        _plan("User likes tea"),
        [
            _candidate("memory-1", "User likes tea"),
            _candidate("memory-2", "User likes coffee"),
        ],
    )

    assert decision.selected_ids == ["memory-1"]
    assert decision.ambiguous is False


def test_exact_strategy_matches_case_insensitively_by_default() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig()).select(
        _plan("user likes tea"),
        [_candidate("memory-1", "User Likes Tea")],
    )

    assert decision.selected_ids == ["memory-1"]


def test_exact_strategy_normalizes_whitespace_by_default() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig()).select(
        _plan("User   likes\n tea"),
        [_candidate("memory-1", "User likes tea")],
    )

    assert decision.selected_ids == ["memory-1"]


def test_exact_strategy_respects_case_sensitive_mismatch() -> None:
    strategy = ExactDeleteStrategy(ExactSelectionConfig(case_sensitive=True))

    decision = strategy.select(
        _plan("user likes tea"),
        [_candidate("memory-1", "User likes tea")],
    )

    assert decision.selected_ids == []
    assert decision.ambiguous is False


def test_exact_strategy_marks_multiple_singular_matches_ambiguous() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig()).select(
        _plan("Same fact", cardinality="one"),
        [
            _candidate("memory-1", "Same fact"),
            _candidate("memory-2", "Same fact"),
        ],
    )

    assert decision.selected_ids == []
    assert decision.ambiguous is True


def test_exact_strategy_selects_multiple_matches_for_all_cardinality() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig(), max_delete_count=2).select(
        _plan("Same fact", cardinality="all"),
        [
            _candidate("memory-1", "Same fact"),
            _candidate("memory-2", "Same fact"),
        ],
    )

    assert decision.selected_ids == ["memory-1", "memory-2"]
    assert decision.ambiguous is False


def test_exact_strategy_fails_closed_when_matches_exceed_cap() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig(), max_delete_count=1).select(
        _plan("Same fact", cardinality="all"),
        [
            _candidate("memory-1", "Same fact"),
            _candidate("memory-2", "Same fact"),
        ],
    )

    assert decision.selected_ids == []
    assert decision.ambiguous is True


def test_exact_strategy_returns_no_selection_without_match() -> None:
    decision = ExactDeleteStrategy(ExactSelectionConfig()).select(
        _plan("User likes tea"),
        [_candidate("memory-1", "User likes coffee")],
    )

    assert decision.selected_ids == []
    assert decision.ambiguous is False


def test_exact_strategy_never_invokes_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm_strategy_module,
        "request_delete_decision",
        lambda model, messages: pytest.fail("Exact selection must not invoke an LLM"),
    )

    decision = ExactDeleteStrategy(ExactSelectionConfig()).select(
        _plan("User likes tea"),
        [_candidate("memory-1", "User likes tea")],
    )

    assert decision.selected_ids == ["memory-1"]


def test_exact_strategy_is_exported() -> None:
    assert delete.ExactDeleteStrategy is ExactDeleteStrategy
