from __future__ import annotations

import pytest

from heams_memory_milamem import design
from heams_memory_milamem.design.delete import lifecycle as lifecycle_module
from heams_memory_milamem.design import delete
from heams_memory_milamem.design.delete.lifecycle import (
    DeleteLifecycle,
    build_active_delete_candidates,
    build_delete_candidates,
    build_delete_plan,
    execute_delete_plan,
    verify_logical_delete,
)
from heams_memory_milamem.design.delete.models import (
    DeletePlan,
    DeleteQueryPlan,
    DeleteReceipt,
)
from heams_memory_milamem.design.delete.planners import BaseDeleteQueryPlanner
from heams_memory_milamem.design.delete.planners import llm as llm_planner_module
from heams_memory_milamem.design.delete.strategies import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
    DeleteStrategyConfig,
    ExactDeleteStrategy,
    ExactThenTopKLLMDeleteStrategy,
    build_delete_strategy,
)
from heams_memory_milamem.design.delete.strategies import (
    llm_judge as llm_strategy_module,
)
from heams_memory_milamem.design.storage_type import CanonicalMemoryRecord
from heams.memory_systems.api import MemoryEntry


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


def _scored_ranker(
    entries: list[MemoryEntry],
    query: str,
    top_k: int,
) -> list[MemoryEntry]:
    del query
    return [
        entry.model_copy(update={"score": float(index) + 0.25})
        for index, entry in enumerate(reversed(entries))
    ][:top_k]


def _candidate(memory_id: str, score: float = 1.0) -> DeleteCandidate:
    return DeleteCandidate(
        memory_id=memory_id,
        content=f"Content for {memory_id}",
        score=score,
    )


def _plan(
    selected_ids: list[str],
    *,
    executable: bool = True,
) -> DeletePlan:
    return DeletePlan(
        query="Forget selected facts.",
        strategy="llm_judge",
        candidate_ids=list(selected_ids),
        selected_ids=list(selected_ids),
        reasoning="The selected facts are directly targeted.",
        ambiguous=False,
        executable=executable,
    )


class RecordingDeleteStrategy(BaseDeleteStrategy):
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


def test_build_delete_candidates_filters_to_active_records() -> None:
    records = [
        _record("active-1", "Active fact", 1),
        _record("deleted-1", "Deleted fact", 2, status="deleted"),
        _record("superseded-1", "Old fact", 3, status="superseded"),
    ]
    seen_entries: list[MemoryEntry] = []

    def rank(entries: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
        seen_entries.extend(entries)
        return _scored_ranker(entries, query, top_k)

    all_candidates = build_active_delete_candidates(records)
    candidates = build_delete_candidates("fact", all_candidates, top_k=5, rank=rank)

    assert [entry.content for entry in seen_entries] == ["Active fact"]
    assert [candidate.memory_id for candidate in candidates] == ["active-1"]
    assert [record.status for record in records] == [
        "active",
        "deleted",
        "superseded",
    ]


def test_build_delete_candidates_limits_ranker_and_output_to_top_k() -> None:
    records = [
        _record("memory-1", "Fact one", 1),
        _record("memory-2", "Fact two", 2),
        _record("memory-3", "Fact three", 3),
    ]
    received_top_k: list[int] = []

    def rank(entries: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
        del query
        received_top_k.append(top_k)
        return [entry.model_copy(update={"score": 1.0}) for entry in entries]

    candidates = build_delete_candidates(
        "fact",
        build_active_delete_candidates(records),
        top_k=2,
        rank=rank,
    )

    assert received_top_k == [2]
    assert len(candidates) == 2


def test_build_delete_candidates_preserves_ids_with_duplicate_content() -> None:
    records = [
        _record("memory-1", "Same content", 1),
        _record("memory-2", "Same content", 2),
    ]

    candidates = build_delete_candidates(
        "same",
        build_active_delete_candidates(records),
        top_k=2,
        rank=_scored_ranker,
    )

    assert [candidate.memory_id for candidate in candidates] == [
        "memory-2",
        "memory-1",
    ]
    assert [candidate.content for candidate in candidates] == [
        "Same content",
        "Same content",
    ]


def test_build_delete_candidates_preserves_scores() -> None:
    records = [
        _record("memory-1", "Fact one", 1),
        _record("memory-2", "Fact two", 2),
    ]

    candidates = build_delete_candidates(
        "fact",
        build_active_delete_candidates(records),
        top_k=2,
        rank=_scored_ranker,
    )

    assert [candidate.score for candidate in candidates] == [0.25, 1.25]


def test_build_delete_candidates_returns_empty_without_records() -> None:
    def unexpected_rank(
        entries: list[MemoryEntry], query: str, top_k: int
    ) -> list[MemoryEntry]:
        pytest.fail("Ranker should not be called without active records")

    assert build_delete_candidates("fact", [], top_k=5, rank=unexpected_rank) == []


@pytest.mark.parametrize(
    ("ranked", "message"),
    [
        ([MemoryEntry(content="Fact", score=1.0)], "canonical memory ID"),
        (
            [
                MemoryEntry(
                    content="Fact",
                    metadata={"_canonical_memory_id": "memory-1"},
                )
            ],
            "score",
        ),
    ],
)
def test_build_delete_candidates_fails_clearly_for_invalid_ranked_entries(
    ranked: list[MemoryEntry],
    message: str,
) -> None:
    records = [_record("memory-1", "Fact", 1)]

    with pytest.raises(ValueError, match=message):
        build_delete_candidates(
            "fact",
            build_active_delete_candidates(records),
            top_k=1,
            rank=lambda entries, query, top_k: ranked,
        )


def test_build_delete_plan_normalizes_empty_query_and_is_not_executable() -> None:
    plan = build_delete_plan(
        " \t\n ",
        [_candidate("memory-1")],
        DeleteDecision(
            selected_ids=["memory-1"],
            reasoning="The candidate matches.",
            ambiguous=False,
        ),
        strategy="llm_judge",
    )

    assert plan.query == ""
    assert plan.selected_ids == []
    assert plan.executable is False


def test_build_delete_plan_clears_ambiguous_selection() -> None:
    plan = build_delete_plan(
        "Forget tea.",
        [_candidate("memory-1")],
        DeleteDecision(
            selected_ids=["memory-1"],
            reasoning="Multiple interpretations remain.",
            ambiguous=True,
        ),
        strategy="llm_judge",
    )

    assert plan.ambiguous is True
    assert plan.selected_ids == []
    assert plan.executable is False


def test_build_delete_plan_with_unmatched_decision_is_not_executable() -> None:
    plan = build_delete_plan(
        "Forget tea.",
        [_candidate("memory-1")],
        DeleteDecision(
            selected_ids=[],
            reasoning="No candidate is directly targeted.",
            ambiguous=False,
        ),
        strategy="llm_judge",
    )

    assert plan.selected_ids == []
    assert plan.executable is False


def test_build_delete_plan_clears_unknown_selected_id() -> None:
    plan = build_delete_plan(
        "Forget tea.",
        [_candidate("memory-1")],
        DeleteDecision(
            selected_ids=["unknown"],
            reasoning="An unknown ID was selected.",
            ambiguous=False,
        ),
        strategy="llm_judge",
    )

    assert plan.candidate_ids == ["memory-1"]
    assert plan.selected_ids == []
    assert plan.executable is False


def test_build_delete_plan_deduplicates_selected_ids_in_order() -> None:
    plan = build_delete_plan(
        "  Forget   both facts. ",
        [_candidate("memory-1"), _candidate("memory-2")],
        DeleteDecision(
            selected_ids=["memory-2", "memory-1", "memory-2"],
            reasoning="Both facts are directly targeted.",
            ambiguous=False,
        ),
        strategy="llm_judge",
    )

    assert plan.query == "Forget both facts."
    assert plan.selected_ids == ["memory-2", "memory-1"]
    assert plan.executable is True


def test_build_delete_plan_preserves_ranked_candidates_and_is_executable() -> None:
    candidates = [_candidate("memory-2", 0.9), _candidate("memory-1", 0.8)]
    reasoning = "The first ranked candidate is directly targeted."

    plan = build_delete_plan(
        "Forget tea.",
        candidates,
        DeleteDecision(
            selected_ids=["memory-2"],
            reasoning=reasoning,
            ambiguous=False,
        ),
        strategy="llm_judge",
    )

    assert plan.candidate_ids == ["memory-2", "memory-1"]
    assert plan.selected_ids == ["memory-2"]
    assert plan.reasoning == reasoning
    assert plan.strategy == "llm_judge"
    assert plan.ambiguous is False
    assert plan.executable is True


def test_execute_delete_plan_tombstones_active_record_and_preserves_fields() -> None:
    records = [_record("memory-1", "Fact one", 1)]
    original = records[0]

    receipt = execute_delete_plan(
        _plan(["memory-1"]),
        records,
        deleted_at=123.5,
    )

    assert receipt.requested_ids == ["memory-1"]
    assert receipt.deleted_ids == ["memory-1"]
    assert receipt.skipped_ids == []
    assert receipt.raw_erasure is False
    assert records[0].status == "deleted"
    assert records[0].deleted_at == 123.5
    assert records[0].model_dump(
        exclude={"status", "deleted_at"}
    ) == original.model_dump(exclude={"status", "deleted_at"})


def test_execute_delete_plan_uses_one_deleted_at_for_all_records() -> None:
    records = [
        _record("memory-1", "Fact one", 1),
        _record("memory-2", "Fact two", 2),
    ]

    receipt = execute_delete_plan(
        _plan(["memory-1", "memory-2"]),
        records,
        deleted_at=456.75,
    )

    assert receipt.deleted_ids == ["memory-1", "memory-2"]
    assert [record.deleted_at for record in records] == [456.75, 456.75]


@pytest.mark.parametrize(
    ("selected_id", "records"),
    [
        ("missing", [_record("memory-1", "Fact one", 1)]),
        (
            "deleted-1",
            [_record("deleted-1", "Deleted fact", 1, status="deleted")],
        ),
        (
            "superseded-1",
            [_record("superseded-1", "Old fact", 1, status="superseded")],
        ),
    ],
)
def test_execute_delete_plan_skips_non_active_or_missing_ids(
    selected_id: str,
    records: list[CanonicalMemoryRecord],
) -> None:
    before = [record.model_copy(deep=True) for record in records]

    receipt = execute_delete_plan(
        _plan([selected_id]),
        records,
        deleted_at=123.5,
    )

    assert receipt.deleted_ids == []
    assert receipt.skipped_ids == [selected_id]
    assert records == before


def test_execute_delete_plan_reports_mixed_deleted_and_skipped_ids() -> None:
    records = [
        _record("active-1", "Active fact", 1),
        _record("deleted-1", "Deleted fact", 2, status="deleted"),
        _record("superseded-1", "Old fact", 3, status="superseded"),
    ]

    receipt = execute_delete_plan(
        _plan(["active-1", "missing", "deleted-1", "superseded-1"]),
        records,
        deleted_at=123.5,
    )

    assert receipt.deleted_ids == ["active-1"]
    assert receipt.skipped_ids == ["missing", "deleted-1", "superseded-1"]


def test_execute_delete_plan_does_not_mutate_for_non_executable_plan() -> None:
    records = [_record("memory-1", "Fact one", 1)]
    before = records[0].model_copy(deep=True)

    receipt = execute_delete_plan(
        _plan(["memory-1"], executable=False),
        records,
        deleted_at=123.5,
    )

    assert receipt.deleted_ids == []
    assert receipt.skipped_ids == ["memory-1"]
    assert records == [before]


def test_verify_logical_delete_returns_verified_after_tombstone() -> None:
    records = [_record("memory-1", "Fact one", 1)]
    plan = _plan(["memory-1"])
    receipt = execute_delete_plan(plan, records, deleted_at=123.5)

    verification = verify_logical_delete(plan, receipt, records)

    assert verification.status == "verified"
    assert verification.remaining_active_ids == []


def test_verify_logical_delete_reports_failed_active_ids() -> None:
    records = [_record("memory-1", "Fact one", 1)]
    plan = _plan(["memory-1"])
    receipt = DeleteReceipt(
        requested_ids=["memory-1"],
        deleted_ids=["memory-1"],
        skipped_ids=[],
    )

    verification = verify_logical_delete(plan, receipt, records)

    assert verification.status == "failed"
    assert verification.remaining_active_ids == ["memory-1"]


@pytest.mark.parametrize(
    ("plan", "receipt"),
    [
        (
            _plan(["memory-1"], executable=False),
            DeleteReceipt(
                requested_ids=["memory-1"],
                deleted_ids=[],
                skipped_ids=["memory-1"],
            ),
        ),
        (
            _plan(["memory-1"]),
            DeleteReceipt(
                requested_ids=["memory-1"],
                deleted_ids=[],
                skipped_ids=["memory-1"],
            ),
        ),
    ],
)
def test_verify_logical_delete_returns_not_run_without_execution(
    plan: DeletePlan,
    receipt: DeleteReceipt,
) -> None:
    verification = verify_logical_delete(plan, receipt, [])

    assert verification.status == "not_run"
    assert verification.remaining_active_ids == []


def test_delete_lifecycle_runs_selection_execution_and_verification() -> None:
    records = [_record("memory-1", "Fact one", 1)]
    strategy = RecordingDeleteStrategy(
        DeleteDecision(
            selected_ids=["memory-1"],
            reasoning="The candidate is directly targeted.",
            ambiguous=False,
        )
    )
    lifecycle = DeleteLifecycle(
        DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"candidate_k": 1},
        ),
        strategy,
    )

    outcome = lifecycle.run(
        "  Forget   fact one. ",
        records,
        rank=_scored_ranker,
        deleted_at=123.5,
    )

    assert len(strategy.calls) == 1
    assert outcome.plan.query == "Forget fact one."
    assert outcome.plan.executable is True
    assert outcome.receipt is not None
    assert outcome.receipt.deleted_ids == ["memory-1"]
    assert outcome.verification.status == "verified"
    assert outcome.changed is True


def test_delete_lifecycle_ranks_with_query_plan_search_query() -> None:
    class SearchQueryPlanner(BaseDeleteQueryPlanner):
        def plan(self, query: str) -> DeleteQueryPlan:
            return DeleteQueryPlan(
                original_query=query,
                target_text="jasmine tea",
                search_query="tea preference",
                cardinality="one",
            )

    records = [_record("memory-1", "jasmine tea", 1)]
    strategy = RecordingDeleteStrategy(
        DeleteDecision(
            selected_ids=[],
            reasoning="No deletion in retrieval test.",
            ambiguous=False,
        )
    )
    ranked_queries: list[str] = []

    def rank(entries: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
        ranked_queries.append(query)
        return [entry.model_copy(update={"score": 1.0}) for entry in entries[:top_k]]

    DeleteLifecycle(
        DeleteStrategyConfig(
            selection_mode="top_k_llm",
            top_k_llm={"candidate_k": 1},
        ),
        strategy,
        query_planner=SearchQueryPlanner(),
    ).run(
        "Forget that I like jasmine tea.",
        records,
        rank=rank,
        deleted_at=123.5,
    )

    assert ranked_queries == ["tea preference"]
    assert strategy.calls[0][0].original_query == "Forget that I like jasmine tea."
    assert strategy.calls[0][0].search_query == "tea preference"


@pytest.mark.parametrize(
    ("query", "records"),
    [("", [_record("memory-1", "Fact one", 1)]), ("Forget fact.", [])],
)
def test_delete_lifecycle_avoids_unnecessary_rank_and_strategy_calls(
    query: str,
    records: list[CanonicalMemoryRecord],
) -> None:
    strategy = RecordingDeleteStrategy(
        DeleteDecision(
            selected_ids=["memory-1"],
            reasoning="This should not be used.",
            ambiguous=False,
        )
    )
    lifecycle = DeleteLifecycle(
        DeleteStrategyConfig(selection_mode="top_k_llm"), strategy
    )

    def unexpected_rank(
        entries: list[MemoryEntry], ranked_query: str, top_k: int
    ) -> list[MemoryEntry]:
        pytest.fail("Ranker should not be called for empty lifecycle input")

    outcome = lifecycle.run(
        query,
        records,
        rank=unexpected_rank,
        deleted_at=123.5,
    )

    assert strategy.calls == []
    assert outcome.plan.executable is False
    assert outcome.receipt is not None
    assert outcome.receipt.deleted_ids == []
    assert outcome.verification.status == "not_run"
    assert outcome.changed is False


def test_delete_lifecycle_helpers_are_exported_from_design() -> None:
    assert design.DeleteLifecycle is DeleteLifecycle
    assert delete.DeleteLifecycle is DeleteLifecycle
    assert delete.build_active_delete_candidates is build_active_delete_candidates
    assert delete.build_delete_candidates is build_delete_candidates
    assert delete.build_delete_plan is build_delete_plan
    assert delete.execute_delete_plan is execute_delete_plan
    assert delete.verify_logical_delete is verify_logical_delete


def test_active_candidate_conversion_preserves_ids_content_and_optional_scores() -> (
    None
):
    candidates = build_active_delete_candidates(
        [
            _record("memory-1", "Same content", 1),
            _record("memory-2", "Same content", 2),
            _record("deleted-1", "Deleted", 3, status="deleted"),
            _record("superseded-1", "Old", 4, status="superseded"),
        ]
    )

    assert [(candidate.memory_id, candidate.content) for candidate in candidates] == [
        ("memory-1", "Same content"),
        ("memory-2", "Same content"),
    ]
    assert [candidate.score for candidate in candidates] == [None, None]


def test_exact_only_lifecycle_does_not_call_ranker() -> None:
    config = DeleteStrategyConfig(selection_mode="exact")
    strategy = build_delete_strategy(config)
    assert isinstance(strategy, ExactDeleteStrategy)

    def unexpected_rank(
        entries: list[MemoryEntry], query: str, top_k: int
    ) -> list[MemoryEntry]:
        pytest.fail("Exact-only selection must not invoke ranking")

    outcome = DeleteLifecycle(config, strategy).run(
        "Forget User likes tea",
        [_record("memory-1", "User likes tea", 1)],
        rank=unexpected_rank,
        deleted_at=123.5,
    )

    assert outcome.changed is True


def test_composed_exact_success_lifecycle_does_not_call_ranker() -> None:
    config = DeleteStrategyConfig(selection_mode="exact_then_top_k_llm")
    strategy = build_delete_strategy(config)
    assert isinstance(strategy, ExactThenTopKLLMDeleteStrategy)

    def unexpected_rank(
        entries: list[MemoryEntry], query: str, top_k: int
    ) -> list[MemoryEntry]:
        pytest.fail("Composed exact success must not invoke ranking")

    outcome = DeleteLifecycle(config, strategy).run(
        "Forget User likes tea",
        [_record("memory-1", "User likes tea", 1)],
        rank=unexpected_rank,
        deleted_at=123.5,
    )

    assert outcome.changed is True


def test_top_k_lifecycle_respects_candidate_k() -> None:
    config = DeleteStrategyConfig(
        selection_mode="top_k_llm",
        top_k_llm={"candidate_k": 1},
    )
    strategy = RecordingDeleteStrategy(
        DeleteDecision(selected_ids=[], reasoning="No match.", ambiguous=False)
    )
    received_top_k: list[int] = []

    def rank(entries: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
        received_top_k.append(top_k)
        return [entry.model_copy(update={"score": 1.0}) for entry in entries[:top_k]]

    DeleteLifecycle(config, strategy).run(
        "Find a fact",
        [
            _record("memory-1", "Fact one", 1),
            _record("memory-2", "Fact two", 2),
        ],
        rank=rank,
        deleted_at=123.5,
    )

    assert received_top_k == [1]
    assert len(strategy.calls[0][1]) == 1
    assert strategy.calls[0][1][0].score == 1.0


def test_lifecycle_converts_active_records_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = DeleteStrategyConfig(selection_mode="top_k_llm")
    strategy = RecordingDeleteStrategy(
        DeleteDecision(selected_ids=[], reasoning="No match.", ambiguous=False)
    )
    conversion_calls: list[list[CanonicalMemoryRecord]] = []
    original_converter = lifecycle_module.build_active_delete_candidates

    def recording_converter(
        records: list[CanonicalMemoryRecord],
    ) -> list[DeleteCandidate]:
        conversion_calls.append(records)
        return original_converter(records)

    monkeypatch.setattr(
        lifecycle_module,
        "build_active_delete_candidates",
        recording_converter,
    )

    DeleteLifecycle(config, strategy).run(
        "Find a fact",
        [_record("memory-1", "Fact one", 1)],
        rank=_scored_ranker,
        deleted_at=123.5,
    )

    assert len(conversion_calls) == 1


@pytest.mark.parametrize(
    (
        "planning_mode",
        "selection_mode",
        "expected_planner_calls",
        "expected_judge_calls",
        "expected_rank_calls",
    ),
    [
        ("none", "exact", 0, 0, 0),
        ("none", "top_k_llm", 0, 1, 1),
        ("none", "exact_then_top_k_llm", 0, 0, 0),
        ("llm", "exact", 1, 0, 0),
        ("llm", "top_k_llm", 1, 1, 1),
        ("llm", "exact_then_top_k_llm", 1, 0, 0),
    ],
)
def test_delete_lifecycle_planning_selection_matrix(
    monkeypatch: pytest.MonkeyPatch,
    planning_mode: str,
    selection_mode: str,
    expected_planner_calls: int,
    expected_judge_calls: int,
    expected_rank_calls: int,
) -> None:
    planner_calls: list[None] = []
    judge_calls: list[None] = []
    rank_calls: list[None] = []

    def request_plan(model: str, messages: list[dict[str, object]]) -> DeleteQueryPlan:
        planner_calls.append(None)
        return DeleteQueryPlan(
            original_query="model output is overwritten",
            target_text="User likes tea",
            search_query="tea preference",
            cardinality="one",
            exclusions=["coffee"],
            reasoning="The request targets one preference.",
        )

    def request_decision(
        model: str,
        messages: list[dict[str, object]],
    ) -> DeleteDecision:
        judge_calls.append(None)
        return DeleteDecision(
            selected_ids=["memory-1"],
            reasoning="The candidate is directly targeted.",
            ambiguous=False,
        )

    def rank(entries: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
        rank_calls.append(None)
        return [entry.model_copy(update={"score": 1.0}) for entry in entries[:top_k]]

    monkeypatch.setattr(
        llm_planner_module,
        "request_delete_query_plan",
        request_plan,
    )
    monkeypatch.setattr(
        llm_strategy_module,
        "request_delete_decision",
        request_decision,
    )
    config = DeleteStrategyConfig(
        planning={"mode": planning_mode},
        selection_mode=selection_mode,
    )
    records = [_record("memory-1", "User likes tea", 1)]

    outcome = DeleteLifecycle(
        config,
        build_delete_strategy(config),
    ).run(
        "Forget User likes tea",
        records,
        rank=rank,
        deleted_at=123.5,
    )

    assert len(planner_calls) == expected_planner_calls
    assert len(judge_calls) == expected_judge_calls
    assert len(rank_calls) == expected_rank_calls
    assert outcome.changed is True
    assert outcome.plan.planning_mode == planning_mode
    assert outcome.plan.query_plan is not None
    assert outcome.plan.query_plan.original_query == "Forget User likes tea"
    assert outcome.plan.query_plan.cardinality == "one"
    assert outcome.plan.query_plan.exclusions == (
        ["coffee"] if planning_mode == "llm" else []
    )
    assert "system_prompt" not in outcome.plan.model_dump_json()
    assert "user_prompt_template" not in outcome.plan.model_dump_json()
