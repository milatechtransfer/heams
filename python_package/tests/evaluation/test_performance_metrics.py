"""Unit tests for component-level recall performance metrics."""

from types import SimpleNamespace

from heams.evaluation.performance import compute_component_performance_metrics
from heams.evaluation.factory import TaskQAEvaluatorConfig
from heams.evaluation.schemas import ComponentMetricsSummary
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationSource,
    MemoryOperationStatus,
    MemoryOperationType,
)
from heams.scenarios import MemoryPriming, Scenario, ScenarioPriming, UserStimulus


def _scenario_with_markers(markers_list: list[dict]) -> Scenario:
    return Scenario(
        id="scenario-a",
        name="Scenario A",
        execution_sequence=[
            UserStimulus(content=f"q{index}", step_evaluation_markers=markers)
            for index, markers in enumerate(markers_list)
        ],
        agents=[],
        memory_systems=[],
        evaluators=[],
        scenario_priming=ScenarioPriming(memory=MemoryPriming(entries=[])),
    )


def _recall_event(
    stimulus_index: int, returned_items: list[str]
) -> MemoryOperationEvent:
    return MemoryOperationEvent(
        operation_type=MemoryOperationType.RECALL,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.RUNTIME,
        component="example.in_memory",
        payload={"run_context": {"stimulus_index": stimulus_index}},
        result={"returned_items": returned_items},
    )


def test_compute_component_performance_metrics_uses_explicit_retrieval_evidence() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["s2"],
                }
            },
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["s9"],
                }
            },
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "a", "metadata": {"session_id": "s1"}}',
                '{"content": "b", "metadata": {"session_id": "s2"}}',
            ],
        ),
        _recall_event(
            1,
            ['{"content": "c", "metadata": {"session_id": "s1"}}'],
        ),
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert len(metrics) == 1
    assert metrics[0].component == "example.in_memory"
    assert metrics[0].recall_eval_count == 2
    assert metrics[0].recall_hit_at_k == 0.5
    assert metrics[0].recall_mrr == 0.25


def test_compute_component_performance_metrics_supports_legacy_evidence_fields() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {"evidence_dia_ids": ["D1:1"]},
            {"answer_session_ids": ["S1"]},
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            ['{"content": "a", "metadata": {"dia_id": "D1:1"}}'],
        ),
        _recall_event(
            1,
            ['{"content": "b", "metadata": {"session_id": "S1"}}'],
        ),
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 2
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_returns_na_shape_when_evidence_missing() -> (
    None
):
    scenario = _scenario_with_markers([{}, {}])
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(0, ['{"content": "a", "metadata": {"session_id": "s1"}}']),
        _recall_event(1, ['{"content": "b", "metadata": {"session_id": "s2"}}']),
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 0
    assert metrics[0].recall_missing_reference_evidence_count == 2
    assert metrics[0].recall_hit_at_k is None
    assert metrics[0].recall_mrr is None


def test_compute_component_performance_metrics_tracks_missing_stimulus_index() -> None:
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["s1"],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        MemoryOperationEvent(
            operation_type=MemoryOperationType.RECALL,
            status=MemoryOperationStatus.SUCCESS,
            source=MemoryOperationSource.RUNTIME,
            component="example.in_memory",
            payload={},
            result={
                "returned_items": ['{"content": "a", "metadata": {"session_id": "s1"}}']
            },
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 0
    assert metrics[0].recall_missing_stimulus_index_count == 1
    assert metrics[0].recall_hit_at_k is None
    assert metrics[0].recall_mrr is None


def test_compute_component_performance_metrics_treats_missing_returned_items_as_unevaluable() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["s2"],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        MemoryOperationEvent(
            operation_type=MemoryOperationType.RECALL,
            status=MemoryOperationStatus.SUCCESS,
            source=MemoryOperationSource.RUNTIME,
            component="example.in_memory",
            payload={"run_context": {"stimulus_index": 0}},
            result={},
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 0
    assert metrics[0].recall_missing_result_evidence_count == 1
    assert metrics[0].recall_hit_at_k is None
    assert metrics[0].recall_mrr is None


def test_compute_component_performance_metrics_keeps_zero_for_evaluable_empty_results() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["s2"],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [_recall_event(0, [])]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 0.0
    assert metrics[0].recall_mrr == 0.0


def test_compute_component_performance_metrics_matches_synthesized_memories_by_content() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "dia_id",
                    "relevant_values": ["D1:3"],
                    "relevant_contents": [
                        "I went to a LGBTQ support group yesterday and it was so powerful."
                    ],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "User, Caroline, attended an LGBTQ support group on May 6, 2026, which she found to be powerful and inspiring.", "metadata": {"session_id": "session_1"}}'
            ],
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_matches_long_mem_eval_content_without_session_id() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["S1"],
                    "relevant_contents": ["I moved to Montreal last spring."],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "Alice said she moved to Montreal last spring, and the assistant replied that it sounded exciting.", "metadata": {"source": "long_mem_eval"}}'
            ],
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_supports_content_only_evidence() -> None:
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "relevant_contents": [
                        "The user changed jobs last winter and now works as a park ranger."
                    ]
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "She changed jobs last winter and now works as a park ranger in Banff.", "metadata": {"source": "synthesized"}}'
            ],
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_prefers_textual_evidence_over_ids() -> (
    None
):
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "match_key": "session_id",
                    "relevant_values": ["stale-session-id"],
                    "relevant_contents": [
                        "Alice moved to Montreal last spring and started a new job."
                    ],
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "Alice moved to Montreal last spring and started a new job downtown.", "metadata": {"session_id": "fresh-session-id"}}'
            ],
        )
    ]

    metrics = compute_component_performance_metrics(base, ledger, scenario)

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_can_use_llm_judge_strategy(
    monkeypatch,
) -> None:
    scenario = _scenario_with_markers(
        [
            {
                "retrieval_evidence": {
                    "relevant_contents": [
                        "She reconciled with her estranged sister after the funeral."
                    ]
                }
            }
        ]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            [
                '{"content": "They made up after the burial.", "metadata": {"source": "semantic"}}'
            ],
        )
    ]
    calls: list[tuple[str, str, str]] = []

    class _FakeJudge:
        def __init__(self, model_id: str):
            self.model_id = model_id

        def run(self, *, context: str, question: str):
            calls.append((self.model_id, context, question))
            return SimpleNamespace(passed="made up" in context.lower(), reasoning="ok")

    monkeypatch.setattr(
        "heams.evaluation.performance._get_retrieval_judge",
        lambda model_id: _FakeJudge(model_id),
    )

    metrics = compute_component_performance_metrics(
        base,
        ledger,
        scenario,
        eval_cfg=TaskQAEvaluatorConfig(
            retrieval_match_mode="llm_judge",
            retrieval_judge_model_id="ollama:test-retrieval-judge",
        ),
    )

    assert calls
    assert calls[0][0] == "ollama:test-retrieval-judge"
    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0


def test_compute_component_performance_metrics_llm_judge_mode_falls_back_to_ids(
    monkeypatch,
) -> None:
    scenario = _scenario_with_markers(
        [{"retrieval_evidence": {"match_key": "session_id", "relevant_values": ["s2"]}}]
    )
    base = [ComponentMetricsSummary(component="example.in_memory")]
    ledger = [
        _recall_event(
            0,
            ['{"content": "a", "metadata": {"session_id": "s2"}}'],
        )
    ]

    monkeypatch.setattr(
        "heams.evaluation.performance._get_retrieval_judge",
        lambda model_id: (_ for _ in ()).throw(AssertionError("judge should not run")),
    )

    metrics = compute_component_performance_metrics(
        base,
        ledger,
        scenario,
        eval_cfg=TaskQAEvaluatorConfig(retrieval_match_mode="llm_judge"),
    )

    assert metrics[0].recall_eval_count == 1
    assert metrics[0].recall_hit_at_k == 1.0
    assert metrics[0].recall_mrr == 1.0
