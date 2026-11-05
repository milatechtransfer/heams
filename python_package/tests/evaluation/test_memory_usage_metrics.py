"""Memory-system model usage in system metrics and reports."""

from types import SimpleNamespace

import pytest

# Import benchmarks before evaluation so the factory/scenarios cycle is already loaded.
import heams.benchmarks  # noqa: F401
from heams.evaluation.efficiency import compute_system_metrics
from heams.evaluation.report_metrics import (
    EXPERIMENT_SYSTEM_REPORT_METRICS,
    NOT_REPORTED,
)
from heams.memory_systems.schemas import MemoryOperationEvent, ModelUsage


def _event(usage: list[ModelUsage] | None) -> MemoryOperationEvent:
    return MemoryOperationEvent(
        operation_type="retain", status="success", source="runtime", usage=usage
    )


def _metrics(ledger: list[MemoryOperationEvent], agent_tokens: int = 0):
    trace = SimpleNamespace(
        spans=[],
        duration=None,
        tokens=SimpleNamespace(
            input_tokens=agent_tokens, output_tokens=0, total_tokens=agent_tokens
        ),
        cost=SimpleNamespace(input_cost=0.5, output_cost=0.0, total_cost=0.5),
    )
    return compute_system_metrics(
        transcript=[],
        audit_data={},
        memory_ledger=ledger,
        agent_responses=[SimpleNamespace(raw_trace=trace)],
    )


def _report(metrics) -> dict[str, object]:
    return {
        name: getter(metrics) for name, getter, _ in EXPERIMENT_SYSTEM_REPORT_METRICS
    }


def test_memory_usage_is_totalled_priced_and_added_to_agent_usage() -> None:
    metrics = _metrics(
        [
            _event(
                [
                    ModelUsage(
                        kind="llm",
                        model="openai:gpt-4o-mini",
                        call_count=2,
                        input_tokens=1_000,
                        output_tokens=100,
                    ),
                    ModelUsage(
                        kind="embedding",
                        model="openai:text-embedding-3-small",
                        call_count=3,
                        input_tokens=500,
                    ),
                ]
            ),
            _event([ModelUsage(kind="llm", model="unpriced-model", input_tokens=7)]),
            _event([]),
        ],
        agent_tokens=40,
    )

    assert metrics.memory_usage_reported_operation_count == 3
    assert metrics.memory_usage_unreported_operation_count == 0
    assert metrics.memory_llm_call_count == 3
    assert metrics.memory_llm_input_tokens == 1_007
    assert metrics.memory_llm_output_tokens == 100
    assert metrics.memory_embedding_call_count == 3
    assert metrics.memory_embedding_input_tokens == 500
    assert metrics.memory_total_tokens == 1_607
    assert metrics.memory_input_cost == pytest.approx(1_000 * 1.5e-7 + 500 * 2e-8)
    assert metrics.memory_output_cost == pytest.approx(100 * 6e-7)
    assert metrics.memory_total_cost == pytest.approx(
        metrics.memory_input_cost + metrics.memory_output_cost
    )
    # Agent fields stay agent-only; the totals combine both.
    assert metrics.llm_total_tokens == 40
    assert metrics.total_tokens == 40 + 1_607
    assert metrics.total_cost == pytest.approx(0.5 + metrics.memory_total_cost)


def test_operations_without_usage_are_counted_and_left_out() -> None:
    metrics = _metrics(
        [_event([ModelUsage(kind="llm", model="m", input_tokens=10)]), _event(None)]
    )

    assert metrics.memory_usage_reported_operation_count == 1
    assert metrics.memory_usage_unreported_operation_count == 1
    assert metrics.memory_total_tokens == 10


def test_unreported_memory_usage_reads_as_not_reported_not_zero() -> None:
    metrics = _metrics([_event(None), _event(None)], agent_tokens=40)

    assert metrics.memory_total_tokens is None
    assert metrics.memory_total_cost is None
    assert metrics.total_tokens == 40
    report = _report(metrics)
    assert report["Memory total tokens"] == NOT_REPORTED
    assert report["Memory total cost"] == NOT_REPORTED
    assert report["Memory ops not reporting usage"] == 2


def test_a_run_without_memory_operations_has_no_memory_usage() -> None:
    metrics = _metrics([])

    assert metrics.memory_total_tokens is None
    assert _report(metrics)["Memory total tokens"] is None
    assert "memory_total_tokens" not in metrics.model_dump()


def test_local_models_are_free_and_reported_cost_wins() -> None:
    metrics = _metrics(
        [
            _event(
                [
                    ModelUsage(
                        kind="embedding",
                        model="st:Qwen/Qwen3-Embedding-0.6B",
                        local=True,
                        input_tokens=4_000,
                    ),
                    # Priced in the table, but the system knows its real cost.
                    ModelUsage(
                        kind="llm",
                        model="openai:gpt-4o-mini",
                        input_tokens=1_000,
                        cost=0.01,
                    ),
                    # A local model must stay free even if its id has a price.
                    ModelUsage(
                        kind="llm", model="gpt-4o-mini", local=True, input_tokens=500
                    ),
                ]
            )
        ]
    )

    assert metrics.memory_local_tokens == 4_500
    assert metrics.memory_total_tokens == 5_500
    assert metrics.memory_total_cost == pytest.approx(0.01)


def test_averaged_report_tables_keep_not_reported() -> None:
    from heams.reporting.tables import render_mean_metric_rows

    unreported = _metrics([_event(None)])
    reported = _metrics([_event([ModelUsage(kind="llm", model="m", input_tokens=10)])])
    specs = [
        spec
        for spec in EXPERIMENT_SYSTEM_REPORT_METRICS
        if spec[0] == "Memory total tokens"
    ]

    ((_, cells),) = render_mean_metric_rows(
        specs,
        ["a", "b"],
        {"a": [unreported, unreported], "b": [reported, unreported]},
    )

    # Every sample unreported keeps the label; a mix averages what was reported.
    assert cells == [NOT_REPORTED, "10.00"]
