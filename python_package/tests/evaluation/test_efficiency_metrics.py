"""Unit tests for efficiency metrics aggregation module."""

import enum
from datetime import timedelta
from types import SimpleNamespace

import pytest

# Import benchmarks before evaluation so the factory/scenarios cycle is already loaded.
import heams.benchmarks  # noqa: F401
from heams.agents.base import AgentResponse
from heams.evaluation.efficiency import (
    compute_system_metrics,
    enhance_component_metrics,
)
from heams.evaluation.schemas import ComponentMetricsSummary
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationStatus,
    MemoryOperationType,
    MemoryOperationSource,
)
from heams.results import summarize_memory_ledger


class TestComputeSystemMetrics:
    """Tests for system-level efficiency metrics computation."""

    def test_empty_transcript_and_ledger(self):
        """System metrics with no data should handle gracefully."""
        metrics = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": True, "accuracy_score": 0.0},
            memory_ledger=None,
        )
        assert metrics.stimulus_count == 0
        assert metrics.agent_turn_count == 0
        assert metrics.run_duration_ms is None
        assert metrics.memory_operation_count == 0
        assert metrics.memory_operation_error_count == 0
        assert metrics.recall_call_count == 0
        assert metrics.retain_call_count == 0
        assert metrics.delete_call_count == 0
        assert metrics.consolidate_call_count == 0
        assert metrics.llm_call_count == 0
        assert metrics.llm_total_tokens == 0
        assert metrics.llm_total_cost == 0.0
        assert metrics.agent_duration_ms == 0.0
        assert metrics.tool_execution_count == 0
        assert metrics.memory_calls_per_stimulus is None
        assert metrics.task_accuracy_per_memory_call is None
        assert metrics.task_success_per_recall_call is None

    def test_trace_durations_and_priced_tokens(self):
        """Provider span tokens price known models; a non-zero trace cost is kept."""

        class _StatusCode(enum.Enum):
            ERROR = 2

        def span(
            operation,
            model,
            input_tokens,
            output_tokens,
            start,
            end,
            error=False,
            status_code=None,
        ):
            if status_code is None:
                status_code = "error" if error else "ok"
            return SimpleNamespace(
                attributes={
                    "gen_ai.operation.name": operation,
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": input_tokens,
                    "gen_ai.usage.output_tokens": output_tokens,
                },
                start_time=start,
                end_time=end,
                status=SimpleNamespace(status_code=status_code),
                is_llm_call=lambda: operation == "call_llm",
                is_tool_execution=lambda: operation == "execute_tool",
            )

        priced = SimpleNamespace(
            spans=[
                span("call_llm", "gpt-5.6-luna", 1000, 100, 0, 2_000_000),
                span("execute_tool", "gpt-5.6-luna", 0, 0, 0, 1_000_000),
                span("execute_tool", "gpt-5.6-luna", 0, 0, 0, 3_000_000, error=True),
                span(
                    "execute_tool",
                    "gpt-5.6-luna",
                    0,
                    0,
                    0,
                    0,
                    status_code=_StatusCode.ERROR,
                ),
            ],
            tokens=SimpleNamespace(
                input_tokens=1000, output_tokens=100, total_tokens=1100
            ),
            cost=SimpleNamespace(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            duration=timedelta(milliseconds=250),
        )
        billed = SimpleNamespace(
            spans=[span("call_llm", "gpt-5.6-luna", 5000, 5000, 0, 0)],
            tokens=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=SimpleNamespace(input_cost=1.0, output_cost=2.0, total_cost=3.0),
            duration=timedelta(0),
        )
        unknown = SimpleNamespace(
            spans=[span("call_llm", "mystery-model", 1000, 1000, 0, 0)],
            tokens=SimpleNamespace(
                input_tokens=1000, output_tokens=1000, total_tokens=2000
            ),
            cost=SimpleNamespace(input_cost=0.0, output_cost=0.0, total_cost=0.0),
            duration=timedelta(0),
        )

        metrics = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": True, "accuracy_score": 1.0},
            memory_ledger=None,
            agent_responses=[
                AgentResponse(answer="priced", raw_trace=priced),
                AgentResponse(answer="billed", raw_trace=billed),
                AgentResponse(answer="unknown", raw_trace=unknown),
            ],
        )

        assert metrics.llm_call_count == 3
        assert metrics.llm_input_tokens == 2001
        assert metrics.llm_output_tokens == 1101
        assert metrics.llm_input_cost == pytest.approx(1.0002)
        assert metrics.llm_output_cost == pytest.approx(2.00012)
        assert metrics.llm_total_cost == pytest.approx(3.00032)
        assert metrics.agent_duration_ms == pytest.approx(250)
        assert metrics.llm_duration_ms == pytest.approx(2)
        assert metrics.tool_execution_count == 3
        assert metrics.tool_execution_error_count == 2
        assert metrics.tool_duration_ms == pytest.approx(4)

    def test_transcript_stimulus_count(self):
        """Stimulus count should be independently controllable from transcript turns."""
        transcript = [
            type("AgentTurn", (), {"question": "q1", "answer": "a1"})(),
            type("AgentTurn", (), {"question": "q2", "answer": "a2"})(),
            type("AgentTurn", (), {"question": "q3", "answer": "a3"})(),
        ]
        metrics = compute_system_metrics(
            transcript=transcript,
            audit_data={"is_successful": True, "accuracy_score": 1.0},
            memory_ledger=None,
            stimulus_count=2,
        )
        assert metrics.stimulus_count == 2
        assert metrics.agent_turn_count == 3

    def test_memory_operation_counts(self):
        """Operation counts should be correctly aggregated from ledger."""
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.ERROR,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.DELETE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
        ]
        metrics = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": False, "accuracy_score": 0.5},
            memory_ledger=ledger,
        )
        assert metrics.memory_operation_count == 5
        assert metrics.memory_operation_error_count == 1
        assert metrics.recall_call_count == 2
        assert metrics.retain_call_count == 1
        assert metrics.delete_call_count == 1
        assert metrics.consolidate_call_count == 1

    def test_memory_calls_per_stimulus(self):
        """Per-stimulus ratio should be computed correctly."""
        transcript = [
            type("AgentTurn", (), {"question": "q", "answer": "a"})(),
            type("AgentTurn", (), {"question": "q", "answer": "a"})(),
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
        ]
        metrics = compute_system_metrics(
            transcript=transcript,
            audit_data={"is_successful": True, "accuracy_score": 1.0},
            memory_ledger=ledger,
        )
        assert metrics.memory_calls_per_stimulus == 2.0  # 4 ops / 2 stimuli

    def test_task_accuracy_per_memory_call(self):
        """Accuracy per memory call should divide accuracy by operation count."""
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
        ]
        metrics = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": True, "accuracy_score": 0.8},
            memory_ledger=ledger,
        )
        assert metrics.task_accuracy_per_memory_call == 0.4  # 0.8 / 2

    def test_task_accuracy_per_memory_call_is_none_without_calls(self):
        metrics = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": True, "accuracy_score": 0.8},
            memory_ledger=None,
        )
        assert metrics.task_accuracy_per_memory_call is None

    def test_task_success_per_recall_call(self):
        """Success metric should divide success flag (0/1) by recall count."""
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
        ]
        metrics_success = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": True, "accuracy_score": 1.0},
            memory_ledger=ledger,
        )
        assert metrics_success.task_success_per_recall_call == 0.5  # 1.0 / 2

        metrics_failure = compute_system_metrics(
            transcript=[],
            audit_data={"is_successful": False, "accuracy_score": 0.0},
            memory_ledger=ledger,
        )
        assert metrics_failure.task_success_per_recall_call == 0.0  # 0.0 / 2

    def test_derived_task_metrics_are_none_when_score_unavailable(self):
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            )
        ]
        metrics = compute_system_metrics(
            transcript=[type("AgentTurn", (), {"question": "q", "answer": "a"})()],
            audit_data={
                "is_successful": False,
                "accuracy_score": 0.0,
                "score_available": False,
            },
            memory_ledger=ledger,
        )
        assert metrics.memory_calls_per_stimulus == 1.0
        assert metrics.task_accuracy_per_memory_call is None
        assert metrics.task_success_per_recall_call is None


class TestEnhanceComponentMetrics:
    """Tests for component-level efficiency enhancement."""

    def test_empty_ledger_returns_unchanged_metrics(self):
        """Empty ledger should return component metrics unchanged."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
            )
        ]
        enhanced = enhance_component_metrics(base_metrics, None)
        assert len(enhanced) == 1
        assert enhanced[0].component == "mem0"
        assert enhanced[0].avg_recall_duration_ms is None
        assert enhanced[0].p95_recall_duration_ms is None

    def test_recall_duration_averaging(self):
        """Average recall duration should be computed correctly."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=2,
                successful_operations=2,
                failed_operations=0,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                duration_ms=100.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                duration_ms=200.0,
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].avg_recall_duration_ms == 150.0  # (100 + 200) / 2

    def test_p95_recall_duration(self):
        """P95 recall duration should be computed from durations."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=5,
                successful_operations=5,
                failed_operations=0,
            )
        ]
        durations = [100.0, 110.0, 120.0, 130.0, 150.0]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                duration_ms=d,
            )
            for d in durations
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].p95_recall_duration_ms is not None
        assert enhanced[0].p95_recall_duration_ms == pytest.approx(146.0)

    def test_operation_type_separation(self):
        """Different operation types should have separate metrics."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=4,
                successful_operations=4,
                failed_operations=0,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                duration_ms=100.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                duration_ms=50.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.DELETE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                duration_ms=80.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                duration_ms=200.0,
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].avg_recall_duration_ms == 100.0
        assert enhanced[0].avg_retain_duration_ms == 50.0
        assert enhanced[0].avg_delete_duration_ms == 80.0
        assert enhanced[0].avg_consolidate_duration_ms == 200.0

    def test_content_metrics_from_result(self):
        """Content metrics should be extracted from result dict."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=2,
                successful_operations=2,
                failed_operations=0,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                result={"result_count": 5, "output_bytes": 1024},
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                result={"result_count": 3, "output_bytes": 512},
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].avg_recall_results_count == 4.0  # (5 + 3) / 2
        assert enhanced[0].avg_recall_output_bytes == 768.0  # (1024 + 512) / 2
        assert enhanced[0].total_recall_output_bytes == 1536

    def test_content_metrics_from_payload(self):
        """Retain content metrics should be extracted from payload."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=2,
                successful_operations=2,
                failed_operations=0,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                payload={"content_bytes": 2048},
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                payload={"content_bytes": 1024},
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].avg_retain_content_bytes == 1536.0  # (2048 + 1024) / 2
        assert enhanced[0].total_retained_content_bytes == 3072

    def test_retains_dropped_by_a_policy_do_not_count_as_failures(self):
        """A governed system declining content is not a failed retain."""
        base_metrics = [
            ComponentMetricsSummary(
                component="privacy_retention",
                total_operations=3,
                successful_operations=1,
                failed_operations=0,
                operation_counts={"retain": 3},
                status_counts={"success": 1, "dropped": 1, "error": 1},
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=status,
                source=MemoryOperationSource.RUNTIME,
                component="privacy_retention",
            )
            for status in (
                MemoryOperationStatus.SUCCESS,
                MemoryOperationStatus.DROPPED,
                MemoryOperationStatus.ERROR,
            )
        ]

        enhanced = enhance_component_metrics(base_metrics, ledger)

        assert enhanced[0].retain_success_rate == 0.5

    def test_dynamic_memory_efficiency_metrics(self):
        """Retain/delete/consolidate dynamic metrics should derive from existing ledger and run data."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=5,
                successful_operations=3,
                failed_operations=1,
                skipped_operations=1,
                operation_counts={"retain": 2, "delete": 1, "consolidate": 2},
                status_counts={"success": 3, "error": 1, "skipped": 1},
                source_counts={"runtime": 5},
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                payload={"content_bytes": 2048},
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.ERROR,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                payload={"content_bytes": 64},
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.DELETE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SKIPPED,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
        ]

        enhanced = enhance_component_metrics(base_metrics, ledger, stimulus_count=4)
        assert enhanced[0].retain_success_rate == 0.5
        assert enhanced[0].delete_success_rate == 1.0
        assert enhanced[0].consolidate_success_rate == 0.5
        assert enhanced[0].retain_operations_per_stimulus == 0.5
        assert enhanced[0].delete_operations_per_stimulus == 0.25
        assert enhanced[0].consolidate_operations_per_stimulus == 0.5
        assert enhanced[0].retain_bytes_per_successful_retain == 2048.0

    def test_content_metrics_from_runtime_field_aliases(self):
        """Runtime-emitted alias fields should normalize into shared metrics."""
        base_metrics = [
            ComponentMetricsSummary(
                component="example.in_memory",
                total_operations=2,
                successful_operations=2,
                failed_operations=0,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="example.in_memory",
                result={"returned_count": 2, "output_size_chars": 120},
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="example.in_memory",
                payload={"payload_size_chars": 80},
            ),
        ]

        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].avg_recall_results_count == 2.0
        assert enhanced[0].avg_recall_output_bytes == 120.0
        assert enhanced[0].total_recall_output_bytes == 120
        assert enhanced[0].avg_retain_content_bytes == 80.0
        assert enhanced[0].total_retained_content_bytes == 80

    def test_error_counting_by_operation_type(self):
        """Errors should be counted separately for each operation type."""
        base_metrics = [
            ComponentMetricsSummary(
                component="mem0",
                total_operations=4,
                successful_operations=2,
                failed_operations=2,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.ERROR,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                error="Not found",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.ERROR,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                error="Failed to write",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.DELETE,
                status=MemoryOperationStatus.ERROR,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                error="Delete failed",
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].recall_errors == 1
        assert enhanced[0].retain_errors == 1
        assert enhanced[0].delete_errors == 1
        assert enhanced[0].consolidate_errors == 0

    def test_multi_component_ledger(self):
        """Multiple components should each receive their own enhanced metrics."""
        base_metrics = [
            ComponentMetricsSummary(component="mem0", total_operations=1),
            ComponentMetricsSummary(component="example.in_memory", total_operations=1),
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="mem0",
                duration_ms=100.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RECALL,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.TOOL_CALL,
                component="example.in_memory",
                duration_ms=200.0,
            ),
        ]
        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert len(enhanced) == 2
        assert enhanced[0].component == "mem0"
        assert enhanced[0].avg_recall_duration_ms == 100.0
        assert enhanced[1].component == "example.in_memory"
        assert enhanced[1].avg_recall_duration_ms == 200.0

    def test_runs_without_ledger_succeed(self):
        """Runs with no ledger should not crash and should emit empty metrics."""
        transcript = [
            type("AgentTurn", (), {"question": "q", "answer": "a"})(),
        ]
        metrics = compute_system_metrics(
            transcript=transcript,
            audit_data={"is_successful": False, "accuracy_score": 0.0},
            memory_ledger=None,
        )
        assert metrics.stimulus_count == 1
        assert metrics.memory_operation_count == 0
        assert metrics.recall_call_count == 0
        assert metrics.memory_calls_per_stimulus == 0.0
        assert metrics.task_accuracy_per_memory_call is None
        assert metrics.task_success_per_recall_call is None

    def test_skipped_operations_are_counted_and_excluded_from_latency(self):
        """Skipped operations should reconcile counts without polluting latency summaries."""
        base_metrics = [
            ComponentMetricsSummary(
                component="in_context",
                total_operations=2,
                successful_operations=0,
                failed_operations=0,
                skipped_operations=2,
            )
        ]
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SKIPPED,
                source=MemoryOperationSource.RUNTIME,
                component="in_context",
                duration_ms=5.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SKIPPED,
                source=MemoryOperationSource.RUNTIME,
                component="in_context",
                duration_ms=7.0,
            ),
        ]

        enhanced = enhance_component_metrics(base_metrics, ledger)
        assert enhanced[0].skipped_operations == 2
        assert enhanced[0].total_duration_ms is None
        assert enhanced[0].avg_consolidate_duration_ms is None

    def test_summarize_memory_ledger_excludes_skipped_durations(self):
        ledger = [
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                duration_ms=12.0,
            ),
            MemoryOperationEvent(
                operation_type=MemoryOperationType.CONSOLIDATE,
                status=MemoryOperationStatus.SKIPPED,
                source=MemoryOperationSource.RUNTIME,
                component="mem0",
                duration_ms=50.0,
            ),
        ]

        summary = summarize_memory_ledger(ledger)[0]
        assert summary.total_operations == 2
        assert summary.skipped_operations == 1
        assert summary.total_duration_ms == 12.0
