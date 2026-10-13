"""Post-run aggregation of system-level and component-level efficiency metrics.

This module derives metrics purely from RunResult data (transcript, audit,
memory_ledger) without modifying memory implementations. All aggregations
are deterministic and side-effect-free.
"""

from collections import defaultdict
from math import ceil, floor
from typing import Any, Sequence

from heams.agents.base import AgentResponse
from heams.evaluation.schemas import ComponentMetricsSummary, SystemMetricsSummary
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationStatus,
    MemoryOperationType,
)


def _bounded_percentile(values: list[float], percentile: float) -> float | None:
    """Compute a bounded percentile using linear interpolation.

    This keeps the result within the observed min/max range, which is more
    appropriate for reporting small-sample latency percentiles.
    """
    if not values:
        return None

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    bounded_percentile = min(max(percentile, 0.0), 100.0)
    rank = (len(ordered) - 1) * (bounded_percentile / 100.0)
    lower_index = floor(rank)
    upper_index = ceil(rank)
    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = rank - lower_index
    return lower_value + (upper_value - lower_value) * weight


def _extract_nonnegative_metric(data: object, *keys: str) -> float | None:
    if not isinstance(data, dict):
        return None

    for key in keys:
        value = data.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)

    return None


# ponytail: static OpenAI list prices (USD per token) for the models in
# configs/agents. Unknown models stay 0. A billing API replaces this when
# prices drift.
_MODEL_USD_PER_TOKEN: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5e-6, 1e-5),
    "gpt-4o-mini": (1.5e-7, 6e-7),
    "gpt-5.6-luna": (2e-7, 1.2e-6),
}


def _is_llm_call_span(span: Any) -> bool:
    is_llm_call = getattr(span, "is_llm_call", None)
    if callable(is_llm_call):
        try:
            return bool(is_llm_call())
        except Exception:
            return False

    attributes = getattr(span, "attributes", None)
    return (
        isinstance(attributes, dict)
        and attributes.get("gen_ai.operation.name") == "call_llm"
    )


def _is_tool_execution_span(span: Any) -> bool:
    is_tool_execution = getattr(span, "is_tool_execution", None)
    if callable(is_tool_execution):
        try:
            return bool(is_tool_execution())
        except Exception:
            return False

    attributes = getattr(span, "attributes", None)
    return (
        isinstance(attributes, dict)
        and attributes.get("gen_ai.operation.name") == "execute_tool"
    )


def _span_duration_ms(span: Any) -> float:
    # OpenTelemetry span times are nanoseconds.
    start = getattr(span, "start_time", None)
    end = getattr(span, "end_time", None)
    if (
        isinstance(start, (int, float))
        and not isinstance(start, bool)
        and isinstance(end, (int, float))
        and not isinstance(end, bool)
        and end >= start
    ):
        return (end - start) / 1_000_000
    return 0.0


def _span_status_is_error(span: Any) -> bool:
    status = getattr(span, "status", None)
    code = getattr(status, "status_code", None)
    if code is None:
        return False
    # OpenTelemetry StatusCode.ERROR has name "ERROR" and value 2.
    name = getattr(code, "name", None)
    if isinstance(name, str) and name.casefold() == "error":
        return True
    value = getattr(code, "value", code)
    return str(value).casefold() == "error"


def _model_token_price(model_id: object) -> tuple[float, float] | None:
    if not isinstance(model_id, str) or not model_id:
        return None
    return _MODEL_USD_PER_TOKEN.get(model_id.rpartition(":")[2])


def _price_llm_spans(spans: list[Any]) -> tuple[float, float]:
    input_cost = 0.0
    output_cost = 0.0
    for span in spans:
        if not _is_llm_call_span(span):
            continue
        attributes = getattr(span, "attributes", None)
        if not isinstance(attributes, dict):
            continue
        price = _model_token_price(attributes.get("gen_ai.request.model"))
        if price is None:
            continue
        input_tokens = attributes.get("gen_ai.usage.input_tokens") or 0
        output_tokens = attributes.get("gen_ai.usage.output_tokens") or 0
        if isinstance(input_tokens, (int, float)) and not isinstance(
            input_tokens, bool
        ):
            input_cost += float(input_tokens) * price[0]
        if isinstance(output_tokens, (int, float)) and not isinstance(
            output_tokens, bool
        ):
            output_cost += float(output_tokens) * price[1]
    return input_cost, output_cost


def _summarize_agent_traces(
    agent_responses: Sequence[AgentResponse] | None,
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "llm_call_count": 0,
        "llm_input_tokens": 0,
        "llm_output_tokens": 0,
        "llm_total_tokens": 0,
        "llm_input_cost": 0.0,
        "llm_output_cost": 0.0,
        "llm_total_cost": 0.0,
        "agent_duration_ms": 0.0,
        "llm_duration_ms": 0.0,
        "tool_execution_count": 0,
        "tool_execution_error_count": 0,
        "tool_duration_ms": 0.0,
    }
    if not agent_responses:
        return summary

    for response in agent_responses:
        trace = getattr(response, "raw_trace", None)
        if trace is None:
            continue

        spans = list(getattr(trace, "spans", None) or [])
        summary["llm_call_count"] += sum(1 for span in spans if _is_llm_call_span(span))
        summary["llm_duration_ms"] += sum(
            _span_duration_ms(span) for span in spans if _is_llm_call_span(span)
        )
        tool_spans = [span for span in spans if _is_tool_execution_span(span)]
        summary["tool_execution_count"] += len(tool_spans)
        summary["tool_execution_error_count"] += sum(
            1 for span in tool_spans if _span_status_is_error(span)
        )
        summary["tool_duration_ms"] += sum(
            _span_duration_ms(span) for span in tool_spans
        )

        try:
            duration = trace.duration
        except Exception:
            duration = None
        total_seconds = getattr(duration, "total_seconds", None)
        if callable(total_seconds):
            summary["agent_duration_ms"] += total_seconds() * 1000

        try:
            tokens = getattr(trace, "tokens", None)
        except Exception:
            tokens = None
        if tokens is not None:
            input_tokens = int(getattr(tokens, "input_tokens", 0) or 0)
            output_tokens = int(getattr(tokens, "output_tokens", 0) or 0)
            total_tokens = int(
                getattr(tokens, "total_tokens", input_tokens + output_tokens) or 0
            )
            summary["llm_input_tokens"] += input_tokens
            summary["llm_output_tokens"] += output_tokens
            summary["llm_total_tokens"] += total_tokens

        try:
            cost = getattr(trace, "cost", None)
        except Exception:
            cost = None
        input_cost = 0.0
        output_cost = 0.0
        total_cost = 0.0
        if cost is not None:
            input_cost = float(getattr(cost, "input_cost", 0.0) or 0.0)
            output_cost = float(getattr(cost, "output_cost", 0.0) or 0.0)
            total_cost = float(
                getattr(cost, "total_cost", input_cost + output_cost) or 0.0
            )
        if total_cost == 0.0:
            input_cost, output_cost = _price_llm_spans(spans)
            total_cost = input_cost + output_cost
        summary["llm_input_cost"] += input_cost
        summary["llm_output_cost"] += output_cost
        summary["llm_total_cost"] += total_cost

    return summary


def compute_system_metrics(
    transcript: list,
    audit_data: dict,
    memory_ledger: list[MemoryOperationEvent] | None,
    agent_responses: Sequence[AgentResponse] | None = None,
    run_duration_ms: float | None = None,
    stimulus_count: int | None = None,
) -> SystemMetricsSummary:
    """Derive system-level efficiency metrics from run data and ledger.

    Args:
        transcript: List of AgentTurn objects representing dialog exchanges.
        audit_data: Dictionary with 'is_successful' (bool) and 'accuracy_score' (float).
        memory_ledger: Optional list of MemoryOperationEvent objects.

    Returns:
        SystemMetricsSummary with computed system-level metrics.
    """
    stimulus_count = len(transcript) if stimulus_count is None else stimulus_count
    agent_turn_count = len(transcript)
    trace_summary = _summarize_agent_traces(agent_responses)

    # Extract memory operation counts from ledger
    memory_operation_count = 0
    memory_operation_error_count = 0
    memory_operation_skipped_count = 0
    recall_call_count = 0
    retain_call_count = 0
    delete_call_count = 0
    consolidate_call_count = 0

    if memory_ledger:
        memory_operation_count = len(memory_ledger)
        memory_operation_error_count = sum(
            1 for event in memory_ledger if event.status == MemoryOperationStatus.ERROR
        )
        memory_operation_skipped_count = sum(
            1
            for event in memory_ledger
            if event.status == MemoryOperationStatus.SKIPPED
        )
        recall_call_count = sum(
            1
            for event in memory_ledger
            if event.operation_type == MemoryOperationType.RECALL
        )
        retain_call_count = sum(
            1
            for event in memory_ledger
            if event.operation_type == MemoryOperationType.RETAIN
        )
        delete_call_count = sum(
            1
            for event in memory_ledger
            if event.operation_type == MemoryOperationType.DELETE
        )
        consolidate_call_count = sum(
            1
            for event in memory_ledger
            if event.operation_type == MemoryOperationType.CONSOLIDATE
        )

    # Compute derived ratios (safe division by nonzero)
    if stimulus_count > 0:
        memory_calls_per_stimulus: float | None = (
            memory_operation_count / stimulus_count
        )
    else:
        memory_calls_per_stimulus = None

    score_available = audit_data.get("score_available", True)
    raw_accuracy_score = audit_data.get("accuracy_score")
    accuracy_score = (
        float(raw_accuracy_score)
        if score_available
        and isinstance(raw_accuracy_score, (int, float))
        and not isinstance(raw_accuracy_score, bool)
        else None
    )
    task_accuracy_per_memory_call = (
        accuracy_score / memory_operation_count
        if accuracy_score is not None and memory_operation_count > 0
        else None
    )

    success_value = audit_data.get("is_successful") if score_available else None
    task_success_per_recall_call = (
        (1.0 if bool(success_value) else 0.0) / recall_call_count
        if isinstance(success_value, bool) and recall_call_count > 0
        else None
    )

    return SystemMetricsSummary(
        run_duration_ms=run_duration_ms,
        stimulus_count=stimulus_count,
        agent_turn_count=agent_turn_count,
        memory_operation_count=memory_operation_count,
        memory_operation_error_count=memory_operation_error_count,
        memory_operation_skipped_count=memory_operation_skipped_count,
        recall_call_count=recall_call_count,
        retain_call_count=retain_call_count,
        delete_call_count=delete_call_count,
        consolidate_call_count=consolidate_call_count,
        llm_call_count=int(trace_summary["llm_call_count"]),
        llm_input_tokens=int(trace_summary["llm_input_tokens"]),
        llm_output_tokens=int(trace_summary["llm_output_tokens"]),
        llm_total_tokens=int(trace_summary["llm_total_tokens"]),
        llm_input_cost=float(trace_summary["llm_input_cost"]),
        llm_output_cost=float(trace_summary["llm_output_cost"]),
        llm_total_cost=float(trace_summary["llm_total_cost"]),
        agent_duration_ms=float(trace_summary["agent_duration_ms"]),
        llm_duration_ms=float(trace_summary["llm_duration_ms"]),
        tool_execution_count=int(trace_summary["tool_execution_count"]),
        tool_execution_error_count=int(trace_summary["tool_execution_error_count"]),
        tool_duration_ms=float(trace_summary["tool_duration_ms"]),
        memory_calls_per_stimulus=memory_calls_per_stimulus,
        task_accuracy_per_memory_call=task_accuracy_per_memory_call,
        task_success_per_recall_call=task_success_per_recall_call,
    )


def enhance_component_metrics(
    component_metrics: list[ComponentMetricsSummary],
    memory_ledger: list[MemoryOperationEvent] | None,
    stimulus_count: int | None = None,
) -> list[ComponentMetricsSummary]:
    """Enhance component metrics with duration percentiles and content metrics.

    Computes efficiency fields (avg/p95 durations, content sizes, error counts)
    for each component from raw ledger events. Returns a new list with all
    fields populated.

    Args:
        component_metrics: Pre-aggregated component summaries from summarize_memory_ledger.
        memory_ledger: Raw list of MemoryOperationEvent objects.

    Returns:
        Enhanced list of ComponentMetricsSummary with efficiency fields populated.
    """
    if not memory_ledger:
        return component_metrics

    # Group ledger events by component for easy access
    events_by_component: dict[str, list[MemoryOperationEvent]] = defaultdict(list)
    for event in memory_ledger:
        component = event.component or "unknown"
        events_by_component[component].append(event)

    # Enhance each summary with derived efficiency metrics
    enhanced: list[ComponentMetricsSummary] = []
    for summary in component_metrics:
        component = summary.component
        events = events_by_component.get(component, [])

        retain_total = summary.operation_counts.get(MemoryOperationType.RETAIN.value, 0)
        delete_total = summary.operation_counts.get(MemoryOperationType.DELETE.value, 0)
        consolidate_total = summary.operation_counts.get(
            MemoryOperationType.CONSOLIDATE.value, 0
        )
        successful_retain_count = 0
        successful_delete_count = 0
        successful_consolidate_count = 0

        # Collect durations by operation type
        recall_durations: list[float] = []
        retain_durations: list[float] = []
        delete_durations: list[float] = []
        consolidate_durations: list[float] = []

        for event in events:
            # Skip no-op events from latency summaries so averages reflect work.
            if (
                event.duration_ms is not None
                and event.status != MemoryOperationStatus.SKIPPED
            ):
                if event.operation_type == MemoryOperationType.RECALL:
                    recall_durations.append(event.duration_ms)
                elif event.operation_type == MemoryOperationType.RETAIN:
                    retain_durations.append(event.duration_ms)
                elif event.operation_type == MemoryOperationType.DELETE:
                    delete_durations.append(event.duration_ms)
                elif event.operation_type == MemoryOperationType.CONSOLIDATE:
                    consolidate_durations.append(event.duration_ms)

            if (
                event.operation_type == MemoryOperationType.RETAIN
                and event.status == MemoryOperationStatus.SUCCESS
            ):
                successful_retain_count += 1
            elif (
                event.operation_type == MemoryOperationType.DELETE
                and event.status == MemoryOperationStatus.SUCCESS
            ):
                successful_delete_count += 1
            elif (
                event.operation_type == MemoryOperationType.CONSOLIDATE
                and event.status == MemoryOperationStatus.SUCCESS
            ):
                successful_consolidate_count += 1

        # Compute average durations
        avg_recall_duration_ms = (
            sum(recall_durations) / len(recall_durations) if recall_durations else None
        )
        avg_retain_duration_ms = (
            sum(retain_durations) / len(retain_durations) if retain_durations else None
        )
        avg_delete_duration_ms = (
            sum(delete_durations) / len(delete_durations) if delete_durations else None
        )
        avg_consolidate_duration_ms = (
            sum(consolidate_durations) / len(consolidate_durations)
            if consolidate_durations
            else None
        )

        # Compute p95 durations (95th percentile)
        p95_recall_duration_ms = _bounded_percentile(recall_durations, 95.0)

        # Compute content metrics from result and payload
        recall_results_counts: list[float] = []
        recall_output_bytes: list[float] = []
        retain_content_bytes: list[float] = []
        successful_retain_content_bytes: list[float] = []
        total_recall_output = 0.0
        total_retained_content = 0.0

        for event in events:
            if event.operation_type == MemoryOperationType.RECALL:
                result_count = _extract_nonnegative_metric(
                    event.result, "result_count", "returned_count"
                )
                if result_count is not None:
                    recall_results_counts.append(result_count)

                output_bytes = _extract_nonnegative_metric(
                    event.result, "output_bytes", "output_size_chars"
                )
                if output_bytes is not None:
                    recall_output_bytes.append(output_bytes)
                    total_recall_output += output_bytes
            elif event.operation_type == MemoryOperationType.RETAIN:
                content_bytes = _extract_nonnegative_metric(
                    event.payload, "content_bytes", "payload_size_chars"
                )
                if content_bytes is not None:
                    retain_content_bytes.append(content_bytes)
                    total_retained_content += content_bytes
                    if event.status == MemoryOperationStatus.SUCCESS:
                        successful_retain_content_bytes.append(content_bytes)

        retain_success_rate = (
            successful_retain_count / retain_total if retain_total > 0 else None
        )
        delete_success_rate = (
            successful_delete_count / delete_total if delete_total > 0 else None
        )
        consolidate_success_rate = (
            successful_consolidate_count / consolidate_total
            if consolidate_total > 0
            else None
        )
        retain_operations_per_stimulus = (
            retain_total / stimulus_count
            if isinstance(stimulus_count, int) and stimulus_count > 0
            else None
        )
        delete_operations_per_stimulus = (
            delete_total / stimulus_count
            if isinstance(stimulus_count, int) and stimulus_count > 0
            else None
        )
        consolidate_operations_per_stimulus = (
            consolidate_total / stimulus_count
            if isinstance(stimulus_count, int) and stimulus_count > 0
            else None
        )
        retain_bytes_per_successful_retain = (
            sum(successful_retain_content_bytes) / successful_retain_count
            if successful_retain_count > 0
            and len(successful_retain_content_bytes) == successful_retain_count
            else None
        )

        # Compute averages
        avg_recall_results_count = (
            sum(recall_results_counts) / len(recall_results_counts)
            if recall_results_counts
            else None
        )
        avg_recall_output_bytes = (
            sum(recall_output_bytes) / len(recall_output_bytes)
            if recall_output_bytes
            else None
        )
        avg_retain_content_bytes = (
            sum(retain_content_bytes) / len(retain_content_bytes)
            if retain_content_bytes
            else None
        )

        # Count errors by operation type
        recall_errors = sum(
            1
            for event in events
            if event.operation_type == MemoryOperationType.RECALL
            and event.status == MemoryOperationStatus.ERROR
        )
        retain_errors = sum(
            1
            for event in events
            if event.operation_type == MemoryOperationType.RETAIN
            and event.status == MemoryOperationStatus.ERROR
        )
        delete_errors = sum(
            1
            for event in events
            if event.operation_type == MemoryOperationType.DELETE
            and event.status == MemoryOperationStatus.ERROR
        )
        consolidate_errors = sum(
            1
            for event in events
            if event.operation_type == MemoryOperationType.CONSOLIDATE
            and event.status == MemoryOperationStatus.ERROR
        )

        # Create enhanced summary
        enhanced.append(
            ComponentMetricsSummary(
                component=summary.component,
                component_type=summary.component_type,
                total_operations=summary.total_operations,
                successful_operations=summary.successful_operations,
                failed_operations=summary.failed_operations,
                skipped_operations=summary.skipped_operations,
                operation_counts=summary.operation_counts,
                status_counts=summary.status_counts,
                source_counts=summary.source_counts,
                total_duration_ms=summary.total_duration_ms,
                retain_success_rate=retain_success_rate,
                delete_success_rate=delete_success_rate,
                consolidate_success_rate=consolidate_success_rate,
                retain_operations_per_stimulus=retain_operations_per_stimulus,
                delete_operations_per_stimulus=delete_operations_per_stimulus,
                consolidate_operations_per_stimulus=consolidate_operations_per_stimulus,
                retain_bytes_per_successful_retain=retain_bytes_per_successful_retain,
                avg_recall_duration_ms=avg_recall_duration_ms,
                p95_recall_duration_ms=p95_recall_duration_ms,
                avg_retain_duration_ms=avg_retain_duration_ms,
                avg_delete_duration_ms=avg_delete_duration_ms,
                avg_consolidate_duration_ms=avg_consolidate_duration_ms,
                avg_recall_results_count=avg_recall_results_count,
                avg_recall_output_bytes=avg_recall_output_bytes,
                total_recall_output_bytes=int(total_recall_output)
                if recall_output_bytes
                else None,
                avg_retain_content_bytes=avg_retain_content_bytes,
                total_retained_content_bytes=int(total_retained_content)
                if retain_content_bytes
                else None,
                recall_errors=recall_errors,
                retain_errors=retain_errors,
                delete_errors=delete_errors,
                consolidate_errors=consolidate_errors,
            )
        )

    return enhanced
