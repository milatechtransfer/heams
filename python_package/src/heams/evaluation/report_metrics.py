from __future__ import annotations

from typing import Any, Callable, TypeAlias


MetricSpec: TypeAlias = tuple[str, Callable[[Any], object], int | None]


# Trace rows belong on the experiment report and detailed CSV. Benchmark
# rollups keep the pre-trace list so those tables stay unchanged.
_EXPERIMENT_TRACE_METRIC_NAMES = frozenset(
    {
        "LLM input tokens",
        "LLM output tokens",
        "LLM duration (ms)",
        "Agent duration (ms)",
        "LLM input cost",
        "LLM output cost",
        "Tool executions",
        "Tool execution errors",
        "Tool duration (ms)",
    }
)


EXPERIMENT_SYSTEM_REPORT_METRICS: list[MetricSpec] = [
    (
        "Run duration (ms)",
        lambda summary: summary.run_duration_ms,
        2,
    ),
    ("Stimuli", lambda summary: summary.stimulus_count, None),
    (
        "Agent turns",
        lambda summary: summary.agent_turn_count,
        None,
    ),
    (
        "Memory operations",
        lambda summary: summary.memory_operation_count,
        None,
    ),
    (
        "Memory operation errors",
        lambda summary: summary.memory_operation_error_count,
        None,
    ),
    (
        "Memory operation skipped",
        lambda summary: summary.memory_operation_skipped_count,
        None,
    ),
    (
        "LLM calls",
        lambda summary: summary.llm_call_count,
        None,
    ),
    (
        "LLM input tokens",
        lambda summary: summary.llm_input_tokens,
        None,
    ),
    (
        "LLM output tokens",
        lambda summary: summary.llm_output_tokens,
        None,
    ),
    (
        "LLM total tokens",
        lambda summary: summary.llm_total_tokens,
        None,
    ),
    (
        "LLM duration (ms)",
        lambda summary: summary.llm_duration_ms,
        2,
    ),
    (
        "Agent duration (ms)",
        lambda summary: summary.agent_duration_ms,
        2,
    ),
    (
        "LLM input cost",
        lambda summary: summary.llm_input_cost,
        6,
    ),
    (
        "LLM output cost",
        lambda summary: summary.llm_output_cost,
        6,
    ),
    (
        "LLM total cost",
        lambda summary: summary.llm_total_cost,
        6,
    ),
    (
        "Tool executions",
        lambda summary: summary.tool_execution_count,
        None,
    ),
    (
        "Tool execution errors",
        lambda summary: summary.tool_execution_error_count,
        None,
    ),
    (
        "Tool duration (ms)",
        lambda summary: summary.tool_duration_ms,
        2,
    ),
    (
        "Memory calls per stimulus",
        lambda summary: summary.memory_calls_per_stimulus,
        2,
    ),
    (
        "Task accuracy per memory call",
        lambda summary: summary.task_accuracy_per_memory_call,
        4,
    ),
    (
        "Task success per recall call",
        lambda summary: summary.task_success_per_recall_call,
        4,
    ),
]


SYSTEM_REPORT_METRICS: list[MetricSpec] = [
    metric
    for metric in EXPERIMENT_SYSTEM_REPORT_METRICS
    if metric[0] not in _EXPERIMENT_TRACE_METRIC_NAMES
]


COMPONENT_REPORT_METRICS: list[MetricSpec] = [
    (
        "Total operations",
        lambda summary: summary.total_operations,
        None,
    ),
    (
        "Successful operations",
        lambda summary: summary.successful_operations,
        None,
    ),
    (
        "Failed operations",
        lambda summary: summary.failed_operations,
        None,
    ),
    (
        "Skipped operations",
        lambda summary: summary.skipped_operations,
        None,
    ),
    (
        "Recall operations",
        lambda summary: summary.operation_counts.get("recall", 0),
        None,
    ),
    (
        "Retain operations",
        lambda summary: summary.operation_counts.get("retain", 0),
        None,
    ),
    (
        "Delete operations",
        lambda summary: summary.operation_counts.get("delete", 0),
        None,
    ),
    (
        "Consolidate operations",
        lambda summary: summary.operation_counts.get("consolidate", 0),
        None,
    ),
    (
        "Retain success rate",
        lambda summary: summary.retain_success_rate,
        4,
    ),
    (
        "Delete success rate",
        lambda summary: summary.delete_success_rate,
        4,
    ),
    (
        "Consolidate success rate",
        lambda summary: summary.consolidate_success_rate,
        4,
    ),
    (
        "Retain operations per stimulus",
        lambda summary: summary.retain_operations_per_stimulus,
        2,
    ),
    (
        "Delete operations per stimulus",
        lambda summary: summary.delete_operations_per_stimulus,
        2,
    ),
    (
        "Consolidate operations per stimulus",
        lambda summary: summary.consolidate_operations_per_stimulus,
        2,
    ),
    (
        "Retain bytes per successful retain",
        lambda summary: summary.retain_bytes_per_successful_retain,
        0,
    ),
    (
        "Avg recall duration (ms)",
        lambda summary: summary.avg_recall_duration_ms,
        2,
    ),
    (
        "P95 recall duration (ms)",
        lambda summary: summary.p95_recall_duration_ms,
        2,
    ),
    (
        "Avg retain duration (ms)",
        lambda summary: summary.avg_retain_duration_ms,
        2,
    ),
    (
        "Avg delete duration (ms)",
        lambda summary: summary.avg_delete_duration_ms,
        2,
    ),
    (
        "Avg consolidate duration (ms)",
        lambda summary: summary.avg_consolidate_duration_ms,
        2,
    ),
    (
        "Avg recall results per call",
        lambda summary: summary.avg_recall_results_count,
        1,
    ),
    (
        "Avg recall output (bytes)",
        lambda summary: summary.avg_recall_output_bytes,
        0,
    ),
    (
        "Total recall output (bytes)",
        lambda summary: summary.total_recall_output_bytes,
        None,
    ),
    (
        "Avg retain content (bytes)",
        lambda summary: summary.avg_retain_content_bytes,
        0,
    ),
    (
        "Total retained content (bytes)",
        lambda summary: summary.total_retained_content_bytes,
        None,
    ),
    (
        "Recall errors",
        lambda summary: summary.recall_errors,
        None,
    ),
    (
        "Retain errors",
        lambda summary: summary.retain_errors,
        None,
    ),
    (
        "Delete errors",
        lambda summary: summary.delete_errors,
        None,
    ),
    (
        "Consolidate errors",
        lambda summary: summary.consolidate_errors,
        None,
    ),
]


COMPONENT_PERFORMANCE_REPORT_METRICS: list[MetricSpec] = [
    (
        "Recall eval count",
        lambda summary: summary.recall_eval_count,
        None,
    ),
    (
        "Recall Hit@k",
        lambda summary: summary.recall_hit_at_k,
        4,
    ),
    (
        "Recall MRR",
        lambda summary: summary.recall_mrr,
        4,
    ),
]


__all__ = [
    "MetricSpec",
    "EXPERIMENT_SYSTEM_REPORT_METRICS",
    "SYSTEM_REPORT_METRICS",
    "COMPONENT_REPORT_METRICS",
    "COMPONENT_PERFORMANCE_REPORT_METRICS",
]
