"""LongMemEval-specific summary report rendering."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from heams.evaluation.scenarios.long_mem_eval_llm_judge import (
    long_mem_eval_tags_from_key,
)
from heams.reporting.aggregation import order_memory_labels
from heams.reporting.stratified import (
    ReportSpec,
    ReportStratifier,
    build_report,
    render_report_sections,
)
from heams.results import RunResult
from heams.scenarios.base import Scenario

_LONG_MEM_EVAL_REPORT_SPEC = ReportSpec(
    summary_title="LongMemEval accuracy per memory backend",
    stratified_title="LongMemEval accuracy by stratum",
    category_tag="question_type",
    categories=[],
    stratifiers=[
        ReportStratifier("question_type", "Question type"),
        ReportStratifier("abstention", "Abstention"),
    ],
)


def _report_records(_scenario: Scenario, run: RunResult) -> Iterator[dict[str, object]]:
    for key in run.audit.passed_keys:
        yield {"passed": True, "tags": long_mem_eval_tags_from_key(key)}
    for key in run.audit.failed_keys:
        yield {"passed": False, "tags": long_mem_eval_tags_from_key(key)}


def render_long_mem_eval_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    """Render LongMemEval accuracy summaries stratified by task category."""
    report = build_report(
        scenarios,
        results,
        records_of=_report_records,
        spec=_LONG_MEM_EVAL_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    return render_report_sections(report, ordered_labels)
