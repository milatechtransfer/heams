"""STALE report helpers and rollup writer.

Aggregates each experiment's ``runs.jsonl`` into per-dimension accuracy
(dim1/dim2/dim3) + overall, optionally split by conflict type (T1/T2), using the
shared reporting engine in ``heams.reporting``. STALE's evaluator already tags each
``EvaluationRecord`` with ``dimension`` / ``dimension_label`` / ``conflict_type``
(see ``stale/judge.py``), so reporting just reads those tagged records — the same
shape ``transparency/report.py`` uses.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from heams.reporting.aggregation import order_memory_labels
from heams.reporting.benchmark_report import (
    render_run_outcome_comparison,
    render_run_trace_details,
)
from heams.reporting.benchmark_rollup import write_typed_rollup_report
from heams.reporting.stratified import (
    ReportSpec,
    ReportStratifier,
    build_report,
    render_report_sections,
)
from heams.results import RunResult
from heams.scenarios.base import Scenario

_STALE_TYPE = "stale"

_STALE_REPORT_SPEC = ReportSpec(
    summary_title="STALE accuracy per memory backend",
    stratified_title="STALE accuracy by dimension",
    category_tag="dimension_label",
    categories=[],
    stratifiers=[
        ReportStratifier("dimension_label", "Dimension"),
        ReportStratifier("conflict_type", "Conflict type"),
    ],
)


def _stale_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    """Yield ``{passed, tags}`` per graded dimension from the audit records.

    Each STALE ``EvaluationRecord`` already carries ``dimension`` /
    ``dimension_label`` / ``conflict_type`` tags, so the report reads them directly.
    """
    for record in run.audit.report_records or []:
        yield {"passed": record.passed, "tags": dict(record.tags)}


def render_stale_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    report = build_report(
        scenarios,
        results,
        records_of=_stale_report_records,
        spec=_STALE_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    return render_report_sections(report, ordered_labels)


def render_stale_detail_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> list[str]:
    """Per-scenario probe outcomes + traces (same shape as the heams Detail)."""
    if not results:
        return []

    by_id = {scenario.id: scenario for scenario in scenarios}
    lines: list[str] = ["## Detail", ""]
    for scenario_id in sorted(results):
        scenario = by_id.get(scenario_id)
        if scenario is None:
            continue
        runs_by_sample: dict[int, list[RunResult]] = defaultdict(list)
        for run in results[scenario_id]:
            runs_by_sample[run.sample_index].append(run)

        for sample_index in sorted(runs_by_sample):
            sample_runs = runs_by_sample[sample_index]
            lines.append(f"### {scenario_id} / sample {sample_index}")
            lines.append("")
            lines.extend(render_run_outcome_comparison(scenario, sample_runs))
            lines.append("")
            lines.extend(render_run_trace_details(scenario, sample_runs))
            lines.append("")
    return lines


def render_stale_rollup_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    sections = render_stale_metrics_summary(scenarios, results, memory_labels)
    detail = render_stale_detail_sections(scenarios, results)
    if sections and detail:
        return [*sections, "", *detail]
    return sections or detail


def write_stale_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate STALE experiment runs.jsonl into `_rollup/stale/report.md`."""
    return write_typed_rollup_report(
        results_dir,
        benchmark_type=_STALE_TYPE,
        render_sections=render_stale_rollup_sections,
        title="stale benchmark",
    )
