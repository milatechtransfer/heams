"""LOCOMO report helpers and CLI rollup writer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from heams.benchmarks.locomo.judge import locomo_category_label_from_key
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

_LOCOMO_REPORT_SPEC = ReportSpec(
    summary_title="LOCOMO accuracy per memory backend",
    stratified_title="LOCOMO accuracy by category",
    category_tag="category",
    categories=[],
    stratifiers=[ReportStratifier("category", "Category")],
)

_LOCOMO_TYPE = "locomo"


def _locomo_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    for key in run.audit.passed_keys:
        yield {
            "passed": True,
            "tags": {"category": locomo_category_label_from_key(key)},
        }
    for key in run.audit.failed_keys:
        yield {
            "passed": False,
            "tags": {"category": locomo_category_label_from_key(key)},
        }


def render_locomo_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    report = build_report(
        scenarios,
        results,
        records_of=_locomo_report_records,
        spec=_LOCOMO_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    return render_report_sections(report, ordered_labels)


def render_locomo_detail_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> list[str]:
    """Per-scenario QA / audit detail (same shape as legacy heams Detail)."""
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


def write_locomo_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate Locomo experiment runs.jsonl into `_rollup/locomo/report.md`."""

    def _render(
        scenarios: Sequence[Scenario],
        results: dict[str, list[RunResult]],
        memory_labels: Sequence[str],
    ) -> list[str]:
        sections = render_locomo_metrics_summary(scenarios, results, memory_labels)
        detail = render_locomo_detail_sections(scenarios, results)
        if sections and detail:
            return [*sections, "", *detail]
        return sections or detail

    return write_typed_rollup_report(
        results_dir,
        benchmark_type=_LOCOMO_TYPE,
        render_sections=_render,
        title="locomo benchmark",
    )
