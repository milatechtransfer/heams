"""BEAM report helpers and rollup writer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from heams.benchmarks.beam.judge import question_type_from_audit_key
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

_BEAM_TYPE = "beam"

_BEAM_REPORT_SPEC = ReportSpec(
    summary_title="BEAM accuracy per memory backend",
    stratified_title="BEAM accuracy by ability",
    category_tag="question_type",
    categories=[],
    stratifiers=[
        ReportStratifier("question_type", "Ability"),
        ReportStratifier("token_scale", "Token scale"),
    ],
)


def _beam_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    records = run.audit.report_records
    if records:
        for record in records:
            yield {"passed": record.passed, "tags": dict(record.tags)}
        return
    for key in run.audit.passed_keys:
        yield {
            "passed": True,
            "tags": {"question_type": question_type_from_audit_key(key)},
        }
    for key in run.audit.failed_keys:
        yield {
            "passed": False,
            "tags": {"question_type": question_type_from_audit_key(key)},
        }


def render_beam_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    report = build_report(
        scenarios,
        results,
        records_of=_beam_report_records,
        spec=_BEAM_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    return render_report_sections(report, ordered_labels)


def render_beam_detail_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> list[str]:
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


def write_beam_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate BEAM experiment runs.jsonl into `_rollup/beam/report.md`."""

    def _render(
        scenarios: Sequence[Scenario],
        results: dict[str, list[RunResult]],
        memory_labels: Sequence[str],
    ) -> list[str]:
        sections = render_beam_metrics_summary(scenarios, results, memory_labels)
        detail = render_beam_detail_sections(scenarios, results)
        if sections and detail:
            return [*sections, "", *detail]
        return sections or detail

    return write_typed_rollup_report(
        results_dir,
        benchmark_type=_BEAM_TYPE,
        render_sections=_render,
        title="beam benchmark",
    )


__all__ = [
    "render_beam_detail_sections",
    "render_beam_metrics_summary",
    "write_beam_rollup_report",
]
