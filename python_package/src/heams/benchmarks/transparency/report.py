"""Transparency report helpers and rollup hook."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from heams.evaluation.scenarios.transparency_llm_judge import (
    TRANSPARENCY_DIMENSION_REPORT_CATEGORY,
)
from heams.reporting.aggregation import memory_label, order_memory_labels
from heams.reporting.stratified import (
    ReportCategory,
    ReportSpec,
    ReportStratifier,
    ReportTally,
    StratifiedReport,
    build_report,
)
from heams.reporting.tables import (
    format_metric_cell,
    render_comparison_table,
    render_stratified_table,
)
from heams.results import RunResult
from heams.scenarios.base import Scenario

TRANSPARENCY_REPORT_CATEGORIES: list[tuple[str, str, str]] = [
    ("required_completeness", "Required/completeness pass rate", "rate"),
    ("provenance_metadata", "Provenance/metadata pass rate", "rate"),
    ("over_disclosure", "Over-disclosure failures", "failures"),
    ("hallucination_negative", "Hallucination/negative-audit failures", "failures"),
]
TRANSPARENCY_STRATIFIERS: list[tuple[str, str]] = [
    ("privacy_domain", "Privacy domain"),
    ("scenario_type", "Scenario type"),
    ("persona_id", "Persona"),
    ("difficulty_level", "Difficulty"),
]
_TRANSPARENCY_REPORT_SPEC = ReportSpec(
    summary_title="Transparency metrics per memory backend",
    stratified_title="Transparency metrics by stratum",
    category_tag="category",
    categories=[
        ReportCategory(category, label, kind)  # type: ignore[arg-type]
        for category, label, kind in TRANSPARENCY_REPORT_CATEGORIES
    ],
    stratifiers=[
        ReportStratifier(stratifier, label)
        for stratifier, label in TRANSPARENCY_STRATIFIERS
    ],
)
_TRANSPARENCY_STRATIFIED_CATEGORY_HEADERS = {
    "required_completeness": "Completeness",
    "provenance_metadata": "Provenance",
    "over_disclosure": "Over-disclosure",
    "hallucination_negative": "Hallucination",
}


def build_transparency_report(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> StratifiedReport:
    return build_report(
        scenarios,
        results,
        records_of=_transparency_report_records,
        spec=_TRANSPARENCY_REPORT_SPEC,
    )


def render_transparency_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    report = build_transparency_report(scenarios, results)
    if not report.by_memory:
        return []

    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []

    lines = render_transparency_summary(report, ordered_labels)
    stratified_lines = render_transparency_stratified_summary(report, ordered_labels)
    if stratified_lines:
        lines.append("")
        lines.extend(stratified_lines)
    return lines


def render_transparency_summary(
    report: StratifiedReport,
    memory_labels: Sequence[str],
) -> list[str]:
    rows: list[tuple[str, list[str]]] = [
        ("Runs", [str(report.by_memory[label].runs) for label in memory_labels]),
        (
            "Criteria passed",
            [str(report.by_memory[label].records_passed) for label in memory_labels],
        ),
        (
            "Criteria total",
            [str(report.by_memory[label].records_total) for label in memory_labels],
        ),
        (
            "Micro accuracy",
            [
                format_metric_cell(report.by_memory[label].micro_accuracy, precision=4)
                for label in memory_labels
            ],
        ),
        (
            "Macro accuracy (scenario type)",
            [
                format_metric_cell(
                    report.macro_accuracy(label, "scenario_type"), precision=4
                )
                for label in memory_labels
            ],
        ),
        (
            "Macro accuracy (privacy domain)",
            [
                format_metric_cell(
                    report.macro_accuracy(label, "privacy_domain"), precision=4
                )
                for label in memory_labels
            ],
        ),
    ]
    for category, label, kind in TRANSPARENCY_REPORT_CATEGORIES:
        rows.append(
            (
                label,
                [
                    _transparency_category_cell(
                        report.by_memory[memory], category, kind
                    )
                    for memory in memory_labels
                ],
            )
        )
    return render_comparison_table(
        "Transparency metrics per memory backend",
        memory_labels,
        rows,
    )


def render_transparency_stratified_summary(
    report: StratifiedReport,
    memory_labels: Sequence[str],
) -> list[str]:
    category_headers = [
        _TRANSPARENCY_STRATIFIED_CATEGORY_HEADERS[category]
        for category, _, _ in TRANSPARENCY_REPORT_CATEGORIES
    ]
    headers = [
        "Stratifier",
        "Value",
        "Memory",
        "Runs",
        "Criteria",
        "Micro accuracy",
        *category_headers,
    ]
    rows: list[list[str]] = []
    for stratifier, stratifier_label in TRANSPARENCY_STRATIFIERS:
        values: set[str] = set()
        for memory in memory_labels:
            values.update(
                report.by_memory_stratifier.get(memory, {}).get(stratifier, {}).keys()
            )
        for value in sorted(values):
            for memory in memory_labels:
                tally = (
                    report.by_memory_stratifier.get(memory, {})
                    .get(stratifier, {})
                    .get(value)
                )
                if tally is None or tally.runs == 0:
                    continue
                category_cells = [
                    _transparency_category_cell(tally, category, kind)
                    for category, _, kind in TRANSPARENCY_REPORT_CATEGORIES
                ]
                rows.append(
                    [
                        stratifier_label,
                        value,
                        memory,
                        str(tally.runs),
                        f"{tally.records_passed}/{tally.records_total}",
                        format_metric_cell(tally.micro_accuracy, precision=4),
                        *category_cells,
                    ]
                )
    return render_stratified_table(
        "Transparency metrics by stratum",
        headers,
        rows,
        numeric_from=3,
    )


def render_transparency_detail_sections(
    scenario: Scenario,
    sample_runs: Sequence[RunResult],
) -> list[str]:
    rows: list[list[str]] = []
    for run in sorted(sample_runs, key=lambda current: current.memory_index):
        for record in run.audit.report_records or []:
            rows.append(
                [
                    memory_label(scenario, run.memory_index),
                    record.key or "-",
                    str(record.passed),
                    format_metric_cell(record.score, precision=4),
                    record.tags.get("dimension", "-"),
                    _compact_cell(record.expected),
                    _compact_cell(record.observed),
                    _compact_cell(record.reason),
                ]
            )
    if not rows:
        return []
    return render_stratified_table(
        "Transparency criteria",
        [
            "Memory",
            "Criterion",
            "Passed",
            "Score",
            "Dimension",
            "Expected",
            "Observed",
            "Reason",
        ],
        rows,
        numeric_from=3,
    )


def _transparency_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    if run.audit.report_records:
        for record in run.audit.report_records:
            tags = dict(record.tags)
            category = TRANSPARENCY_DIMENSION_REPORT_CATEGORY.get(
                tags.get("dimension", "")
            )
            if category is not None:
                tags["category"] = category
            yield {"passed": record.passed, "tags": tags}
        return

    for _ in run.audit.passed_keys:
        yield {"passed": True, "tags": {}}
    for _ in run.audit.failed_keys:
        yield {"passed": False, "tags": {}}


def _transparency_category_cell(tally: ReportTally, category: str, kind: str) -> str:
    if kind == "failures":
        return str(tally.category_failures(category))
    return format_metric_cell(tally.category_rate(category), precision=4)


def _compact_cell(value: str | None) -> str:
    if not value:
        return "-"
    return " ".join(value.split()).replace("|", "\\|")


def render_transparency_rollup_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    """Summary + stratified + per-scenario detail for artifact rollup."""
    from collections import defaultdict

    from heams.reporting.benchmark_report import (
        render_run_outcome_comparison,
        render_run_trace_details,
    )

    sections = render_transparency_metrics_summary(scenarios, results, memory_labels)
    if not results:
        return sections

    by_id = {scenario.id: scenario for scenario in scenarios}
    detail: list[str] = ["## Detail", ""]
    for scenario_id in sorted(results):
        scenario = by_id.get(scenario_id)
        if scenario is None:
            continue
        runs_by_sample: dict[int, list[RunResult]] = defaultdict(list)
        for run in results[scenario_id]:
            runs_by_sample[run.sample_index].append(run)
        for sample_index in sorted(runs_by_sample):
            sample_runs = runs_by_sample[sample_index]
            detail.append(f"### {scenario_id} / sample {sample_index}")
            detail.append("")
            detail.extend(render_run_outcome_comparison(scenario, sample_runs))
            detail.append("")
            detail.extend(render_transparency_detail_sections(scenario, sample_runs))
            detail.append("")
            detail.extend(render_run_trace_details(scenario, sample_runs))
            detail.append("")
    if len(detail) > 2:
        if sections:
            sections = [*sections, ""]
        sections.extend(detail)
    return sections


__all__ = [
    "TRANSPARENCY_REPORT_CATEGORIES",
    "TRANSPARENCY_STRATIFIERS",
    "build_transparency_report",
    "render_transparency_detail_sections",
    "render_transparency_metrics_summary",
    "render_transparency_rollup_sections",
    "render_transparency_stratified_summary",
    "render_transparency_summary",
]
