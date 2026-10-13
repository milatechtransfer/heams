"""Generic pass-rate reports over tagged evaluation records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from heams.reporting.aggregation import RateAccumulator, iter_labeled_runs
from heams.reporting.tables import format_metric_cell, render_comparison_table
from heams.reporting.tables import render_stratified_table
from heams.results import RunResult
from heams.scenarios.base import Scenario

CategoryKind = Literal["rate", "failures"]


@dataclass(frozen=True)
class ReportCategory:
    """A report column derived from one tag value."""

    key: str
    label: str
    kind: CategoryKind = "rate"


@dataclass(frozen=True)
class ReportStratifier:
    """A tag rendered as a grouped breakdown."""

    tag: str
    label: str


@dataclass(frozen=True)
class ReportSpec:
    """Declarative shape for a tagged pass-rate report."""

    summary_title: str
    stratified_title: str
    category_tag: str
    categories: Sequence[ReportCategory]
    stratifiers: Sequence[ReportStratifier] = ()


@dataclass
class RunRecordSummary:
    """Per-run pass/fail counts split by category and tags."""

    overall: RateAccumulator = field(default_factory=RateAccumulator)
    categories: dict[str, RateAccumulator] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    strata: dict[str, dict[str, "RunRecordSummary"]] = field(default_factory=dict)

    def record(self, *, category: str | None, passed: bool) -> None:
        self.overall.add(1.0 if passed else 0.0, 1.0)
        if category is not None:
            self.categories.setdefault(category, RateAccumulator()).add(
                1.0 if passed else 0.0, 1.0
            )


@dataclass
class ReportTally:
    """Accumulated pass/fail counts for one memory backend or stratum."""

    runs: int = 0
    overall: RateAccumulator = field(default_factory=RateAccumulator)
    categories: dict[str, RateAccumulator] = field(
        default_factory=lambda: defaultdict(RateAccumulator)
    )

    def add_run(self, summary: RunRecordSummary) -> None:
        self.runs += 1
        self.overall.merge(summary.overall)
        for category, accumulator in summary.categories.items():
            self.categories[category].merge(accumulator)

    @property
    def records_passed(self) -> int:
        return int(self.overall.passed)

    @property
    def records_total(self) -> int:
        return int(self.overall.total)

    @property
    def micro_accuracy(self) -> float | None:
        return self.overall.rate

    def category_rate(self, category: str) -> float | None:
        accumulator = self.categories.get(category)
        return accumulator.rate if accumulator is not None else None

    def category_failures(self, category: str) -> int:
        accumulator = self.categories.get(category)
        return int(accumulator.failures) if accumulator is not None else 0


@dataclass
class StratifiedReport:
    """Aggregated tagged-record report."""

    spec: ReportSpec
    by_memory: dict[str, ReportTally]
    by_memory_stratifier: dict[str, dict[str, dict[str, ReportTally]]]

    def memory_labels(self) -> list[str]:
        return list(self.by_memory.keys())

    def macro_accuracy(self, memory_label: str, stratifier: str) -> float | None:
        strata = self.by_memory_stratifier.get(memory_label, {}).get(stratifier, {})
        rates = [
            tally.micro_accuracy
            for tally in strata.values()
            if tally.micro_accuracy is not None
        ]
        if not rates:
            return None
        return sum(rates) / len(rates)


def build_report(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    *,
    records_of: Callable[[Scenario, RunResult], Iterable[object]],
    spec: ReportSpec,
) -> StratifiedReport:
    """Build a pass-rate report from benchmark-provided tagged records."""

    by_memory: dict[str, ReportTally] = defaultdict(ReportTally)
    by_memory_stratifier: dict[str, dict[str, dict[str, ReportTally]]] = defaultdict(
        lambda: {
            stratifier.tag: defaultdict(ReportTally) for stratifier in spec.stratifiers
        }
    )

    for label, scenario, run in iter_labeled_runs(scenarios, results):
        summary = summarize_run_records(records_of(scenario, run), spec)
        if summary is None:
            continue

        by_memory[label].add_run(summary)
        for stratifier in spec.stratifiers:
            for value, stratum_summary in summary.strata.get(
                stratifier.tag, {}
            ).items():
                by_memory_stratifier[label][stratifier.tag][value].add_run(
                    stratum_summary
                )

    return StratifiedReport(
        spec=spec,
        by_memory=dict(by_memory),
        by_memory_stratifier={
            label: {stratifier: dict(values) for stratifier, values in strata.items()}
            for label, strata in by_memory_stratifier.items()
        },
    )


def summarize_run_records(
    records: Iterable[object],
    spec: ReportSpec,
) -> RunRecordSummary | None:
    """Summarize one run's records without knowing their concrete type."""

    summary = RunRecordSummary()
    for record in records:
        tags = _record_tags(record)
        category = tags.get(spec.category_tag)
        passed = _record_passed(record)
        summary.tags.update(tags)
        summary.record(
            category=category,
            passed=passed,
        )
        for stratifier in spec.stratifiers:
            value = tags.get(stratifier.tag, "unspecified")
            stratum_summary = summary.strata.setdefault(stratifier.tag, {}).setdefault(
                value, RunRecordSummary()
            )
            stratum_summary.record(category=category, passed=passed)
            stratum_summary.tags.update(tags)
    if summary.overall.total <= 0:
        return None
    return summary


def render_report_sections(
    report: StratifiedReport,
    memory_labels: Sequence[str] | None = None,
) -> list[str]:
    """Render summary and stratified sections for a built report."""

    labels = (
        list(memory_labels) if memory_labels is not None else report.memory_labels()
    )
    lines = render_summary_section(report, labels)
    stratified_lines = render_stratified_section(report, labels)
    if lines and stratified_lines:
        lines.append("")
    lines.extend(stratified_lines)
    return lines


def render_summary_section(
    report: StratifiedReport,
    memory_labels: Sequence[str] | None = None,
) -> list[str]:
    labels = (
        list(memory_labels) if memory_labels is not None else report.memory_labels()
    )
    if not labels:
        return []

    rows: list[tuple[str, list[str]]] = [
        (
            "Micro accuracy",
            [
                format_metric_cell(
                    report.by_memory.get(label).micro_accuracy, precision=4
                )
                if label in report.by_memory
                else "na"
                for label in labels
            ],
        )
    ]
    for stratifier in report.spec.stratifiers:
        rows.append(
            (
                f"Macro accuracy ({stratifier.label.lower()})",
                [
                    format_metric_cell(
                        report.macro_accuracy(label, stratifier.tag), precision=4
                    )
                    for label in labels
                ],
            )
        )
    for category in report.spec.categories:
        rows.append(
            (
                category.label,
                [
                    _category_cell(report.by_memory.get(label), category)
                    for label in labels
                ],
            )
        )
    return render_comparison_table(report.spec.summary_title, labels, rows)


def render_stratified_section(
    report: StratifiedReport,
    memory_labels: Sequence[str] | None = None,
) -> list[str]:
    labels = (
        list(memory_labels) if memory_labels is not None else report.memory_labels()
    )
    rows: list[list[str]] = []
    headers = ["Stratifier", "Value", "Memory", "Runs", "Records", "Micro accuracy"]
    headers.extend(category.label for category in report.spec.categories)

    for stratifier in report.spec.stratifiers:
        for memory_label in labels:
            values = report.by_memory_stratifier.get(memory_label, {}).get(
                stratifier.tag, {}
            )
            for value in sorted(values):
                tally = values[value]
                rows.append(
                    [
                        stratifier.label,
                        value,
                        memory_label,
                        str(tally.runs),
                        f"{tally.records_passed}/{tally.records_total}",
                        format_metric_cell(tally.micro_accuracy, precision=4),
                        *[
                            _category_cell(tally, category)
                            for category in report.spec.categories
                        ],
                    ]
                )

    return render_stratified_table(
        report.spec.stratified_title,
        headers,
        rows,
        numeric_from=3,
    )


def _category_cell(tally: ReportTally | None, category: ReportCategory) -> str:
    if tally is None:
        return "na" if category.kind == "rate" else "0"
    if category.kind == "failures":
        return str(tally.category_failures(category.key))
    return format_metric_cell(tally.category_rate(category.key), precision=4)


def _record_passed(record: object) -> bool:
    if isinstance(record, Mapping):
        return bool(record["passed"])
    return bool(getattr(record, "passed"))


def _record_tags(record: object) -> dict[str, str]:
    raw_tags: Mapping[str, Any] | None
    if isinstance(record, Mapping):
        raw_tags = record.get("tags", {})  # type: ignore[assignment]
    else:
        raw_tags = getattr(record, "tags", {})
    if raw_tags is None:
        return {}
    return {
        str(key): str(value)
        for key, value in raw_tags.items()
        if value is not None and str(value) != ""
    }


__all__ = [
    "ReportCategory",
    "ReportSpec",
    "ReportStratifier",
    "ReportTally",
    "RunRecordSummary",
    "StratifiedReport",
    "build_report",
    "render_report_sections",
    "render_stratified_section",
    "render_summary_section",
    "summarize_run_records",
]
