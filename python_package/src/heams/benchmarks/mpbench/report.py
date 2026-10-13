"""MPBench report helpers and rollup writer."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from heams.benchmarks.mpbench.dataset import ATTACK_CLASSES
from heams.benchmarks.mpbench.judge import attack_class_from_audit_key
from heams.reporting.aggregation import order_memory_labels
from heams.reporting.benchmark_report import (
    render_run_outcome_comparison,
    render_run_trace_details,
)
from heams.reporting.benchmark_rollup import write_typed_rollup_report
from heams.reporting.stratified import (
    ReportCategory,
    ReportSpec,
    ReportStratifier,
    build_report,
    render_report_sections,
)
from heams.results import RunResult
from heams.scenarios.base import Scenario

_MPBENCH_TYPE = "mpbench"

_MPBENCH_REPORT_SPEC = ReportSpec(
    summary_title="MPBench resistance per memory backend",
    stratified_title="MPBench resistance by attack class",
    category_tag="attack_class",
    categories=[ReportCategory(key=name, label=name) for name in ATTACK_CLASSES],
    stratifiers=[ReportStratifier("attack_class", "Attack class")],
)


def _mpbench_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    for key in run.audit.passed_keys:
        yield {
            "passed": True,
            "tags": {"attack_class": attack_class_from_audit_key(key)},
        }
    for key in run.audit.failed_keys:
        yield {
            "passed": False,
            "tags": {"attack_class": attack_class_from_audit_key(key)},
        }


def _macro_and_worst_section(
    results: dict[str, list[RunResult]],
) -> list[str]:
    counts: dict[str, list[bool]] = defaultdict(list)
    for runs in results.values():
        for run in runs:
            for key in run.audit.passed_keys:
                counts[attack_class_from_audit_key(key)].append(True)
            for key in run.audit.failed_keys:
                counts[attack_class_from_audit_key(key)].append(False)
    if not counts:
        return []
    lines = ["## Macro resistance", ""]
    class_scores: list[tuple[str, float, int]] = []
    for attack_class in ATTACK_CLASSES:
        outcomes = counts.get(attack_class) or []
        if not outcomes:
            continue
        resistance = sum(1 for ok in outcomes if ok) / len(outcomes)
        class_scores.append((attack_class, resistance, len(outcomes)))
        lines.append(
            f"- `{attack_class}`: {resistance:.3f} resistance "
            f"({sum(1 for ok in outcomes if ok)}/{len(outcomes)} resisted)"
        )
    if class_scores:
        macro = sum(score for _, score, _ in class_scores) / len(class_scores)
        worst_class, worst_score, _ = min(class_scores, key=lambda item: item[1])
        lines.extend(
            [
                "",
                f"- **macro-average**: {macro:.3f}",
                f"- **worst class**: `{worst_class}` ({worst_score:.3f})",
                "",
            ]
        )
    return lines


def render_mpbench_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    report = build_report(
        scenarios,
        results,
        records_of=_mpbench_report_records,
        spec=_MPBENCH_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    sections = render_report_sections(report, ordered_labels)
    macro = _macro_and_worst_section(results)
    if sections and macro:
        return [*sections, "", *macro]
    return sections or macro


def render_mpbench_detail_sections(
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


def write_mpbench_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate MPBench experiment runs.jsonl into `_rollup/mpbench/report.md`."""

    def _render(
        scenarios: Sequence[Scenario],
        results: dict[str, list[RunResult]],
        memory_labels: Sequence[str],
    ) -> list[str]:
        sections = render_mpbench_metrics_summary(scenarios, results, memory_labels)
        detail = render_mpbench_detail_sections(scenarios, results)
        if sections and detail:
            return [*sections, "", *detail]
        return sections or detail

    return write_typed_rollup_report(
        results_dir,
        benchmark_type=_MPBENCH_TYPE,
        render_sections=_render,
        title="mpbench benchmark",
    )


__all__ = [
    "render_mpbench_detail_sections",
    "render_mpbench_metrics_summary",
    "write_mpbench_rollup_report",
]
