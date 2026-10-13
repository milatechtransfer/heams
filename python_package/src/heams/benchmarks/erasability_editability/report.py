"""EE report helpers and rollup writer.

Two tables, in this order:

1. **Trivial-pass filtered score** - the headline. A probe counts as correct only
   if the system also answered the matching before-probe correctly, which is the
   scoring rule stated on the published dataset card. The four buckets
   (``real`` / ``trivial`` / ``knew_but_failed`` / ``never_knew``) are shown
   beside it so an inflated deletion score is visible rather than hidden: a
   system that never encoded a fact "forgets" it for free, and lands in
   ``trivial``.
2. **Raw judge pass rate** - the ungated per-probe verdicts, stratified by task
   type, phase, cascade hop and domain. Diagnostic only; it does not apply the
   trivial-pass filter and will read higher than the headline.

Both are artifact-only: they read the tags the EE evaluator already stamped onto
each ``EvaluationRecord`` (see ``erasability_editability/judge.py``) and never
re-materialize the benchmark.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path

from heams.benchmarks.erasability_editability.judge import (
    EE_PASS_TYPE_LABELS,
    EE_PASS_TYPES,
)
from heams.reporting.aggregation import iter_labeled_runs, order_memory_labels
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

_EE_TYPE = "ee"

_EE_REPORT_SPEC = ReportSpec(
    summary_title="EE raw judge pass rate per memory backend",
    stratified_title="EE raw judge pass rate by task type and phase",
    category_tag="task_phase",
    categories=[],
    stratifiers=[
        ReportStratifier("task_phase", "Task / phase"),
        ReportStratifier("task_type", "Task type"),
        ReportStratifier("phase", "Phase"),
        ReportStratifier("hop", "Cascade hop"),
        ReportStratifier("domain", "Domain"),
    ],
)


def _ee_report_records(
    _scenario: Scenario, run: RunResult
) -> Iterator[dict[str, object]]:
    """Yield ``{passed, tags}`` per graded probe from the audit records."""
    for record in run.audit.report_records or []:
        yield {"passed": record.passed, "tags": dict(record.tags)}


def render_ee_trivial_pass_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
) -> list[str]:
    """Headline table: trivial-pass filtered score plus the four buckets.

    Only after-probes carry a ``pass_type`` tag, so this counts exactly the
    probes the dataset card's rule applies to.
    """
    if not results:
        return []

    # (memory_label, task_type) -> {pass_type: count}
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(EE_PASS_TYPES, 0)
    )
    for label, _scenario, run in iter_labeled_runs(scenarios, results):
        for record in run.audit.report_records or []:
            pass_type = record.tags.get("pass_type")
            if pass_type not in EE_PASS_TYPES:
                continue  # before-phase probe, or an older artifact
            task_type = record.tags.get("task_type", "unknown")
            totals[(label, task_type)][pass_type] += 1

    if not totals:
        return []

    lines = [
        "## EE trivial-pass filtered score",
        "",
        (
            "A probe counts as correct only when the matching pre-change probe for "
            "the same entity also passed (the published dataset card's rule). "
            "`trivial` counts probes that passed only because the system never "
            "encoded the fact — a system that knows nothing forgets everything for "
            "free, so these must not be scored as successes."
        ),
        "",
    ]

    # Headline: one row per backend, one column per task type.
    task_types = sorted({task_type for _label, task_type in totals})
    columns = [f"{task_type}_accuracy" for task_type in task_types]
    lines.append("| Memory backend | " + " | ".join(columns) + " |")
    lines.append("| --- |" + " ---: |" * len(columns))
    for label in sorted({label for label, _task_type in totals}):
        cells = []
        for task_type in task_types:
            counts = totals.get((label, task_type))
            if not counts or not sum(counts.values()):
                cells.append("n/a")
                continue
            total = sum(counts.values())
            real = counts["real"]
            cells.append(f"{real / total:.1%} ({real}/{total})")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    # Bucket detail, one row per (backend, task type).
    lines.extend(
        [
            "",
            "| Memory backend | Task | Score | " + " | ".join(EE_PASS_TYPES) + " |",
            "| --- | --- | ---: |" + " ---: |" * len(EE_PASS_TYPES),
        ]
    )
    for (label, task_type), counts in sorted(totals.items()):
        total = sum(counts.values())
        real = counts["real"]
        score = f"{real / total:.1%} ({real}/{total})" if total else "n/a"
        cells = " | ".join(str(counts[name]) for name in EE_PASS_TYPES)
        lines.append(f"| {label} | {task_type} | {score} | {cells} |")

    # Legend from the same mapping the tags and audit notes use, so the wording
    # cannot drift between the report and the raw records.
    lines.extend(
        ["", "| Bucket | Meaning | Counts toward score |", "| --- | --- | --- |"]
    )
    for name in EE_PASS_TYPES:
        counted = "yes" if name == "real" else "no"
        lines.append(f"| `{name}` | {EE_PASS_TYPE_LABELS[name]} | {counted} |")
    lines.append("")
    return lines


def render_ee_metrics_summary(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    """Raw per-probe judge pass rate, before the trivial-pass filter."""
    report = build_report(
        scenarios,
        results,
        records_of=_ee_report_records,
        spec=_EE_REPORT_SPEC,
    )
    if not report.by_memory:
        return []
    ordered_labels = order_memory_labels(memory_labels, report.memory_labels())
    if not ordered_labels:
        return []
    sections = render_report_sections(report, ordered_labels)
    return [
        (
            "_Raw judge verdicts, without the trivial-pass filter. Use the "
            "trivial-pass table above as the headline score._"
        ),
        "",
        *sections,
    ]


def render_ee_detail_sections(
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


def render_ee_rollup_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    blocks = [
        render_ee_trivial_pass_sections(scenarios, results),
        render_ee_metrics_summary(scenarios, results, memory_labels),
        render_ee_detail_sections(scenarios, results),
    ]
    sections: list[str] = []
    for block in blocks:
        if not block:
            continue
        if sections:
            sections.append("")
        sections.extend(block)
    return sections


def write_ee_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate EE experiment runs.jsonl into `_rollup/ee/report.md`."""
    return write_typed_rollup_report(
        results_dir,
        benchmark_type=_EE_TYPE,
        render_sections=render_ee_rollup_sections,
        title="erasability & editability benchmark",
    )


__all__ = [
    "render_ee_detail_sections",
    "render_ee_metrics_summary",
    "render_ee_rollup_sections",
    "render_ee_trivial_pass_sections",
    "write_ee_rollup_report",
]
