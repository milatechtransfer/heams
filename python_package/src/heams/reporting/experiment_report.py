"""Per-experiment Markdown reports built directly from a saved ``runs.jsonl``.

Unlike `heams.reporting.benchmark_report`, this module never needs the live
`Scenario`/`Benchmark` objects that produced a run - `RunResult` is fully
self-contained, so a report can be regenerated after the fact from disk
alone. This is what backs the `heams generate-report` CLI command.
"""

from __future__ import annotations

import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from heams.evaluation.report_metrics import (
    COMPONENT_PERFORMANCE_REPORT_METRICS,
    COMPONENT_REPORT_METRICS,
    EXPERIMENT_SYSTEM_REPORT_METRICS,
)
from heams.evaluation.schemas import EvaluationRecord
from heams.execution_traces import ExecutionTraceRecord, load_execution_traces
from heams.reporting.benchmark_report import render_tool_result_output
from heams.reporting.benchmark_rollup import _ROLLUP_DIRNAME
from heams.reporting.tables import (
    format_metric_cell,
    render_comparison_table,
    render_mean_metric_rows,
    render_stratified_table,
)
from heams.results import RunResult

REPORT_FILENAME = "report.md"
RUNS_JSONL_FILENAME = "runs.jsonl"
EXECUTION_TRACES_JSONL_FILENAME = "execution_traces.jsonl"

#: Excluded by name, not just by the absence of a `runs.jsonl`, so a future
#: rollup artifact shape never gets mistaken for a per-experiment directory.
_RESERVED_EXPERIMENT_DIR_NAMES = frozenset({_ROLLUP_DIRNAME})

#: Single pseudo-label used for experiment-wide mean tables; an experiment
#: has exactly one resolved agent/memory combo, so there is nothing to
#: compare across (unlike the legacy per-scenario memory-backend tables).
_EXPERIMENT_LABEL = "experiment"


def discover_experiment_dirs(results_dir: Path) -> list[Path]:
    """Immediate subdirectories of `results_dir` that look like experiments."""
    if not results_dir.is_dir():
        return []
    return sorted(
        child
        for child in results_dir.iterdir()
        if child.is_dir()
        and child.name not in _RESERVED_EXPERIMENT_DIR_NAMES
        and (child / RUNS_JSONL_FILENAME).is_file()
    )


def load_run_results(jsonl_path: Path) -> list[RunResult]:
    """Parse a `runs.jsonl` file into `RunResult`s, skipping malformed lines.

    Splits on physical LF bytes rather than decoding to text and calling
    `str.splitlines()`, which also treats U+2028/U+2029/U+0085 as record
    boundaries even when they occur inside a JSON string value.
    """
    results: list[RunResult] = []
    for line_number, raw_line in enumerate(
        jsonl_path.read_bytes().split(b"\n"), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            results.append(RunResult.model_validate_json(raw_line))
        except ValidationError as exc:
            print(
                f"Skipping malformed run record at {jsonl_path}:{line_number}: {exc}",
                file=sys.stderr,
            )
    return results


def _mean_accuracy(results: Sequence[RunResult]) -> float | None:
    if not results:
        return None
    return statistics.mean(run.audit.accuracy_score for run in results)


def _render_top_level_metrics(results: Sequence[RunResult]) -> list[str]:
    lines = ["## Top-level metrics", ""]
    total = len(results)
    successes = sum(1 for run in results if run.audit.is_successful)
    lines.append(f"- Runs: {total}")
    lines.append(f"- Successful runs: {successes} / {total}")
    lines.append(
        f"- Mean accuracy: {format_metric_cell(_mean_accuracy(results), precision=4)}"
    )
    lines.append("")

    system_metrics = [
        run.system_metrics for run in results if run.system_metrics is not None
    ]
    if system_metrics:
        rows = render_mean_metric_rows(
            EXPERIMENT_SYSTEM_REPORT_METRICS,
            [_EXPERIMENT_LABEL],
            {_EXPERIMENT_LABEL: system_metrics},
        )
        lines.extend(
            render_comparison_table("Mean system metrics", [_EXPERIMENT_LABEL], rows)
        )
        lines.append("")

    component_metrics_by_name: dict[str, list] = defaultdict(list)
    for run in results:
        for summary in run.component_metrics or []:
            component_metrics_by_name[summary.component].append(summary)
    if component_metrics_by_name:
        component_names = sorted(component_metrics_by_name)
        rows = render_mean_metric_rows(
            COMPONENT_REPORT_METRICS, component_names, component_metrics_by_name
        )
        lines.extend(
            render_comparison_table("Mean component metrics", component_names, rows)
        )
        lines.append("")

    component_perf_by_name: dict[str, list] = defaultdict(list)
    for run in results:
        for summary in run.component_performance_metrics or []:
            component_perf_by_name[summary.component].append(summary)
    if component_perf_by_name:
        component_names = sorted(component_perf_by_name)
        rows = render_mean_metric_rows(
            COMPONENT_PERFORMANCE_REPORT_METRICS,
            component_names,
            component_perf_by_name,
        )
        lines.extend(
            render_comparison_table(
                "Mean component performance metrics", component_names, rows
            )
        )
        lines.append("")

    lines.extend(_render_sub_metric_tables(results))

    return lines


#: Tag keys a benchmark can use to get better-than-flat reporting:
#: - `record_type`: different kinds of checks (e.g. leak vs. helpfulness).
#:   Splits every other tag's table by this instead of pooling them.
#: - `pass_type`: whether a check is a genuine ("real") pass vs. one that
#:   only happened via a loophole. Adds a filtered rate next to the raw one.
_RECORD_TYPE_TAG_KEY = "record_type"
_PASS_TYPE_TAG_KEY = "pass_type"
_PASS_TYPE_REAL_VALUE = "real"

#: `foo_label` is a human-readable companion to `foo` (e.g. STALE's
#: dimension/dimension_label, EE's task_type/task_type_label) - not a
#: separate dimension, so it gets folded into `foo`'s table as the display
#: value instead of rendering its own, duplicate table.
_LABEL_TAG_SUFFIX = "_label"

#: Minimum distinct values before the identifier heuristic below applies -
#: too few, and a real category can look like an identifier by chance.
_IDENTIFIER_MIN_DISTINCT_VALUES = 8
#: Max average records per value still counted as "looks like an identifier"
#: (sample_id, question_id) rather than a real, repeating category.
_IDENTIFIER_MAX_RECORDS_PER_VALUE = 1.5


def _tag_display_value(
    record: EvaluationRecord, tag_key: str, label_key: str | None
) -> str | None:
    """The value to show for `tag_key`, preferring its `_label` companion."""
    if label_key is not None:
        label = record.tags.get(label_key)
        if label is not None:
            return label
    return record.tags.get(tag_key)


def _bucket_stats(records: Sequence[tuple[bool, float | None]]) -> dict[str, object]:
    passed = sum(1 for is_passed, _ in records if is_passed)
    scores = [score for _, score in records if score is not None]
    return {"passed": passed, "total": len(records), "scores": scores}


def _looks_like_identifier(
    buckets: dict[str, list[tuple[bool, float | None]]],
) -> bool:
    """Heuristic for a tag that's a raw identifier rather than a category.

    Only a real-campaign-scale check, not a smoke-scale one: with too few
    distinct values there isn't enough data to tell an identifier apart from
    a real category that happens to have one record per value in a small run.
    """
    distinct_values = len(buckets)
    if distinct_values < _IDENTIFIER_MIN_DISTINCT_VALUES:
        return False
    total_records = sum(len(records) for records in buckets.values())
    return total_records / distinct_values <= _IDENTIFIER_MAX_RECORDS_PER_VALUE


def _render_value_table(
    title: str, buckets: dict[str, list[tuple[bool, float | None]]]
) -> list[str]:
    if not buckets:
        return []
    if _looks_like_identifier(buckets):
        return [
            f"_{title}: omitted - {len(buckets)} distinct values with about "
            "one record each look like identifiers, not a real category, so "
            "a breakdown table would add no aggregate information._",
            "",
        ]
    rows = []
    rates: dict[str, float] = {}
    for value in sorted(buckets):
        stats = _bucket_stats(buckets[value])
        passed, total, scores = stats["passed"], stats["total"], stats["scores"]
        rate = passed / total if total else None
        if rate is not None:
            rates[value] = rate
        rows.append(
            [
                value,
                f"{passed}/{total}",
                format_metric_cell(rate, precision=4),
                format_metric_cell(
                    statistics.mean(scores) if scores else None, precision=4
                ),
            ]
        )
    lines = render_stratified_table(
        title, ["Value", "Records", "Pass rate", "Mean score"], rows, numeric_from=1
    )
    # A single value's "average"/"weakest" would just restate that one rate,
    # so these only add information once there's more than one to compare.
    if len(rates) > 1:
        macro_average = statistics.mean(rates.values())
        weakest_value = min(rates, key=lambda value: rates[value])
        lines.append(
            f"- Average across values: {format_metric_cell(macro_average, precision=4)}"
        )
        lines.append(
            f"- Weakest value: `{weakest_value}` "
            f"({format_metric_cell(rates[weakest_value], precision=4)})"
        )
    lines.append("")
    return lines


def _render_record_type_split_table(
    title: str,
    tag_key: str,
    records: Sequence[EvaluationRecord],
    label_key: str | None = None,
) -> list[str]:
    """One table with `record_type` as a column, not one table per type.

    A pooled rate mixes different kinds of checks (e.g. a leak check and a
    helpfulness check) under one tag value into a meaningless number, so this
    breaks it out by record_type - but as columns in one table, not as
    several separately-headed tables a reader has to piece back together.
    """
    value_buckets: dict[str, list[tuple[bool, float | None]]] = defaultdict(list)
    combo_buckets: dict[tuple[str, str], list[tuple[bool, float | None]]] = defaultdict(
        list
    )
    for record in records:
        value = _tag_display_value(record, tag_key, label_key)
        if value is None:
            continue
        record_type = record.tags.get(_RECORD_TYPE_TAG_KEY, "unspecified")
        value_buckets[value].append((record.passed, record.score))
        combo_buckets[(value, record_type)].append((record.passed, record.score))
    if not combo_buckets:
        return []
    if _looks_like_identifier(value_buckets):
        return [
            f"_{title}: omitted - {len(value_buckets)} distinct values with "
            "about one record each look like identifiers, not a real "
            "category, so a breakdown table would add no aggregate "
            "information._",
            "",
        ]
    rows = []
    for value, record_type in sorted(combo_buckets):
        stats = _bucket_stats(combo_buckets[(value, record_type)])
        passed, total, scores = stats["passed"], stats["total"], stats["scores"]
        rows.append(
            [
                value,
                record_type,
                f"{passed}/{total}",
                format_metric_cell(passed / total if total else None, precision=4),
                format_metric_cell(
                    statistics.mean(scores) if scores else None, precision=4
                ),
            ]
        )
    lines = render_stratified_table(
        title,
        ["Value", "Check type", "Records", "Pass rate", "Mean score"],
        rows,
        numeric_from=2,
    )
    lines.append("")
    return lines


def _render_raw_vs_real_table(
    title: str,
    tag_key: str,
    records: Sequence[EvaluationRecord],
    label_key: str | None = None,
) -> list[str]:
    """One table with both the raw and the anti-gaming-filtered rate.

    Showing the real number in its own separate table risked a reader only
    seeing the raw one and stopping there, so both live in the same row: the
    gap between them is itself the signal (a system that "passes" raw but
    not real relied on a loophole rather than a genuine success).
    """
    raw_buckets: dict[str, list[tuple[bool, float | None]]] = defaultdict(list)
    real_buckets: dict[str, list[tuple[bool, float | None]]] = defaultdict(list)
    for record in records:
        value = _tag_display_value(record, tag_key, label_key)
        if value is None:
            continue
        raw_buckets[value].append((record.passed, record.score))
        pass_type = record.tags.get(_PASS_TYPE_TAG_KEY)
        if pass_type is not None:
            is_real = pass_type == _PASS_TYPE_REAL_VALUE
            real_buckets[value].append((is_real, 1.0 if is_real else 0.0))
    if not raw_buckets:
        return []
    if _looks_like_identifier(raw_buckets):
        return [
            f"_{title}: omitted - {len(raw_buckets)} distinct values with "
            "about one record each look like identifiers, not a real "
            "category, so a breakdown table would add no aggregate "
            "information._",
            "",
        ]
    rows = []
    real_rates: dict[str, float] = {}
    for value in sorted(raw_buckets):
        raw_stats = _bucket_stats(raw_buckets[value])
        raw_passed, raw_total = raw_stats["passed"], raw_stats["total"]
        raw_rate = raw_passed / raw_total if raw_total else None
        real_cell = "n/a"
        if value in real_buckets:
            real_stats = _bucket_stats(real_buckets[value])
            real_passed, real_total = real_stats["passed"], real_stats["total"]
            real_rate = real_passed / real_total if real_total else None
            if real_rate is not None:
                real_rates[value] = real_rate
            real_cell = f"{format_metric_cell(real_rate, precision=4)} ({real_passed}/{real_total})"
        rows.append(
            [
                value,
                f"{raw_passed}/{raw_total}",
                format_metric_cell(raw_rate, precision=4),
                real_cell,
            ]
        )
    lines = render_stratified_table(
        title,
        ["Value", "Records", "Raw pass rate", "Real pass rate"],
        rows,
        numeric_from=1,
    )
    if len(real_rates) > 1:
        macro_average = statistics.mean(real_rates.values())
        weakest_value = min(real_rates, key=lambda value: real_rates[value])
        lines.append(
            f"- Average real pass rate: "
            f"{format_metric_cell(macro_average, precision=4)}"
        )
        lines.append(
            f"- Weakest value: `{weakest_value}` "
            f"({format_metric_cell(real_rates[weakest_value], precision=4)})"
        )
    lines.append("")
    return lines


def _render_sub_metric_tables(results: Sequence[RunResult]) -> list[str]:
    """Aggregate pass rate and mean score per tag key found on `report_records`.

    Tag keys aren't known ahead of time, so they're discovered per experiment
    rather than declared up front like the fixed metric tables above. Records
    carrying the reserved `record_type`/`pass_type` tags get one combined
    table (record_type as a column, or raw+real side by side) instead of a
    flat pass rate - see the module-level constants above.
    """
    records: list[EvaluationRecord] = [
        record for run in results for record in (run.audit.report_records or [])
    ]
    if not records:
        return []

    has_record_type = any(_RECORD_TYPE_TAG_KEY in record.tags for record in records)
    has_pass_type = any(_PASS_TYPE_TAG_KEY in record.tags for record in records)
    tag_keys = sorted({key for record in records for key in record.tags})

    lines: list[str] = []
    # One-time, plain-language explanation of what the extra column(s) below
    # mean - printed once here rather than repeated under every table, since
    # a tag key can produce many tables (task_type, domain, entity, ...).
    if has_pass_type:
        lines.append(
            "_Real is the actual score. Raw is more lenient: it also counts a "
            "check as passing when there was nothing to get wrong in the "
            "first place - for example, a system that never learned a fact "
            'will always "correctly" say it doesn\'t know it, without ever '
            "proving it can track a real change. A big gap between Raw and "
            "Real means many of those raw passes were free wins like this, "
            "not genuine successes._"
        )
        lines.append("")
    if has_record_type:
        lines.append(
            "_Each row is one kind of check, not a combined score - whether a "
            "secret leaked and whether the response was actually helpful are "
            "different questions with different pass/fail meanings, so "
            "averaging them together would produce a number that doesn't "
            "answer either one; if one check type represents the final "
            "outcome, that's the row to look at first._"
        )
        lines.append("")
    # `foo_label` is a display-only companion to `foo`, not its own
    # dimension - fold it into `foo`'s table instead of rendering a
    # duplicate table with the same grouping but nicer-looking values.
    label_only_keys = {
        key
        for key in tag_keys
        if key.endswith(_LABEL_TAG_SUFFIX)
        and key[: -len(_LABEL_TAG_SUFFIX)] in tag_keys
    }

    for tag_key in tag_keys:
        if tag_key in label_only_keys:
            continue
        label_key = f"{tag_key}{_LABEL_TAG_SUFFIX}"
        label_key = label_key if label_key in tag_keys else None
        title = f"Mean sub-metrics ({tag_key})"
        if tag_key in (_RECORD_TYPE_TAG_KEY, _PASS_TYPE_TAG_KEY):
            buckets: dict[str, list[tuple[bool, float | None]]] = defaultdict(list)
            for record in records:
                value = _tag_display_value(record, tag_key, label_key)
                if value is not None:
                    buckets[value].append((record.passed, record.score))
            lines.extend(_render_value_table(title, buckets))
        elif has_pass_type:
            lines.extend(_render_raw_vs_real_table(title, tag_key, records, label_key))
        elif has_record_type:
            lines.extend(
                _render_record_type_split_table(title, tag_key, records, label_key)
            )
        else:
            buckets = defaultdict(list)
            for record in records:
                value = _tag_display_value(record, tag_key, label_key)
                if value is not None:
                    buckets[value].append((record.passed, record.score))
            lines.extend(_render_value_table(title, buckets))

    return lines


def _sorted_results(results: Sequence[RunResult]) -> list[RunResult]:
    return sorted(
        results,
        key=lambda run: (run.scenario_id, run.sample_index, run.evaluator_index),
    )


def _render_summary_table(results: Sequence[RunResult]) -> list[str]:
    lines = [
        "## Summary",
        "",
        "| Scenario | Sample | Evaluator | Success | Accuracy | Failed keys |",
        "|---|---:|---:|---|---:|---|",
    ]
    for run in _sorted_results(results):
        failed_keys = ", ".join(run.audit.failed_keys) if run.audit.failed_keys else "-"
        lines.append(
            f"| {run.scenario_id} | {run.sample_index} | {run.evaluator_index} | "
            f"{run.audit.is_successful} | {run.audit.accuracy_score:.2f} | "
            f"{failed_keys} |"
        )
    lines.append("")
    return lines


def _render_report_records(run: RunResult) -> list[str]:
    records = run.audit.report_records
    if not records:
        return []
    lines = [
        "**LLM judge output:**",
        "",
        "| Key | Passed | Score | Expected | Observed | Reason |",
        "|---|---|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.key or '-'} | {record.passed} | "
            f"{format_metric_cell(record.score, precision=4)} | "
            f"{record.expected or '-'} | {record.observed or '-'} | "
            f"{record.reason or '-'} |"
        )
    lines.append("")
    return lines


def _render_transcript(run: RunResult) -> list[str]:
    if not run.transcript:
        return []
    lines = ["**Transcript:**", ""]
    for turn_idx, turn in enumerate(run.transcript):
        lines.append(f"**Q{turn_idx}:** {turn.question}")
        lines.append("")
        lines.append(f"**A{turn_idx}:** {turn.answer}")
        lines.append("")
        if turn.target_answer is not None:
            lines.append(f"**Target{turn_idx}:** {turn.target_answer}")
            lines.append("")
        if turn.tool_results:
            lines.append("Tool results:")
            lines.append("")
            for result in turn.tool_results:
                lines.append(f"- `{result.tool_name}`")
                lines.append("")
                lines.append("```json")
                lines.append(render_tool_result_output(result.output))
                lines.append("```")
                lines.append("")
    return lines


def _trace_content_preview(value: object, *, limit: int = 180) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _render_counts(counts: Counter[str]) -> str:
    return ", ".join(f"`{name}` × {count}" for name, count in sorted(counts.items()))


def _render_execution_trace(trace: ExecutionTraceRecord | None) -> list[str]:
    if trace is None:
        return []

    type_counts = Counter(event.type for event in trace.events)
    tool_counts = Counter(
        event.name for event in trace.events if event.type == "tool_call"
    )
    lines = [
        "**Execution trace:**",
        "",
        f"- Outcome: {trace.outcome}",
        f"- Events: {len(trace.events)}",
        f"- Event types: {_render_counts(type_counts)}",
    ]
    if tool_counts:
        lines.append(f"- Tools: {_render_counts(tool_counts)}")
    lines.append("")

    for event in trace.events:
        duration = (
            f", {event.duration_ms:.2f} ms" if event.duration_ms is not None else ""
        )
        preview_source = event.content or event.error or event.result
        preview = (
            f" — {_trace_content_preview(preview_source)}"
            if preview_source is not None
            else ""
        )
        lines.append(
            f"{event.sequence_index + 1}. `{event.phase}` / `{event.type}` / "
            f"`{event.name}` — {event.status}{duration}{preview}"
        )
    lines.append("")
    return lines


def _select_example_runs(
    results: Sequence[RunResult], max_examples: int | None
) -> list[RunResult]:
    """Pick which runs get a full detail section, keeping both outcomes visible.

    Below the cap, every run is shown. Above it, the cap is split evenly
    between failed and successful runs (favoring failures by one on an odd
    split) so a capped report never reads as all-failure or all-success by
    luck of ordering; a bucket that runs out is backfilled from the other.
    """
    ordered = _sorted_results(results)
    if max_examples is None or len(ordered) <= max_examples:
        return ordered

    failures = [run for run in ordered if not run.audit.is_successful]
    successes = [run for run in ordered if run.audit.is_successful]

    failure_quota = max_examples - max_examples // 2
    success_quota = max_examples // 2

    selected_failures = failures[:failure_quota]
    selected_successes = successes[:success_quota]

    shortfall = max_examples - len(selected_failures) - len(selected_successes)
    if shortfall > 0:
        selected_successes.extend(
            successes[len(selected_successes) : len(selected_successes) + shortfall]
        )
        shortfall = max_examples - len(selected_failures) - len(selected_successes)
    if shortfall > 0:
        selected_failures.extend(
            failures[len(selected_failures) : len(selected_failures) + shortfall]
        )

    selected_ids = {id(run) for run in (*selected_failures, *selected_successes)}
    return [run for run in ordered if id(run) in selected_ids]


def _render_run_detail(
    run: RunResult, trace: ExecutionTraceRecord | None = None
) -> list[str]:
    lines = [
        f"### {run.scenario_id} / sample {run.sample_index} "
        f"/ evaluator {run.evaluator_index}",
        "",
    ]
    lines.append(f"- is_successful: {run.audit.is_successful}")
    lines.append(f"- accuracy: {run.audit.accuracy_score:.2f}")
    if run.audit.passed_keys:
        lines.append(f"- passed_keys: {run.audit.passed_keys}")
    if run.audit.failed_keys:
        lines.append(f"- failed_keys: {run.audit.failed_keys}")
    lines.append(f"- notes: {run.audit.audit_notes}")
    lines.append("")
    lines.extend(_render_report_records(run))
    lines.extend(_render_transcript(run))
    lines.extend(_render_execution_trace(trace))
    return lines


def render_experiment_report(
    experiment_name: str,
    results: Sequence[RunResult],
    *,
    max_examples: int | None = None,
    traces_by_id: Mapping[str, ExecutionTraceRecord] | None = None,
) -> str:
    """Render the full Markdown report for one experiment's `RunResult`s.

    `max_examples` caps how many runs get a full detail section (transcript +
    judge output); see `_select_example_runs` for how the cap is split
    between failures and successes. `None` means no cap - every run gets
    full detail. The Summary table above Detail always lists every run
    regardless of the cap, so no outcome is ever silently hidden.
    """
    lines = [f"# Experiment report: {experiment_name}", ""]
    if not results:
        lines.append("No run results found.")
        return "\n".join(lines) + "\n"

    lines.extend(_render_top_level_metrics(results))
    lines.extend(_render_summary_table(results))

    example_runs = _select_example_runs(results, max_examples)
    lines.append("## Detail")
    lines.append("")
    if len(example_runs) < len(results):
        failures_shown = sum(1 for run in example_runs if not run.audit.is_successful)
        successes_shown = len(example_runs) - failures_shown
        lines.append(
            f"Showing full detail for {len(example_runs)} of {len(results)} runs "
            f"({failures_shown} failed, {successes_shown} successful) - "
            "see Summary above for every run's outcome."
        )
        lines.append("")
    for run in example_runs:
        trace = (
            traces_by_id.get(run.trace_id) if traces_by_id and run.trace_id else None
        )
        lines.extend(_render_run_detail(run, trace))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_experiment_reports(
    results_dir: Path, *, max_examples: int | None = None, detailed: bool = False
) -> list[Path]:
    """Write `report.md` for every experiment under `results_dir`.

    An experiment whose `runs.jsonl` is missing or empty is skipped (with a
    warning) rather than aborting the rest of the run folder. `detailed=True`
    also writes `submetrics.csv`, but only once at least one report was written.
    """
    written: list[Path] = []
    for experiment_dir in discover_experiment_dirs(results_dir):
        jsonl_path = experiment_dir / RUNS_JSONL_FILENAME
        results = load_run_results(jsonl_path)
        if not results:
            print(
                f"Skipping {experiment_dir}: no run results in {jsonl_path}",
                file=sys.stderr,
            )
            continue
        traces, trace_warnings = load_execution_traces(
            experiment_dir / EXECUTION_TRACES_JSONL_FILENAME
        )
        for warning in trace_warnings:
            print(warning, file=sys.stderr)
        for run in results:
            if run.trace_id is not None and run.trace_id not in traces:
                print(
                    f"Missing execution trace {run.trace_id!r} for "
                    f"{run.scenario_id} sample {run.sample_index} "
                    f"in {experiment_dir}.",
                    file=sys.stderr,
                )
        report = render_experiment_report(
            experiment_dir.name,
            results,
            max_examples=max_examples,
            traces_by_id=traces,
        )
        report_path = experiment_dir / REPORT_FILENAME
        report_path.write_text(report, encoding="utf-8")
        written.append(report_path)

    if detailed and written:
        # Local import: `submetric_csv` imports `discover_experiment_dirs`/
        # `load_run_results` from this module, so importing it back at module
        # level would be circular.
        from heams.reporting.submetric_csv import generate_submetric_csv

        csv_path = generate_submetric_csv(results_dir)
        if csv_path is None:
            print(
                f"No submetrics.csv was generated under {results_dir}.",
                file=sys.stderr,
            )
        else:
            written.append(csv_path)

    return written
