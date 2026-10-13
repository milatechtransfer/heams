"""Aggregated per-run sub-metrics CSV export.

Flattens every benchmark's tagged evaluation scores and efficiency metrics
into one row-per-metric CSV per run, so results can be pivoted/aggregated
outside `heams` instead of only appearing in per-experiment Markdown reports.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from heams.evaluation.report_metrics import (
    COMPONENT_PERFORMANCE_REPORT_METRICS,
    COMPONENT_REPORT_METRICS,
    EXPERIMENT_SYSTEM_REPORT_METRICS,
    MetricSpec,
)
from heams.evaluation.schemas import EvaluationRecord
from heams.reporting.benchmark_rollup import (
    load_experiment_config,
    memory_label_from_variant_payload,
)
from heams.reporting.experiment_report import discover_experiment_dirs, load_run_results
from heams.results import RunResult

SUBMETRIC_CSV_FILENAME = "submetrics.csv"

_CSV_FIELDS = [
    "campaign",
    "experiment",
    "run_id",
    "memory_system",
    "benchmark",
    "metric_name",
    "metric_value",
    "agent_model_name",
    "timestamp",
]

#: `foo_label` is a human-readable companion to `foo` (same convention as
#: `heams.reporting.experiment_report`'s `_LABEL_TAG_SUFFIX`) - folded into
#: `foo`'s value here rather than emitted as its own metric row.
_LABEL_TAG_SUFFIX = "_label"


def _tag_metric_items(record: EvaluationRecord) -> list[tuple[str, str]]:
    """(tag_key, display_value) pairs for `record`, folding `<key>_label` in."""
    keys = set(record.tags)
    label_only_keys = {
        key
        for key in keys
        if key.endswith(_LABEL_TAG_SUFFIX) and key[: -len(_LABEL_TAG_SUFFIX)] in keys
    }
    return [
        (key, record.tags.get(f"{key}{_LABEL_TAG_SUFFIX}", record.tags[key]))
        for key in sorted(keys - label_only_keys)
    ]


def _format_metric_value(value: object, precision: int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and precision is not None:
        return f"{value:.{precision}f}"
    return str(value)


def _metric_rows(
    shared: dict[str, str], specs: list[MetricSpec], summary: object
) -> list[dict[str, str]]:
    return [
        {
            **shared,
            "metric_name": label,
            "metric_value": _format_metric_value(accessor(summary), precision),
        }
        for label, accessor, precision in specs
    ]


def _run_result_rows(run: RunResult, *, shared: dict[str, str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    for record in run.audit.report_records or []:
        metric_value = (
            record.score if record.score is not None else float(record.passed)
        )
        for tag_key, tag_value in _tag_metric_items(record):
            rows.append(
                {
                    **shared,
                    "metric_name": f"{tag_key}={tag_value}",
                    "metric_value": _format_metric_value(metric_value, 4),
                }
            )

    if run.system_metrics is not None:
        rows.extend(
            _metric_rows(shared, EXPERIMENT_SYSTEM_REPORT_METRICS, run.system_metrics)
        )

    for summary in run.component_metrics or []:
        rows.extend(
            {**row, "metric_name": f"{summary.component}:{row['metric_name']}"}
            for row in _metric_rows(shared, COMPONENT_REPORT_METRICS, summary)
        )

    for perf_summary in run.component_performance_metrics or []:
        rows.extend(
            {**row, "metric_name": f"{perf_summary.component}:{row['metric_name']}"}
            for row in _metric_rows(
                shared, COMPONENT_PERFORMANCE_REPORT_METRICS, perf_summary
            )
        )

    return rows


def _load_run_metadata(path: Path) -> dict[str, str] | None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return {
        "campaign": str(payload.get("run_name") or ""),
        "run_id": str(payload.get("run_id") or ""),
        "timestamp": str(payload.get("executed_at") or ""),
    }


def _shared_columns(
    run_metadata: dict[str, str], experiment_config: dict[str, object]
) -> dict[str, str]:
    benchmark_config = experiment_config.get("benchmark_config")
    agent_config = experiment_config.get("agent_config")
    return {
        **run_metadata,
        "experiment": str(experiment_config.get("experiment_name") or ""),
        "memory_system": memory_label_from_variant_payload(
            experiment_config.get("memory_system_config")
        ),
        "benchmark": str(benchmark_config.get("type", ""))
        if isinstance(benchmark_config, dict)
        else "",
        "agent_model_name": str(agent_config.get("model_id", ""))
        if isinstance(agent_config, dict)
        else "",
    }


def generate_submetric_csv(results_dir: Path) -> Path | None:
    """Write `submetrics.csv` for a run, or return `None` if there's nothing to write."""
    run_snapshots = sorted(results_dir.glob("run_config.*.yaml"))
    if not run_snapshots:
        return None
    run_metadata = _load_run_metadata(run_snapshots[-1])
    if run_metadata is None:
        return None

    rows: list[dict[str, str]] = []
    for experiment_dir in discover_experiment_dirs(results_dir):
        experiment_config = load_experiment_config(experiment_dir)
        if experiment_config is None:
            continue
        shared = _shared_columns(run_metadata, experiment_config)
        for run in load_run_results(experiment_dir / "runs.jsonl"):
            rows.extend(_run_result_rows(run, shared=shared))

    if not rows:
        return None

    csv_path = results_dir / SUBMETRIC_CSV_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


__all__ = ["SUBMETRIC_CSV_FILENAME", "generate_submetric_csv"]
