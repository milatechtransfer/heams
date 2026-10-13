"""Roll run results up into the Retention Policy report sections."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heams.benchmarks.retention_policy.metrics import (
    compute_metrics,
    render_markdown,
    render_matrix,
)
from heams.reporting.benchmark_rollup import write_typed_rollup_report
from heams.results import RunResult
from heams.scenarios.base import Scenario

RETENTION_POLICY_TYPE = "retention_policy"


def collect_decisions(
    scenarios: Sequence[Scenario], results: dict[str, list[RunResult]]
) -> list[dict]:
    """Recover per-fact outcomes from the evaluator's `report_records`.

    Each sample now scores every fact under one policy in a single audit
    (`RetentionPolicyEvaluator.audit`), so `report_records` -- one entry per
    fact -- is the only source of per-fact outcomes. There is no single-fact
    ground truth left on the scenario to fall back to the way a one-row-per-
    sample design could.
    """
    del scenarios  # kept for signature parity with other rollup collectors
    decisions: list[dict] = []

    for runs in results.values():
        for run in runs:
            for record in run.audit.report_records or []:
                tags = record.tags or {}
                if "policy_id" not in tags:
                    continue
                decisions.append(
                    {
                        "observed": record.observed,
                        "expected": record.expected,
                        "policy_id": tags["policy_id"],
                        "category": tags.get("category", "?"),
                    }
                )
    return decisions


def render_retention_policy_rollup_sections(
    scenarios: Sequence[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: Sequence[str],
) -> list[str]:
    del memory_labels  # Metrics are pooled across backends for this benchmark.
    decisions = collect_decisions(scenarios, results)
    if not decisions:
        return []
    return render_markdown(compute_metrics(decisions)) + render_matrix(decisions)


def write_retention_policy_rollup_report(results_dir: Path) -> Path | None:
    """Aggregate experiment runs.jsonl into `_rollup/retention_policy/report.md`.

    Matches the writer each other benchmark exposes. The matrix runs as one
    experiment covering every policy, so a single run's rollup shows the full
    12x12 picture. Several experiments in one results directory (e.g. one per
    backend) have their decisions pooled into the same tables.
    """
    return write_typed_rollup_report(
        results_dir,
        benchmark_type=RETENTION_POLICY_TYPE,
        render_sections=render_retention_policy_rollup_sections,
        title="retention policy benchmark",
    )


__all__ = [
    "RETENTION_POLICY_TYPE",
    "collect_decisions",
    "render_retention_policy_rollup_sections",
    "write_retention_policy_rollup_report",
]
