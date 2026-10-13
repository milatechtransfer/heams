from __future__ import annotations

import yaml

from heams.benchmarks.contextual_integrity.report import (
    _new_stats,
    accumulate_ci_stats,
    render_ci_metrics_summary,
    write_ci_rollup_report,
)
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams.results import RunResult


def _summary_run(*, leaked: bool, helpful: bool, allowed_total: int, covered: int):
    records = [
        EvaluationRecord(
            key="s:summary",
            passed=not leaked,
            score=0.0 if leaked else 1.0,
            tags={
                "record_type": "ci_summary",
                "leaked": "true" if leaked else "false",
                "helpful": "true" if helpful else "false",
                "allowed_total": str(allowed_total),
                "allowed_covered": str(covered),
            },
        ),
        EvaluationRecord(
            key="s:secret:ssn",
            passed=not leaked,
            score=0.0 if leaked else 1.0,
            tags={"record_type": "secret", "field": "ssn"},
        ),
    ]
    return RunResult(
        scenario_id="CI-train-0000",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=helpful and not leaked,
            accuracy_score=1.0,
            report_records=records,
        ),
    )


def test_metrics_summary_counts_leakage_and_coverage():
    stats = accumulate_ci_stats(
        [
            _summary_run(leaked=True, helpful=True, allowed_total=4, covered=2),
            _summary_run(leaked=False, helpful=True, allowed_total=4, covered=4),
        ],
        _new_stats(),
    )
    assert stats["scenarios"] == 2
    assert stats["leaked_scen"] == 1
    assert stats["secrets"] == 2
    assert stats["secrets_leaked"] == 1
    assert stats["fields"] == 8
    assert stats["fields_covered"] == 6

    lines = render_ci_metrics_summary({"mem": stats}, ["mem"])
    row = lines[-1]
    assert "| mem | 2 |" in row
    assert "1/2 (50%)" in row  # per-scenario leakage
    assert "6/8 (75%)" in row


def test_metrics_summary_is_empty_without_scenarios():
    assert render_ci_metrics_summary({"mem": _new_stats()}, ["mem"]) == []


def test_rollup_writer_emits_report(tmp_path):
    experiment_dir = tmp_path / "gpt4o_mini_no_memory_ci_run1"
    experiment_dir.mkdir()
    (experiment_dir / "experiment_config.run1.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "ci"},
                "memory_system_config": {"id": "no_memory"},
            }
        )
    )
    (experiment_dir / "runs.jsonl").write_text(
        _summary_run(
            leaked=False, helpful=True, allowed_total=4, covered=3
        ).model_dump_json()
        + "\n"
    )

    report_path = write_ci_rollup_report(tmp_path)
    assert report_path == tmp_path / "_rollup" / "ci" / "report.md"
    body = report_path.read_text()
    assert "# ci benchmark" in body
    assert "no_memory" in body


def test_rollup_writer_ignores_non_ci_experiments(tmp_path):
    experiment_dir = tmp_path / "locomo_run"
    experiment_dir.mkdir()
    (experiment_dir / "experiment_config.run1.yaml").write_text(
        yaml.safe_dump({"benchmark_config": {"type": "locomo"}})
    )
    (experiment_dir / "runs.jsonl").write_text("")

    assert write_ci_rollup_report(tmp_path) is None
