"""Artifact-only STALE rollup tests (read runs.jsonl -> _rollup/stale/report.md)."""

from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.stale.report import write_stale_rollup_report
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports
from heams.results import RunResult

_DIMS = [
    ("dim1", "explicit_probing", True),
    ("dim2", "adversarial_robustness", False),
    ("dim3", "implicit_task", True),
]


def _stale_run(scenario_id: str, conflict_type: str) -> RunResult:
    records = [
        EvaluationRecord(
            key=f"q{i}:{scenario_id}:{dim}:{conflict_type}",
            passed=passed,
            score=1.0 if passed else 0.0,
            tags={
                "dimension": dim,
                "dimension_label": label,
                "conflict_type": conflict_type,
            },
            reason="ok" if passed else "missed",
        )
        for i, (dim, label, passed) in enumerate(_DIMS)
    ]
    passed_keys = [r.key for r in records if r.passed]
    failed_keys = [r.key for r in records if not r.passed]
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=len(passed_keys) / len(records),
            passed_keys=passed_keys,
            failed_keys=failed_keys,
            audit_notes="stub",
            report_records=records,
        ),
    )


def _write_stale_experiment(results_dir: Path) -> None:
    exp = results_dir / "gpt4o_mini_example_memory_stale_run"
    exp.mkdir(parents=True)
    (exp / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "stale"},
                "memory_system_config": {
                    "id": "example.in_memory",
                    "backend_config": {
                        "type": "example.in_memory",
                        "name": "example.in_memory",
                        "top_k": 10,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    lines = [
        _stale_run("STALE-1", "T1").model_dump_json(),
        _stale_run("STALE-2", "T2").model_dump_json(),
    ]
    (exp / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stale_rollup_written_with_per_dimension_breakdown(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_stale_experiment(results_dir)

    written = write_benchmark_rollup_reports(results_dir)

    assert "stale" in written  # dispatcher registration works
    report = written["stale"].read_text(encoding="utf-8")
    assert "# stale benchmark" in report
    assert "STALE accuracy per memory backend" in report
    assert "STALE accuracy by dimension" in report
    # Per-dimension stratification is present.
    for _dim, label, _passed in _DIMS:
        assert label in report
    # Conflict-type split is present.
    assert "T1" in report and "T2" in report


def test_write_stale_rollup_report_direct(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_stale_experiment(results_dir)

    path = write_stale_rollup_report(results_dir)

    assert path is not None
    assert path == results_dir / "_rollup" / "stale" / "report.md"
    assert path.is_file()


def test_stale_rollup_absent_without_stale_experiments(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    assert write_stale_rollup_report(results_dir) is None
    assert "stale" not in write_benchmark_rollup_reports(results_dir)
