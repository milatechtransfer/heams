"""Tests for the `heams generate-report --detailed` sub-metrics CSV export."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from heams.evaluation.schemas import (
    AuditRecord,
    ComponentMetricsSummary,
    EvaluationRecord,
    SystemMetricsSummary,
)
from heams.reporting.submetric_csv import SUBMETRIC_CSV_FILENAME, generate_submetric_csv
from heams.results import RunResult


def _run(
    *,
    scenario_id: str = "conv-1",
    sample_index: int = 0,
    report_records: list[EvaluationRecord] | None = None,
    system_metrics: SystemMetricsSummary | None = None,
    component_metrics: list[ComponentMetricsSummary] | None = None,
) -> RunResult:
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=sample_index,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            report_records=report_records,
        ),
        system_metrics=system_metrics,
        component_metrics=component_metrics,
    )


def _write_run_snapshot(
    results_dir: Path,
    *,
    run_id: str = "20260101-000000-abcd1234",
    run_name: str = "campaign-a",
    executed_at: str = "2026-01-01T00:00:00+00:00",
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"run_config.{run_id}.yaml").write_text(
        yaml.safe_dump(
            {"run_id": run_id, "run_name": run_name, "executed_at": executed_at}
        ),
        encoding="utf-8",
    )


def _write_experiment_snapshot(
    experiment_dir: Path,
    *,
    run_id: str = "20260101-000000-abcd1234",
    experiment_name: str = "exp_a",
    benchmark_type: str = "locomo",
    agent_model_id: str = "openai:gpt-4o-mini",
    memory_id: str = "no_memory",
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / f"experiment_config.{run_id}.yaml").write_text(
        yaml.safe_dump(
            {
                "experiment_name": experiment_name,
                "benchmark_config": {"type": benchmark_type},
                "agent_config": {"model_id": agent_model_id},
                "memory_system_config": {
                    "id": memory_id,
                    "backend_config": {"type": "no_memory"},
                },
            }
        ),
        encoding="utf-8",
    )


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def test_generate_submetric_csv_writes_tag_and_efficiency_rows(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_run_snapshot(results_dir)
    _write_experiment_snapshot(experiment_dir)

    run = _run(
        report_records=[
            EvaluationRecord(
                key="q0", passed=True, score=0.75, tags={"category": "temporal"}
            ),
        ],
        system_metrics=SystemMetricsSummary(
            run_duration_ms=1234.5,
            llm_input_tokens=30,
            llm_output_tokens=12,
            llm_total_tokens=42,
            llm_input_cost=0.000002,
            tool_execution_count=2,
            agent_duration_ms=250,
        ),
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    csv_path = generate_submetric_csv(results_dir)

    assert csv_path == results_dir / SUBMETRIC_CSV_FILENAME
    rows = _read_rows(csv_path)
    shared = {
        "campaign": "campaign-a",
        "experiment": "exp_a",
        "run_id": "20260101-000000-abcd1234",
        "memory_system": "no_memory",
        "benchmark": "locomo",
        "agent_model_name": "openai:gpt-4o-mini",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    assert {
        **shared,
        "metric_name": "category=temporal",
        "metric_value": "0.7500",
    } in rows
    assert {
        **shared,
        "metric_name": "Run duration (ms)",
        "metric_value": "1234.50",
    } in rows
    assert {**shared, "metric_name": "LLM total tokens", "metric_value": "42"} in rows
    assert {**shared, "metric_name": "LLM input tokens", "metric_value": "30"} in rows
    assert {**shared, "metric_name": "LLM output tokens", "metric_value": "12"} in rows
    assert {
        **shared,
        "metric_name": "LLM input cost",
        "metric_value": "0.000002",
    } in rows
    assert {**shared, "metric_name": "Tool executions", "metric_value": "2"} in rows
    assert {
        **shared,
        "metric_name": "Agent duration (ms)",
        "metric_value": "250.00",
    } in rows


def test_generate_submetric_csv_folds_label_companion_tag(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_run_snapshot(results_dir)
    _write_experiment_snapshot(experiment_dir)

    run = _run(
        report_records=[
            EvaluationRecord(
                key="q0",
                passed=True,
                score=1.0,
                tags={"dimension": "d1", "dimension_label": "Human readable"},
            ),
        ],
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    rows = _read_rows(generate_submetric_csv(results_dir))
    metric_names = [row["metric_name"] for row in rows]

    assert "dimension=Human readable" in metric_names
    assert not any(name.startswith("dimension_label=") for name in metric_names)


def test_generate_submetric_csv_prefixes_component_metrics_by_component(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_run_snapshot(results_dir)
    _write_experiment_snapshot(experiment_dir)

    run = _run(
        component_metrics=[
            ComponentMetricsSummary(component="vector_store", total_operations=3),
        ],
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    rows = _read_rows(generate_submetric_csv(results_dir))

    assert {row["metric_name"]: row["metric_value"] for row in rows}[
        "vector_store:Total operations"
    ] == "3"


def test_generate_submetric_csv_uses_score_fallback_when_absent(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_run_snapshot(results_dir)
    _write_experiment_snapshot(experiment_dir)

    run = _run(
        report_records=[
            EvaluationRecord(key="q0", passed=False, tags={"category": "temporal"}),
        ],
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    rows = _read_rows(generate_submetric_csv(results_dir))

    [row] = [row for row in rows if row["metric_name"] == "category=temporal"]
    assert row["metric_value"] == "0.0000"


def test_generate_submetric_csv_returns_none_without_run_snapshot(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_experiment_snapshot(experiment_dir)
    run = _run(
        report_records=[EvaluationRecord(key="q0", passed=True, tags={"c": "v"})]
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    assert generate_submetric_csv(results_dir) is None
    assert not (results_dir / SUBMETRIC_CSV_FILENAME).exists()


def test_generate_submetric_csv_returns_none_without_experiment_dirs(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    _write_run_snapshot(results_dir)

    assert generate_submetric_csv(results_dir) is None


def test_generate_submetric_csv_returns_none_when_runs_have_no_tags_or_metrics(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a"
    _write_run_snapshot(results_dir)
    _write_experiment_snapshot(experiment_dir)
    run = _run()
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n", encoding="utf-8"
    )

    assert generate_submetric_csv(results_dir) is None
