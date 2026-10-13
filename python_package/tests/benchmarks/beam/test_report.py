"""BEAM reporting and rollup tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.beam.report import render_beam_metrics_summary
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports
from heams.results import RunResult
from heams.scenarios import UserStimulus
from heams.scenarios.base import Scenario


def test_render_beam_metrics_summary_by_ability() -> None:
    scenarios = [
        Scenario(
            id="BEAM-100K-fixture-1",
            name="s1",
            memory_systems=[ExampleMemoryConfig()],
            execution_sequence=[UserStimulus(content="q")],
        )
    ]
    results = {
        "BEAM-100K-fixture-1": [
            RunResult(
                scenario_id="BEAM-100K-fixture-1",
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=True,
                    accuracy_score=0.5,
                    passed_keys=["q0:information_extraction"],
                    failed_keys=["q1:event_ordering"],
                    report_records=[
                        EvaluationRecord(
                            key="q0:information_extraction",
                            passed=True,
                            score=1.0,
                            tags={
                                "question_type": "information_extraction",
                                "token_scale": "100K",
                            },
                        ),
                        EvaluationRecord(
                            key="q1:event_ordering",
                            passed=False,
                            score=0.0,
                            tags={
                                "question_type": "event_ordering",
                                "token_scale": "100K",
                            },
                        ),
                    ],
                ),
            )
        ]
    }
    sections = render_beam_metrics_summary(
        scenarios, results, memory_labels=["example.in_memory"]
    )
    assert sections
    text = "\n".join(sections)
    assert "BEAM accuracy" in text
    assert "information_extraction" in text
    assert "event_ordering" in text


def test_write_beam_rollup_filters_other_benchmarks(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    other_dir = results_dir / "locomo_exp"
    beam_dir = results_dir / "beam_exp"
    other_dir.mkdir(parents=True)
    beam_dir.mkdir(parents=True)
    (other_dir / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "locomo"},
                "memory_system_config": {
                    "id": "no_memory",
                    "backend_config": {
                        "type": "no_memory",
                        "name": "no_memory",
                        "top_k": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (beam_dir / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "beam", "split": "100K"},
                "memory_system_config": {
                    "id": "example.in_memory",
                    "backend_config": {
                        "type": "example.in_memory",
                        "name": "example.in_memory",
                        "top_k": 50,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    run = RunResult(
        scenario_id="BEAM-100K-fixture-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["q0:information_extraction"],
            failed_keys=[],
        ),
    )
    (beam_dir / "runs.jsonl").write_text(run.model_dump_json() + "\n", encoding="utf-8")
    written = write_benchmark_rollup_reports(results_dir)
    assert "beam" in written
    assert "locomo" not in written
    report = written["beam"].read_text(encoding="utf-8")
    assert "# beam benchmark" in report
    assert "BEAM accuracy" in report
