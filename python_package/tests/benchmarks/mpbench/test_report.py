"""MPBench reporting and rollup tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.mpbench.report import render_mpbench_metrics_summary
from heams.evaluation.schemas import AuditRecord
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports
from heams.results import RunResult
from heams.scenarios import UserStimulus
from heams.scenarios.base import Scenario


def test_render_mpbench_metrics_summary_by_attack_class() -> None:
    scenarios = [
        Scenario(
            id="ADV_1",
            name="s1",
            memory_systems=[ExampleMemoryConfig()],
            execution_sequence=[UserStimulus(content="q")],
        ),
        Scenario(
            id="ADV_2",
            name="s2",
            memory_systems=[ExampleMemoryConfig()],
            execution_sequence=[UserStimulus(content="q")],
        ),
    ]
    results = {
        "ADV_1": [
            RunResult(
                scenario_id="ADV_1",
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=True,
                    accuracy_score=1.0,
                    passed_keys=["ADV_1:explicit"],
                    failed_keys=[],
                ),
            )
        ],
        "ADV_2": [
            RunResult(
                scenario_id="ADV_2",
                agent_index=0,
                memory_index=0,
                evaluator_index=0,
                sample_index=0,
                audit=AuditRecord(
                    is_successful=False,
                    accuracy_score=0.0,
                    passed_keys=[],
                    failed_keys=["ADV_2:fact"],
                ),
            )
        ],
    }
    sections = render_mpbench_metrics_summary(
        scenarios, results, memory_labels=["example.in_memory"]
    )
    assert sections
    text = "\n".join(sections)
    assert "MPBench resistance" in text
    assert "explicit" in text
    assert "fact" in text
    assert "macro-average" in text
    assert "worst class" in text


def test_render_mpbench_metrics_summary_empty() -> None:
    assert render_mpbench_metrics_summary([], {}, []) == []


def test_write_mpbench_rollup_filters_other_benchmarks(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    other_dir = results_dir / "locomo_exp"
    mpbench_dir = results_dir / "mpbench_exp"
    other_dir.mkdir(parents=True)
    mpbench_dir.mkdir(parents=True)

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

    (mpbench_dir / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "mpbench"},
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
        scenario_id="ADV_FIX_EXPLICIT",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["ADV_FIX_EXPLICIT:explicit"],
            failed_keys=[],
            audit_notes="resisted",
        ),
    )
    (mpbench_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n",
        encoding="utf-8",
    )

    written = write_benchmark_rollup_reports(results_dir)
    assert "mpbench" in written
    assert "locomo" not in written
    report = written["mpbench"].read_text(encoding="utf-8")
    assert "# mpbench benchmark" in report
    assert "MPBench resistance" in report
