"""Tests for the LOCOMO per-category report stratification."""

from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.locomo.report import (
    render_locomo_detail_sections,
    render_locomo_metrics_summary,
    write_locomo_rollup_report,
)
from heams.evaluation.schemas import AuditRecord
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.results import AgentTurn, RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


def _scenario(scenario_id: str = "conv-1") -> Scenario:
    return Scenario(
        id=scenario_id,
        name=scenario_id,
        execution_sequence=[UserStimulus(content="q")],
        agents=[],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[],
    )


def _run(
    scenario: Scenario,
    *,
    passed_keys: list[str],
    failed_keys: list[str],
) -> RunResult:
    total = len(passed_keys) + len(failed_keys)
    return RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=len(passed_keys) / total if total else 0.0,
            passed_keys=passed_keys,
            failed_keys=failed_keys,
            audit_notes="stub",
        ),
    )


def test_render_locomo_metrics_summary_break_accuracy_down_by_category() -> None:
    scenario = _scenario()
    run = _run(
        scenario,
        passed_keys=["q0:1", "q1:4", "q2:2"],
        failed_keys=["q3:1", "q4:3"],
    )

    rendered = "\n".join(
        render_locomo_metrics_summary(
            [scenario],
            {scenario.id: [run]},
            ["milamem"],
        )
    )

    assert "**LOCOMO accuracy per memory backend:**" in rendered
    assert "| Micro accuracy | 0.6000 |" in rendered
    assert "| Macro accuracy (category) | 0.6250 |" in rendered

    assert "**LOCOMO accuracy by category:**" in rendered
    assert "| Category | multi-hop | example.in_memory | 1 | 1/2 | 0.5000 |" in rendered
    assert (
        "| Category | single-hop | example.in_memory | 1 | 1/1 | 1.0000 |" in rendered
    )
    assert "| Category | temporal | example.in_memory | 1 | 1/1 | 1.0000 |" in rendered
    assert (
        "| Category | open-domain | example.in_memory | 1 | 0/1 | 0.0000 |" in rendered
    )


def test_render_locomo_metrics_summary_empty_without_runs() -> None:
    scenario = _scenario()

    rendered = render_locomo_metrics_summary(
        [scenario],
        {},
        ["milamem"],
    )

    assert rendered == []


def test_write_locomo_rollup_report_includes_qa_detail(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "gpt4o_mini_nomemory_locomo_run"
    experiment_dir.mkdir(parents=True)

    (experiment_dir / "experiment_config.run.yaml").write_text(
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
    run = RunResult(
        scenario_id="conv-26",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["q0:2"],
            failed_keys=[],
            audit_notes="LLM-judged 1 LOCOMO questions.",
        ),
        transcript=[
            AgentTurn(
                question="Question: When did Caroline go to the LGBTQ support group?",
                answer="The information is not mentioned.",
                target_answer="8 May 2023",
            )
        ],
    )
    (experiment_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report_path = write_locomo_rollup_report(results_dir)
    assert report_path is not None
    report = report_path.read_text(encoding="utf-8")

    assert "## Detail" in report
    assert "### conv-26 / sample 0" in report
    assert "**Run outcomes:**" in report
    assert "LLM-judged 1 LOCOMO questions." in report
    assert (
        "**Q0:** Question: When did Caroline go to the LGBTQ support group?" in report
    )
    assert "**A0:** The information is not mentioned." in report
    assert "**Target0:** 8 May 2023" in report


def test_locomo_detail_sections_include_target_answer_next_to_answer() -> None:
    scenario = Scenario(
        id="conv-26",
        name="conv-26",
        execution_sequence=[
            UserStimulus(
                content="Question: When did Caroline go to the LGBTQ support group?"
            )
        ],
        agents=[],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=[],
    )
    run = RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["q0:2"],
            failed_keys=[],
            audit_notes="stub",
        ),
        transcript=[
            AgentTurn(
                question="Question: When did Caroline go to the LGBTQ support group?",
                answer="The information is not mentioned.",
                target_answer="8 May 2023",
            )
        ],
    )

    rendered = "\n".join(
        render_locomo_detail_sections(
            scenarios=[scenario],
            results={scenario.id: [run]},
        )
    )

    assert "**A0:** The information is not mentioned." in rendered
    assert "**Target0:** 8 May 2023" in rendered
    assert rendered.index("**A0:**") < rendered.index("**Target0:**")
