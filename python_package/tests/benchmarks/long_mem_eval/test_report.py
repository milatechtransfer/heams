"""Tests for the LongMemEval per-stratum report stratification."""

from __future__ import annotations

from types import SimpleNamespace

from heams.benchmarks.long_mem_eval.report import render_long_mem_eval_metrics_summary
from heams.evaluation.schemas import AuditRecord
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.results import RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


def _scenario(scenario_id: str = "lme-1") -> Scenario:
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


def test_render_long_mem_eval_metrics_summary_break_accuracy_down_by_question_type() -> (
    None
):
    scenario = _scenario()
    run = _run(
        scenario,
        passed_keys=[
            "q0:qa1:single-session-user",
            "q1:qa2:temporal-reasoning",
        ],
        failed_keys=[
            "q2:qa3:single-session-user",
            "q3:qa4:multi-session:abstention",
        ],
    )

    rendered = "\n".join(
        render_long_mem_eval_metrics_summary(
            [scenario],
            {scenario.id: [run]},
            ["example.in_memory"],
        )
    )

    assert "**LongMemEval accuracy per memory backend:**" in rendered
    assert "| Micro accuracy | 0.5000 |" in rendered
    assert "| Macro accuracy (question type) |" in rendered
    assert "| Macro accuracy (abstention) |" in rendered

    assert "**LongMemEval accuracy by stratum:**" in rendered
    assert (
        "| Question type | single-session-user | example.in_memory | 1 | 1/2 | 0.5000 |"
        in rendered
    )
    assert (
        "| Question type | temporal-reasoning | example.in_memory | 1 | 1/1 | 1.0000 |"
        in rendered
    )
    assert (
        "| Question type | multi-session | example.in_memory | 1 | 0/1 | 0.0000 |"
        in rendered
    )
    assert (
        "| Abstention | abstention | example.in_memory | 1 | 0/1 | 0.0000 |" in rendered
    )
    assert (
        "| Abstention | answerable | example.in_memory | 1 | 2/3 | 0.6667 |" in rendered
    )


def test_render_long_mem_eval_metrics_summary_empty_without_runs() -> None:
    scenario = _scenario()

    rendered = render_long_mem_eval_metrics_summary(
        [scenario],
        {},
        ["example.in_memory"],
    )

    assert rendered == []


def _stub_scenario(scenario: Scenario) -> SimpleNamespace:
    return SimpleNamespace(
        id=scenario.id,
        name=scenario.id,
        memory_systems=[
            SimpleNamespace(type=memory.type) for memory in scenario.memory_systems
        ],
        evaluators=[],
    )


def _round_trip(results: dict[str, list[RunResult]]) -> dict[str, list[RunResult]]:
    reloaded: dict[str, list[RunResult]] = {}
    for runs in results.values():
        for run in runs:
            restored = RunResult.model_validate_json(run.model_dump_json())
            reloaded.setdefault(restored.scenario_id, []).append(restored)
    return reloaded


def test_report_sections_survive_disk_round_trip() -> None:
    scenario = Scenario(
        id="lme-1",
        name="lme-1",
        execution_sequence=[UserStimulus(content="q")],
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
            is_successful=False,
            accuracy_score=0.5,
            passed_keys=["q0:qa1:single-session-user", "q1:qa2:temporal-reasoning"],
            failed_keys=[
                "q2:qa3:single-session-user",
                "q3:qa4:multi-session:abstention",
            ],
            audit_notes="stub",
        ),
    )
    results = {scenario.id: [run]}
    memory_labels = ["milamem"]

    live = render_long_mem_eval_metrics_summary([scenario], results, memory_labels)
    disk = render_long_mem_eval_metrics_summary(
        [_stub_scenario(scenario)],  # type: ignore[list-item]
        _round_trip(results),
        memory_labels,
    )

    assert live
    assert disk == live
