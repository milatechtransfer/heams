"""Per-benchmark disk-equivalence tests for stratified report sections.

Each benchmark's rollup ``render_*_metrics_summary`` must render identically
whether built from live scenarios or from a ``runs.jsonl`` round-trip paired
with stub scenarios (id + memory backend labels only). This guards the Option-A
contract that the cross-memory rollup rebuilds every report from disk without
reloading any dataset or ground truth.
"""

from __future__ import annotations

from types import SimpleNamespace

from heams.benchmarks.locomo.report import render_locomo_metrics_summary
from heams.benchmarks.transparency.report import render_transparency_metrics_summary
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.results import RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


def _scenario(scenario_id: str, *, evaluators: list[object] | None = None) -> Scenario:
    return Scenario(
        id=scenario_id,
        name=scenario_id,
        execution_sequence=[UserStimulus(content="q")],
        agents=[],
        memory_systems=[ExampleMemoryConfig()],
        evaluators=evaluators or [],
    )


def _stub_scenario(scenario: Scenario) -> SimpleNamespace:
    """Mirror the rollup stub: id + memory backend labels, no evaluators."""
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


def _assert_disk_equivalent(
    render,
    scenarios: list[Scenario],
    results: dict[str, list[RunResult]],
    memory_labels: list[str],
) -> None:
    live = render(scenarios, results, memory_labels)
    disk = render(
        [_stub_scenario(scenario) for scenario in scenarios],
        _round_trip(results),
        memory_labels,
    )

    assert live, "expected non-empty live report sections"
    assert disk == live


def test_locomo_report_sections_survive_disk_round_trip() -> None:
    scenario = _scenario("conv-1")
    run = RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=0.6,
            passed_keys=["q0:1", "q1:4", "q2:2"],
            failed_keys=["q3:1", "q4:3"],
            audit_notes="stub",
        ),
    )
    _assert_disk_equivalent(
        render_locomo_metrics_summary,
        [scenario],
        {scenario.id: [run]},
        ["milamem"],
    )


def _transparency_record(
    *, key: str, passed: bool, dimension: str, scenario_type: str
) -> EvaluationRecord:
    return EvaluationRecord(
        key=key,
        passed=passed,
        score=1.0 if passed else 0.0,
        tags={
            "dimension": dimension,
            "scenario_type": scenario_type,
            "privacy_domain": "health",
            "persona_id": "persona_a",
            "difficulty_level": "easy",
        },
        expected="rubric",
        observed="answer",
    )


def test_transparency_report_sections_survive_disk_round_trip() -> None:
    scenario = _scenario("TRANS-1")
    run = RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=0.5,
            passed_keys=["q0:TRANS-1:direct_inspection:completeness:x"],
            failed_keys=["q0:TRANS-1:direct_inspection:data_minimization:y"],
            audit_notes="stub",
            report_records=[
                _transparency_record(
                    key="q0:TRANS-1:direct_inspection:completeness:x",
                    passed=True,
                    dimension="completeness",
                    scenario_type="direct_inspection",
                ),
                _transparency_record(
                    key="q0:TRANS-1:direct_inspection:data_minimization:y",
                    passed=False,
                    dimension="data_minimization",
                    scenario_type="direct_inspection",
                ),
            ],
        ),
    )
    _assert_disk_equivalent(
        render_transparency_metrics_summary,
        [scenario],
        {scenario.id: [run]},
        ["milamem"],
    )
