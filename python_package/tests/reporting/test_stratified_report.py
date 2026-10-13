"""Tests for generic tagged-record report aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from heams.evaluation.schemas import AuditRecord
from heams.memory_systems import InContextConfig
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.reporting.stratified import (
    ReportCategory,
    ReportSpec,
    ReportStratifier,
    build_report,
    render_report_sections,
    summarize_run_records,
)
from heams.results import RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


@dataclass(frozen=True)
class ObjectRecord:
    passed: bool
    tags: dict[str, str]


SPEC = ReportSpec(
    summary_title="Generic metrics per memory backend",
    stratified_title="Generic metrics by stratum",
    category_tag="category",
    categories=[
        ReportCategory("required", "Required pass rate"),
        ReportCategory("leak", "Leak failures", "failures"),
    ],
    stratifiers=[
        ReportStratifier("scenario_type", "Scenario type"),
        ReportStratifier("domain", "Domain"),
    ],
)


def _scenario(scenario_id: str, memory_count: int = 1) -> Scenario:
    memories = [InContextConfig()]
    if memory_count > 1:
        memories.append(ExampleMemoryConfig())
    return Scenario(
        id=scenario_id,
        name=scenario_id,
        execution_sequence=[UserStimulus(content="audit")],
        agents=[],
        memory_systems=memories,
        evaluators=[],
    )


def _run(scenario: Scenario, *, memory_index: int = 0) -> RunResult:
    return RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=memory_index,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            audit_notes="stub",
        ),
    )


def test_summarize_run_records_accepts_objects_and_mappings() -> None:
    summary = summarize_run_records(
        [
            ObjectRecord(
                passed=True,
                tags={
                    "category": "required",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            ),
            {
                "passed": False,
                "tags": {
                    "category": "leak",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            },
        ],
        SPEC,
    )

    assert summary is not None
    assert summary.overall.rate == 0.5
    assert summary.categories["required"].rate == 1.0
    assert summary.categories["leak"].failures == 1
    assert summary.tags["scenario_type"] == "direct"


def test_summarize_run_records_treats_null_tags_as_empty() -> None:
    summary = summarize_run_records([{"passed": True, "tags": None}], SPEC)

    assert summary is not None
    assert summary.overall.rate == 1.0
    assert summary.tags == {}
    assert summary.strata["scenario_type"]["unspecified"].overall.total == 1


def test_build_report_aggregates_micro_macro_and_categories() -> None:
    direct = _scenario("direct")
    negative = _scenario("negative")
    scenarios = [direct, negative]
    direct_run = _run(direct)
    negative_run = _run(negative)
    records_by_scenario = {
        "direct": [
            ObjectRecord(
                passed=True,
                tags={
                    "category": "required",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            ),
            ObjectRecord(
                passed=False,
                tags={
                    "category": "leak",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            ),
        ],
        "negative": [
            ObjectRecord(
                passed=True,
                tags={
                    "category": "leak",
                    "scenario_type": "negative",
                    "domain": "work",
                },
            )
        ],
    }

    report = build_report(
        scenarios,
        {direct.id: [direct_run], negative.id: [negative_run]},
        records_of=lambda _scenario, run: records_by_scenario[run.scenario_id],
        spec=SPEC,
    )
    tally = report.by_memory["in_context"]

    assert tally.runs == 2
    assert tally.records_passed == 2
    assert tally.records_total == 3
    assert tally.micro_accuracy == 2 / 3
    assert tally.category_rate("required") == 1.0
    assert tally.category_failures("leak") == 1
    assert report.macro_accuracy("in_context", "scenario_type") == (0.5 + 1.0) / 2


def test_build_report_keeps_strata_per_record_within_one_run() -> None:
    scenario = _scenario("mixed")
    run = _run(scenario)

    report = build_report(
        [scenario],
        {scenario.id: [run]},
        records_of=lambda _scenario, _run: [
            ObjectRecord(
                passed=True,
                tags={
                    "category": "required",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            ),
            ObjectRecord(
                passed=False,
                tags={
                    "category": "leak",
                    "scenario_type": "negative",
                    "domain": "work",
                },
            ),
        ],
        spec=SPEC,
    )

    scenario_type = report.by_memory_stratifier["in_context"]["scenario_type"]
    assert scenario_type["direct"].runs == 1
    assert scenario_type["direct"].records_passed == 1
    assert scenario_type["direct"].records_total == 1
    assert scenario_type["negative"].runs == 1
    assert scenario_type["negative"].records_passed == 0
    assert scenario_type["negative"].records_total == 1
    assert report.macro_accuracy("in_context", "scenario_type") == 0.5


def test_render_report_sections_includes_missing_memory_cells() -> None:
    scenario = _scenario("direct", memory_count=2)
    run = _run(scenario, memory_index=0)
    report = build_report(
        [scenario],
        {scenario.id: [run]},
        records_of=lambda _scenario, _run: [
            ObjectRecord(
                passed=True,
                tags={
                    "category": "required",
                    "scenario_type": "direct",
                    "domain": "health",
                },
            )
        ],
        spec=SPEC,
    )

    rendered = "\n".join(
        render_report_sections(
            report,
            memory_labels=["in_context", "example.in_memory"],
        )
    )

    assert "**Generic metrics per memory backend:**" in rendered
    assert "| Micro accuracy | 1.0000 | na |" in rendered
    assert "| Required pass rate | 1.0000 | na |" in rendered
    assert "| Leak failures | 0 | 0 |" in rendered
    assert "**Generic metrics by stratum:**" in rendered
    assert "| Scenario type | direct | in_context | 1 | 1/1 | 1.0000 |" in rendered
    assert "| Scenario type | direct | example.in_memory |" not in rendered
