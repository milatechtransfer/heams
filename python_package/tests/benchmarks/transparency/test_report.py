"""Tests for transparency aggregate reporting in the benchmark report."""

from __future__ import annotations

from pathlib import Path

import yaml
from heams_memory_example.plugin import ExampleMemoryConfig

from heams.benchmarks.transparency import build_transparency_report
from heams.benchmarks.transparency.report import render_transparency_metrics_summary
from heams.evaluation.factory import TransparencyLlmJudgeConfig
from heams.evaluation.scenarios.transparency_llm_judge import (
    build_transparency_rubric_items,
    transparency_qa_key,
)
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams.memory_systems import MemorySystemVariant
from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports
from heams.results import AgentTurn, RunResult
from heams.scenarios.actions import UserStimulus
from heams.scenarios.base import Scenario


def _transparency_scenario(
    *,
    scenario_id: str,
    question_id: str,
    scenario_type: str,
    privacy_domain: str,
    persona_id: str,
    difficulty: str,
    expected_facts: list[str],
    prohibited_facts: list[str],
) -> Scenario:
    return Scenario(
        id=scenario_id,
        name=scenario_id,
        execution_sequence=[
            UserStimulus(
                content="audit question",
                step_evaluation_markers={
                    "question_id": question_id,
                    "question": "audit question",
                    "scenario_type": scenario_type,
                    "privacy_domain": privacy_domain,
                    "persona": {"id": persona_id},
                    "difficulty": {"level": difficulty},
                    "expected_facts": expected_facts,
                    "prohibited_facts": prohibited_facts,
                    "retrieval_evidence": {"relevant_contents": expected_facts},
                    "rubric": "stub rubric",
                },
            )
        ],
        agents=[],
        memory_systems=[
            MemorySystemVariant(
                id="example.in_memory",
                backend_config=ExampleMemoryConfig(),
            )
        ],
        evaluators=[TransparencyLlmJudgeConfig()],
    )


def _audit_keys(scenario: Scenario) -> dict[str, str]:
    ground_truth = TransparencyLlmJudgeConfig().to_ground_truth(scenario)
    keys: dict[str, str] = {}
    for index, qa in enumerate(ground_truth.qa):
        prefix = transparency_qa_key(index, qa)
        for criterion in build_transparency_rubric_items(qa):
            keys[criterion.key] = f"{prefix}:{criterion.key}"
    return keys


def _run(scenario: Scenario, *, passed: list[str], failed: list[str]) -> RunResult:
    keys = _audit_keys(scenario)
    records = _report_records(scenario, passed=passed, failed=failed)
    return RunResult(
        scenario_id=scenario.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=not failed,
            accuracy_score=len(passed) / max(len(passed) + len(failed), 1),
            passed_keys=[keys[key] for key in passed],
            failed_keys=[keys[key] for key in failed],
            audit_notes="stub",
            report_records=records,
        ),
        transcript=[AgentTurn(question="audit question", answer="answer")],
    )


def _report_records(
    scenario: Scenario, *, passed: list[str], failed: list[str]
) -> list[EvaluationRecord]:
    passed_set = set(passed)
    failed_set = set(failed)
    records: list[EvaluationRecord] = []
    ground_truth = TransparencyLlmJudgeConfig().to_ground_truth(scenario)
    for index, qa in enumerate(ground_truth.qa):
        prefix = transparency_qa_key(index, qa)
        for criterion in build_transparency_rubric_items(qa):
            if criterion.key not in passed_set and criterion.key not in failed_set:
                continue
            is_passed = criterion.key in passed_set
            records.append(
                EvaluationRecord(
                    key=f"{prefix}:{criterion.key}",
                    passed=is_passed,
                    score=1.0 if is_passed else 0.0,
                    tags={
                        "question_id": qa.question_id,
                        "scenario_type": qa.scenario_type,
                        "criterion_type": criterion.criterion_type,
                        "dimension": criterion.dimension,
                        "transparency_metric": criterion.transparency_metric,
                        "privacy_domain": qa.privacy_domain or "unspecified",
                        "persona_id": str(qa.persona.get("id", "unspecified")),
                        "difficulty_level": str(
                            qa.difficulty.get("level", "unspecified")
                        ),
                    },
                    expected=criterion.rubric,
                    observed="answer",
                )
            )
    return records


def _direct_inspection_scenario() -> Scenario:
    return _transparency_scenario(
        scenario_id="TRANS-DIRECT",
        question_id="TRANS-DIRECT-001",
        scenario_type="direct_inspection",
        privacy_domain="health",
        persona_id="persona_a",
        difficulty="easy",
        expected_facts=["Allergic to penicillin"],
        prohibited_facts=["Salary is 100k"],
    )


def _negative_audit_scenario() -> Scenario:
    return _transparency_scenario(
        scenario_id="TRANS-NEG",
        question_id="TRANS-NEG-001",
        scenario_type="negative_audit",
        privacy_domain="work",
        persona_id="persona_b",
        difficulty="hard",
        expected_facts=[],
        prohibited_facts=["Has a dog"],
    )


def _direct_results(direct: Scenario) -> list[RunResult]:
    return [
        _run(
            direct,
            passed=["scenario:direct_inspection", "completeness:stored_inventory"],
            failed=["data_minimization:prohibited_facts"],
        )
    ]


def _negative_results(negative: Scenario) -> list[RunResult]:
    return [
        _run(
            negative,
            passed=["data_minimization:prohibited_facts"],
            failed=["scenario:negative_audit"],
        )
    ]


def test_build_transparency_report_classifies_categories() -> None:
    direct = _direct_inspection_scenario()
    negative = _negative_audit_scenario()
    scenarios = [direct, negative]
    results = {
        direct.id: _direct_results(direct),
        negative.id: _negative_results(negative),
    }

    report = build_transparency_report(scenarios, results)
    tally = report.by_memory["example.in_memory"]

    assert tally.runs == 2
    assert tally.records_passed == 3
    assert tally.records_total == 5
    assert tally.micro_accuracy == 3 / 5
    assert tally.category_rate("required_completeness") == 1.0
    assert tally.category_failures("over_disclosure") == 1
    assert tally.category_failures("hallucination_negative") == 1
    assert (
        report.macro_accuracy("example.in_memory", "scenario_type")
        == ((2 / 3) + 0.5) / 2
    )


def test_build_transparency_report_counts_unknown_legacy_audit_keys() -> None:
    direct = _direct_inspection_scenario()
    run = RunResult(
        scenario_id=direct.id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=0.5,
            passed_keys=["legacy:extra_pass"],
            failed_keys=["legacy:extra_fail"],
            audit_notes="legacy audit",
        ),
        transcript=[AgentTurn(question="audit question", answer="answer")],
    )

    report = build_transparency_report([direct], {direct.id: [run]})
    tally = report.by_memory["example.in_memory"]

    # Legacy runs without serialized report_records still contribute to micro
    # accuracy from the audit keys, but carry no ground-truth-derived stratum
    # tags, so every stratifier collapses to "unspecified".
    assert tally.records_passed == 1
    assert tally.records_total == 2
    assert tally.micro_accuracy == 0.5
    assert tally.category_rate("required_completeness") is None
    assert tally.category_failures("over_disclosure") == 0
    scenario_type = report.by_memory_stratifier["example.in_memory"]["scenario_type"]
    assert scenario_type["unspecified"].records_total == 2


def test_build_transparency_report_prefers_memory_entry_id() -> None:
    scenario = _direct_inspection_scenario().model_copy(
        update={
            "memory_systems": [
                MemorySystemVariant(
                    id="deterministic_rp",
                    backend_config=ExampleMemoryConfig(),
                )
            ]
        }
    )
    results = {scenario.id: _direct_results(scenario)}

    report = build_transparency_report([scenario], results)

    assert "deterministic_rp" in report.by_memory


def test_build_transparency_report_falls_back_to_backend_type_without_id() -> None:
    scenario = _direct_inspection_scenario().model_copy(
        update={
            "memory_systems": [
                MemorySystemVariant(
                    backend_config=ExampleMemoryConfig(),
                    id=None,
                )
            ]
        }
    )
    results = {scenario.id: _direct_results(scenario)}

    report = build_transparency_report([scenario], results)

    assert "example.in_memory" in report.by_memory


def test_render_transparency_metrics_summary_includes_sections() -> None:
    direct = _direct_inspection_scenario()
    negative = _negative_audit_scenario()
    scenarios = [direct, negative]
    results = {
        direct.id: _direct_results(direct),
        negative.id: _negative_results(negative),
    }

    report = "\n".join(
        render_transparency_metrics_summary(
            scenarios,
            results,
            ["basic_complete_memory_system"],
        )
    )

    assert "**Transparency metrics per memory backend:**" in report
    assert "| Micro accuracy | 0.6000 |" in report
    assert "| Macro accuracy (scenario type) | 0.5833 |" in report
    assert "| Macro accuracy (privacy domain) | 0.5833 |" in report
    assert "| Required/completeness pass rate | 1.0000 |" in report
    assert "| Over-disclosure failures | 1 |" in report
    assert "| Hallucination/negative-audit failures | 1 |" in report

    assert "**Transparency metrics by stratum:**" in report
    assert "| Privacy domain | health | example.in_memory |" in report
    assert "| Privacy domain | work | example.in_memory |" in report
    assert "| Scenario type | direct_inspection | example.in_memory |" in report
    assert "| Scenario type | negative_audit | example.in_memory |" in report
    assert "| Persona | persona_a | example.in_memory |" in report
    assert "| Difficulty | hard | example.in_memory |" in report


def test_write_transparency_rollup_filters_other_benchmarks(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    locomo_dir = results_dir / "locomo_exp"
    transparency_dir = results_dir / "transparency_exp"
    locomo_dir.mkdir(parents=True)
    transparency_dir.mkdir(parents=True)

    # Locomo experiment without runs.jsonl is ignored by the rollup scanner.
    (locomo_dir / "experiment_config.run.yaml").write_text(
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

    (transparency_dir / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "transparency"},
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
        scenario_id="TRANS-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=True,
            accuracy_score=1.0,
            passed_keys=["q0:ok"],
            failed_keys=[],
            audit_notes="stub",
            report_records=[
                EvaluationRecord(
                    key="q0:ok",
                    passed=True,
                    score=1.0,
                    tags={
                        "dimension": "completeness",
                        "scenario_type": "direct_inspection",
                        "privacy_domain": "health",
                        "persona_id": "persona_a",
                        "difficulty_level": "easy",
                        "category": "required_completeness",
                    },
                    expected="fact",
                    observed="fact",
                    reason="ok",
                )
            ],
        ),
    )
    (transparency_dir / "runs.jsonl").write_text(
        run.model_dump_json() + "\n",
        encoding="utf-8",
    )

    written = write_benchmark_rollup_reports(results_dir)
    assert "transparency" in written
    assert "locomo" not in written
    report = written["transparency"].read_text(encoding="utf-8")
    assert "# transparency benchmark" in report
    assert "Transparency metrics" in report
