"""Artifact-only EE rollup tests (read runs.jsonl -> _rollup/ee/report.md)."""

from __future__ import annotations

from pathlib import Path

import yaml

from heams.benchmarks.erasability_editability.report import (
    render_ee_trivial_pass_sections,
    write_ee_rollup_report,
)
from heams.evaluation.schemas import AuditRecord, EvaluationRecord
from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports
from heams.results import RunResult

# (task_type, ordinal, entity, hop, before_passed, after_passed)
_PROBES = [
    ("edit", 0, "medication", "1", True, True),  # learned, cascaded  -> real
    ("edit", 1, "gym", "2", True, False),  # learned, missed    -> knew_but_failed
    ("erase", 0, "partner", "n/a", False, False),  # never learned      -> never_knew
]


def _record(
    scenario_id: str,
    task_type: str,
    ordinal: int,
    entity: str,
    hop: str,
    phase: str,
    passed: bool,
    pass_type: str | None,
) -> EvaluationRecord:
    tags = {
        "task_type": task_type,
        "task_type_label": "editability" if task_type == "edit" else "erasability",
        "phase": phase,
        "task_phase": f"{task_type}/{phase}",
        "hop": hop,
        "domain": "personal_life",
        "entity": entity,
    }
    if pass_type is not None:
        tags["pass_type"] = pass_type
    return EvaluationRecord(
        key=f"{scenario_id}:{task_type}:{phase}{ordinal + 1}",
        passed=passed,
        score=1.0 if passed else 0.0,
        tags=tags,
        reason="ok" if passed else "missed",
    )


def _classify(before: bool, after: bool) -> str:
    if after and before:
        return "real"
    if after:
        return "trivial"
    return "knew_but_failed" if before else "never_knew"


def _ee_run(scenario_id: str, probes=_PROBES) -> RunResult:
    records = [
        _record(scenario_id, t, i, e, h, "before", before, None)
        for t, i, e, h, before, _a in probes
    ] + [
        _record(scenario_id, t, i, e, h, "after", after, _classify(before, after))
        for t, i, e, h, before, after in probes
    ]
    passed_keys = [r.key for r in records if r.passed]
    failed_keys = [r.key for r in records if not r.passed]
    real = sum(1 for r in records if r.tags.get("pass_type") == "real")
    after_total = sum(1 for r in records if r.tags["phase"] == "after")
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        audit=AuditRecord(
            is_successful=False,
            accuracy_score=real / after_total,
            passed_keys=passed_keys,
            failed_keys=failed_keys,
            audit_notes="stub",
            report_records=records,
        ),
    )


def _scenario_stub(scenario_id: str):
    """Minimal Scenario carrying the one memory system the runs point at."""
    from heams.memory_systems.factory import MemorySystemVariant
    from heams.scenarios.actions import UserStimulus
    from heams.scenarios.base import Scenario

    return Scenario(
        id=scenario_id,
        name=scenario_id,
        execution_sequence=[UserStimulus(content="q")],
        memory_systems=[
            MemorySystemVariant.model_validate(
                {
                    "id": "example.in_memory",
                    "backend_config": {
                        "type": "example.in_memory",
                        "name": "example.in_memory",
                        "top_k": 10,
                    },
                }
            )
        ],
    )


def _write_ee_experiment(results_dir: Path, runs=None) -> None:
    exp = results_dir / "gpt4o_mini_example_memory_ee_run"
    exp.mkdir(parents=True)
    (exp / "experiment_config.run.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_config": {"type": "ee"},
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
    runs = runs if runs is not None else [_ee_run("pl_001"), _ee_run("sp_001")]
    (exp / "runs.jsonl").write_text(
        "\n".join(r.model_dump_json() for r in runs) + "\n", encoding="utf-8"
    )


def _rows(lines: list[str]) -> set[tuple[str, ...]]:
    return {
        tuple(cell.strip() for cell in line.strip("|").split("|"))
        for line in lines
        if line.startswith("| ") and "---" not in line
    }


def test_ee_rollup_written_with_trivial_pass_and_raw_tables(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_ee_experiment(results_dir)

    written = write_benchmark_rollup_reports(results_dir)

    assert "ee" in written  # dispatcher registration works
    report = written["ee"].read_text(encoding="utf-8")
    assert "# ee benchmark" in report

    # Headline table, and it comes first.
    assert "EE trivial-pass filtered score" in report
    assert report.index("EE trivial-pass filtered score") < report.index(
        "EE raw judge pass rate"
    )
    # Raw table is explicitly labelled as unfiltered.
    assert "without the trivial-pass filter" in report
    for label in ("edit/before", "edit/after", "erase/before", "erase/after"):
        assert label in report
    assert "Cascade hop" in report

    # Headline: one row per backend, one column per task type.
    assert "| Memory backend | edit_accuracy | erase_accuracy |" in report
    assert report.index("edit_accuracy") < report.index("| Memory backend | Task |")

    # Every bucket is explained in a legend, from the same mapping the records
    # are tagged from — so the report and the raw records cannot drift.
    from heams.benchmarks.erasability_editability.judge import EE_PASS_TYPE_LABELS

    assert "| Bucket | Meaning | Counts toward score |" in report
    for name, meaning in EE_PASS_TYPE_LABELS.items():
        assert f"| `{name}` | {meaning} |" in report


def test_trivial_pass_table_counts_each_bucket() -> None:
    results = {"pl_001": [_ee_run("pl_001")]}
    rows = _rows(render_ee_trivial_pass_sections([_scenario_stub("pl_001")], results))

    # edit: one real, one knew_but_failed -> 1/2.
    assert (
        "example.in_memory",
        "edit",
        "50.0% (1/2)",
        "1",
        "0",
        "1",
        "0",
    ) in rows
    # erase: never learned it, so never_knew -> 0/1.
    assert (
        "example.in_memory",
        "erase",
        "0.0% (0/1)",
        "0",
        "0",
        "0",
        "1",
    ) in rows


def test_no_memory_style_run_scores_zero_despite_passing_deletion() -> None:
    """The whole point of the filter: a system that knows nothing scores 0."""
    # Every before-probe fails, every after-probe "passes" -> all trivial.
    probes = [
        ("erase", 0, "partner", "n/a", False, True),
        ("erase", 1, "allergy", "n/a", False, True),
    ]
    run = _ee_run("pl_001", probes=probes)
    results = {"pl_001": [run]}

    rows = _rows(render_ee_trivial_pass_sections([_scenario_stub("pl_001")], results))
    assert (
        "example.in_memory",
        "erase",
        "0.0% (0/2)",
        "0",
        "2",
        "0",
        "0",
    ) in rows

    # The raw judge pass rate would have said 100% — that is the trap.
    assert all(r.passed for r in run.audit.report_records if r.tags["phase"] == "after")
    assert run.audit.accuracy_score == 0.0


def test_trivial_pass_section_empty_without_pass_type_tags() -> None:
    """Older artifacts carry no pass_type; the section degrades quietly."""
    run = _ee_run("pl_001")
    for record in run.audit.report_records:
        record.tags.pop("pass_type", None)

    assert (
        render_ee_trivial_pass_sections([_scenario_stub("pl_001")], {"pl_001": [run]})
        == []
    )


def test_write_ee_rollup_report_direct(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_ee_experiment(results_dir)

    path = write_ee_rollup_report(results_dir)

    assert path is not None
    assert path == results_dir / "_rollup" / "ee" / "report.md"
    assert path.is_file()
    # The direct writer supplies the friendly title.
    assert "# erasability & editability benchmark" in path.read_text(encoding="utf-8")


def test_ee_rollup_absent_without_ee_experiments(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    assert write_ee_rollup_report(results_dir) is None
    assert "ee" not in write_benchmark_rollup_reports(results_dir)
