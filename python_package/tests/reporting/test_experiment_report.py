"""Tests for the `heams generate-report` per-experiment report builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from heams.evaluation.schemas import AuditRecord, EvaluationRecord, SystemMetricsSummary
from heams.execution_traces import ExecutionTraceEvent, ExecutionTraceRecord
from heams.reporting.experiment_report import (
    discover_experiment_dirs,
    generate_experiment_reports,
    load_run_results,
    render_experiment_report,
)
from heams.results import AgentTurn, RunResult, ToolResult


def _run(
    *,
    scenario_id: str = "conv-1",
    sample_index: int = 0,
    is_successful: bool = True,
    accuracy_score: float = 1.0,
    failed_keys: list[str] | None = None,
    transcript: list[AgentTurn] | None = None,
    report_records: list[EvaluationRecord] | None = None,
    trace_id: str | None = None,
) -> RunResult:
    return RunResult(
        scenario_id=scenario_id,
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=sample_index,
        trace_id=trace_id,
        audit=AuditRecord(
            is_successful=is_successful,
            accuracy_score=accuracy_score,
            passed_keys=[] if failed_keys else ["q0"],
            failed_keys=failed_keys or [],
            audit_notes="stub audit",
            report_records=report_records,
        ),
        transcript=transcript or [],
    )


def _trace(trace_id: str = "trace-1") -> ExecutionTraceRecord:
    return ExecutionTraceRecord(
        trace_id=trace_id,
        run_id="run-1",
        scenario_id="conv-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
        outcome="completed",
        events=[
            ExecutionTraceEvent(
                sequence_index=0,
                phase="priming",
                type="memory_priming",
                name="prime_memory_entries",
                content="2 entries",
                start_offset_ms=0,
                duration_ms=2,
                status="ok",
            ),
            ExecutionTraceEvent(
                sequence_index=1,
                phase="execution",
                type="tool_call",
                name="recall",
                content="first recalled fact",
                start_offset_ms=3,
                duration_ms=4,
                status="ok",
            ),
            ExecutionTraceEvent(
                sequence_index=2,
                phase="execution",
                type="tool_call",
                name="recall",
                content="second recalled fact",
                start_offset_ms=8,
                duration_ms=5,
                status="ok",
            ),
        ],
    )


def test_load_run_results_skips_malformed_lines(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "runs.jsonl"
    good_run = _run()
    jsonl_path.write_text(
        good_run.model_dump_json() + "\n" + "{not valid json\n" + "\n",
        encoding="utf-8",
    )

    results = load_run_results(jsonl_path)

    assert len(results) == 1
    assert results[0].scenario_id == "conv-1"


def test_load_run_results_preserves_unicode_line_separators(tmp_path: Path) -> None:
    """Regression test for #582: U+2028/U+2029/U+0085 inside a JSON string
    value must not be treated as a JSONL record boundary."""
    jsonl_path = tmp_path / "runs.jsonl"
    separator = chr(0x2028) + chr(0x2029) + chr(0x85)
    notes_with_separators = "line one" + separator + "line two"
    good_run = _run()
    run_with_separators = good_run.model_copy(
        update={
            "audit": good_run.audit.model_copy(
                update={"audit_notes": notes_with_separators}
            )
        }
    )
    jsonl_path.write_bytes(
        (run_with_separators.model_dump_json() + "\n").encode("utf-8")
    )

    results = load_run_results(jsonl_path)

    assert len(results) == 1
    assert results[0].audit.audit_notes == notes_with_separators


def test_discover_experiment_dirs_requires_runs_jsonl(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a_run1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "runs.jsonl").write_text("", encoding="utf-8")

    (results_dir / "not_an_experiment").mkdir()
    (results_dir / "_rollup").mkdir()

    discovered = discover_experiment_dirs(results_dir)

    assert discovered == [experiment_dir]


def test_discover_experiment_dirs_excludes_rollup_even_with_a_stray_runs_jsonl(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a_run1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "runs.jsonl").write_text("", encoding="utf-8")

    # `_rollup` is a reserved, benchmark_rollup-owned directory name; it must
    # never be treated as an experiment even if something under it happens to
    # have its own `runs.jsonl`.
    rollup_dir = results_dir / "_rollup"
    rollup_dir.mkdir()
    (rollup_dir / "runs.jsonl").write_text("", encoding="utf-8")

    discovered = discover_experiment_dirs(results_dir)

    assert discovered == [experiment_dir]


def test_render_experiment_report_includes_top_level_metrics_and_detail() -> None:
    run = _run(
        transcript=[
            AgentTurn(
                question="Where did Caroline go?",
                answer="The LGBTQ support group.",
                target_answer="LGBTQ support group",
                tool_results=[
                    ToolResult(
                        tool_name="recall",
                        arguments={"query": "Caroline"},
                        output=["fact"],
                    )
                ],
            )
        ],
        report_records=[
            EvaluationRecord(
                key="q0",
                passed=True,
                expected="LGBTQ support group",
                observed="The LGBTQ support group.",
                reason="Semantically equivalent.",
            )
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "# Experiment report: exp_a" in report
    assert "## Top-level metrics" in report
    assert "- Runs: 1" in report
    assert "- Successful runs: 1 / 1" in report
    assert "## Summary" in report
    assert "| conv-1 | 0 | 0 | True | 1.00 | - |" in report
    assert "## Detail" in report
    assert "### conv-1 / sample 0 / evaluator 0" in report
    assert "**Q0:** Where did Caroline go?" in report
    assert "**A0:** The LGBTQ support group." in report
    assert "**Target0:** LGBTQ support group" in report
    assert "**LLM judge output:**" in report
    assert (
        "| q0 | True | na | LGBTQ support group | The LGBTQ support group. | Semantically equivalent. |"
        in report
    )


def test_render_experiment_report_includes_execution_sequence_and_counts() -> None:
    run = _run(trace_id="trace-1")

    report = render_experiment_report(
        "exp_a", [run], traces_by_id={"trace-1": _trace()}
    )

    assert "**Execution trace:**" in report
    assert "- Event types: `memory_priming` × 1, `tool_call` × 2" in report
    assert "- Tools: `recall` × 2" in report
    assert (
        "1. `priming` / `memory_priming` / `prime_memory_entries` "
        "— ok, 2.00 ms — 2 entries" in report
    )
    assert (
        "3. `execution` / `tool_call` / `recall` "
        "— ok, 5.00 ms — second recalled fact" in report
    )


def test_render_experiment_report_prints_trace_efficiency_rows() -> None:
    run = _run().model_copy(
        update={
            "system_metrics": SystemMetricsSummary(
                llm_input_tokens=10,
                llm_output_tokens=4,
                llm_input_cost=0.000002,
                llm_output_cost=0.000004,
                llm_total_cost=0.000006,
                agent_duration_ms=250,
                llm_duration_ms=2,
                tool_execution_count=2,
                tool_execution_error_count=1,
                tool_duration_ms=4,
            )
        }
    )

    report = render_experiment_report("exp_a", [run])

    assert "| LLM input tokens | 10.00 |" in report
    assert "| LLM output tokens | 4.00 |" in report
    assert "| LLM input cost | 0.000002 |" in report
    assert "| LLM output cost | 0.000004 |" in report
    assert "| LLM duration (ms) | 2.00 |" in report
    assert "| Agent duration (ms) | 250.00 |" in report
    assert "| Tool executions | 2.00 |" in report
    assert "| Tool execution errors | 1.00 |" in report
    assert "| Tool duration (ms) | 4.00 |" in report


def test_render_experiment_report_aggregates_tagged_sub_metrics() -> None:
    runs = [
        _run(
            scenario_id="conv-1",
            report_records=[
                EvaluationRecord(
                    key="q0",
                    passed=True,
                    score=1.0,
                    tags={"category": "temporal"},
                ),
                EvaluationRecord(
                    key="q1",
                    passed=False,
                    score=0.0,
                    tags={"category": "single-hop"},
                ),
            ],
        ),
        _run(
            scenario_id="conv-2",
            report_records=[
                EvaluationRecord(
                    key="q0",
                    passed=True,
                    score=0.5,
                    tags={"category": "temporal"},
                ),
            ],
        ),
    ]

    report = render_experiment_report("exp_a", runs)

    assert "Mean sub-metrics (category)" in report
    assert "| temporal | 2/2 | 1.0000 | 0.7500 |" in report
    assert "| single-hop | 0/1 | 0.0000 | 0.0000 |" in report
    assert "- Average across values: 0.5000" in report
    assert "- Weakest value: `single-hop` (0.0000)" in report


def test_render_experiment_report_sub_metric_table_omits_summary_for_one_value() -> (
    None
):
    """A tag with only one value has nothing to average or compare, so the
    average/weakest lines would just restate that one number - skip them."""
    run = _run(
        report_records=[
            EvaluationRecord(key="q0", passed=True, tags={"category": "temporal"}),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (category)" in report
    assert "Average across values" not in report
    assert "Weakest value" not in report


def test_render_experiment_report_omits_identifier_like_tag_tables() -> None:
    """A tag with many distinct, ~one-off values (sample_id, question_id,
    response_index) is a raw identifier, not a real category - a per-value
    breakdown table for it would be noise, not information."""
    run = _run(
        report_records=[
            EvaluationRecord(key=f"q{i}", passed=True, tags={"question_id": f"Q{i}"})
            for i in range(10)
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (question_id): omitted" in report
    assert "| Q0 |" not in report


def test_render_experiment_report_keeps_small_category_despite_one_record_each() -> (
    None
):
    """A real category with few values, each backed by only one record in a
    small run (e.g. one attack class per smoke run), must still render - it
    is not distinguishable from an identifier below the distinct-value floor."""
    run = _run(
        report_records=[
            EvaluationRecord(key=f"q{i}", passed=i == 0, tags={"attack_class": value})
            for i, value in enumerate(["explicit", "fact", "skill"])
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (attack_class):" in report
    assert "| explicit | 1/1 | 1.0000 | na |" in report
    assert "omitted" not in report


def test_render_experiment_report_omits_sub_metrics_when_no_tags() -> None:
    run = _run(
        report_records=[
            EvaluationRecord(key="q0", passed=True),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "sub-metric" not in report.lower()


def test_render_experiment_report_folds_label_companion_into_base_table() -> None:
    """`dimension_label` is a display-only companion to `dimension` (STALE's
    machine code vs. human label), not a separate dimension - it must not
    get its own duplicate table, and its nicer text should be what shows up
    in `dimension`'s table."""
    run = _run(
        report_records=[
            EvaluationRecord(
                key="q0",
                passed=True,
                tags={"dimension": "dim1", "dimension_label": "explicit_probing"},
            ),
            EvaluationRecord(
                key="q1",
                passed=False,
                tags={"dimension": "dim2", "dimension_label": "adversarial_robustness"},
            ),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (dimension):" in report
    assert "| explicit_probing | 1/1 | 1.0000 | na |" in report
    assert "| adversarial_robustness | 0/1 | 0.0000 | na |" in report
    assert "dim1" not in report
    assert "Mean sub-metrics (dimension_label):" not in report


def test_render_experiment_report_pass_type_shows_raw_and_real_side_by_side() -> None:
    """A `pass_type` tag opts a benchmark into a combined raw+real table.

    Mirrors the erasability/editability shape: an `edit` task type where
    every raw pass only "succeeded" via a loophole (the fact was never
    known, or the check failed outright) - the real column must show 0%
    right next to the raw 50%, in the same row, not a separate table.
    """
    run = _run(
        report_records=[
            EvaluationRecord(
                key="before1", passed=True, score=1.0, tags={"task_type": "edit"}
            ),
            EvaluationRecord(
                key="before2", passed=False, score=0.0, tags={"task_type": "edit"}
            ),
            EvaluationRecord(
                key="after1",
                passed=False,
                score=0.0,
                tags={"task_type": "edit", "pass_type": "knew_but_failed"},
            ),
            EvaluationRecord(
                key="after2",
                passed=True,
                score=1.0,
                tags={"task_type": "edit", "pass_type": "trivial"},
            ),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (task_type):" in report
    assert "| edit | 2/4 | 0.5000 | 0.0000 (0/2) |" in report
    # No separate "(filtered)" table - the real column above already carries it.
    assert "filtered" not in report.lower()
    # The raw/real distinction is explained once in plain language, not left
    # for the reader to guess from the column headers alone.
    assert report.count("Real is the actual score") == 1


def test_render_experiment_report_record_type_uses_one_table_with_a_column() -> None:
    """A `record_type` tag opts a benchmark out of pooling different check
    kinds under one tag value.

    Mirrors the contextual-integrity shape: a `domain` tagged on both a
    privacy-leak check (passing = good) and a helpfulness check (failing)
    must not be pooled into one meaningless "domain" rate - but the split
    lands in one table with `record_type` as a column, not one table per
    type.
    """
    run = _run(
        report_records=[
            EvaluationRecord(
                key="secret0",
                passed=True,
                tags={"domain": "Hospitality", "record_type": "secret"},
            ),
            EvaluationRecord(
                key="coverage0",
                passed=False,
                tags={"domain": "Hospitality", "record_type": "allowed_coverage"},
            ),
            EvaluationRecord(
                key="helpful0",
                passed=False,
                tags={"domain": "Hospitality", "record_type": "helpfulness"},
            ),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Mean sub-metrics (domain):" in report
    assert "| Hospitality | allowed_coverage | 0/1 | 0.0000 | na |" in report
    assert "| Hospitality | helpfulness | 0/1 | 0.0000 | na |" in report
    assert "| Hospitality | secret | 1/1 | 1.0000 | na |" in report
    # Not split into several separately-headed tables.
    assert "record_type=secret" not in report
    assert "Mean sub-metrics (record_type):" in report
    # The check-type split is explained once in plain language.
    assert report.count("not a combined score") == 1


def test_render_experiment_report_omits_explanatory_notes_without_the_tags() -> None:
    """Neither reserved-tag convention is in play for a plain category tag,
    so neither explanatory note should appear - they'd be noise here."""
    run = _run(
        report_records=[
            EvaluationRecord(key="q0", passed=True, tags={"category": "temporal"}),
        ],
    )

    report = render_experiment_report("exp_a", [run])

    assert "Real is the actual score" not in report
    assert "not a combined score" not in report


def test_render_experiment_report_max_examples_mixes_failures_and_successes() -> None:
    failures = [
        _run(scenario_id=f"fail-{i}", is_successful=False, accuracy_score=0.0)
        for i in range(3)
    ]
    successes = [
        _run(scenario_id=f"pass-{i}", is_successful=True, accuracy_score=1.0)
        for i in range(3)
    ]

    report = render_experiment_report("exp_a", failures + successes, max_examples=2)

    assert "- Runs: 6" in report
    # Summary always lists every run, capped or not.
    for run in failures + successes:
        assert f"| {run.scenario_id} |" in report
    assert "Showing full detail for 2 of 6 runs (1 failed, 1 successful)" in report
    detail_section = report.split("## Detail", 1)[1]
    assert sum(detail_section.count(f"### fail-{i}") for i in range(3)) == 1
    assert sum(detail_section.count(f"### pass-{i}") for i in range(3)) == 1


def test_render_experiment_report_max_examples_backfills_from_other_outcome() -> None:
    failures = [
        _run(scenario_id=f"fail-{i}", is_successful=False, accuracy_score=0.0)
        for i in range(1)
    ]
    successes = [
        _run(scenario_id=f"pass-{i}", is_successful=True, accuracy_score=1.0)
        for i in range(5)
    ]

    report = render_experiment_report("exp_a", failures + successes, max_examples=3)

    assert "Showing full detail for 3 of 6 runs (1 failed, 2 successful)" in report


def test_render_experiment_report_no_cap_by_default() -> None:
    results = [
        _run(scenario_id=f"conv-{i}", is_successful=(i % 2 == 0)) for i in range(20)
    ]

    report = render_experiment_report("exp_a", results)

    assert "Showing full detail for" not in report
    for result in results:
        assert f"### {result.scenario_id}" in report


def test_render_experiment_report_handles_no_results() -> None:
    report = render_experiment_report("exp_empty", [])

    assert "# Experiment report: exp_empty" in report
    assert "No run results found." in report


def test_generate_experiment_reports_writes_one_report_per_experiment(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    exp_a = results_dir / "exp_a_run1"
    exp_b = results_dir / "exp_b_run1"
    exp_a.mkdir(parents=True)
    exp_b.mkdir(parents=True)

    (exp_a / "runs.jsonl").write_text(
        _run(scenario_id="conv-1").model_dump_json() + "\n", encoding="utf-8"
    )
    (exp_b / "runs.jsonl").write_text(
        _run(
            scenario_id="conv-2", is_successful=False, accuracy_score=0.0
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    written = generate_experiment_reports(results_dir)

    assert written == [exp_a / "report.md", exp_b / "report.md"]
    assert "conv-1" in (exp_a / "report.md").read_text(encoding="utf-8")
    assert "conv-2" in (exp_b / "report.md").read_text(encoding="utf-8")


def test_generate_experiment_reports_joins_execution_trace_sidecar(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    experiment_dir = results_dir / "exp_a_run1"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "runs.jsonl").write_text(
        _run(trace_id="trace-1").model_dump_json() + "\n", encoding="utf-8"
    )
    (experiment_dir / "execution_traces.jsonl").write_text(
        _trace().model_dump_json() + "\n", encoding="utf-8"
    )

    generate_experiment_reports(results_dir)

    report = (experiment_dir / "report.md").read_text(encoding="utf-8")
    assert "**Execution trace:**" in report
    assert "- Tools: `recall` × 2" in report


def test_generate_experiment_reports_passes_max_examples_through(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    exp_a = results_dir / "exp_a_run1"
    exp_a.mkdir(parents=True)
    runs = [_run(scenario_id=f"conv-{i}") for i in range(5)]
    (exp_a / "runs.jsonl").write_text(
        "\n".join(run.model_dump_json() for run in runs) + "\n", encoding="utf-8"
    )

    generate_experiment_reports(results_dir, max_examples=2)

    report = (exp_a / "report.md").read_text(encoding="utf-8")
    assert "Showing full detail for 2 of 5 runs" in report


def test_generate_experiment_reports_skips_experiment_with_no_results(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    empty_exp = results_dir / "exp_empty_run1"
    empty_exp.mkdir(parents=True)
    (empty_exp / "runs.jsonl").write_text("", encoding="utf-8")

    written = generate_experiment_reports(results_dir)

    assert written == []
    assert not (empty_exp / "report.md").exists()


def test_generate_experiment_reports_detailed_appends_the_csv_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`detailed=True` writes submetrics.csv via `submetric_csv` - this only
    checks the orchestration (call made, path appended); the CSV's own
    content is covered by test_submetric_csv.py."""
    results_dir = tmp_path / "results"
    exp_a = results_dir / "exp_a_run1"
    exp_a.mkdir(parents=True)
    (exp_a / "runs.jsonl").write_text(
        _run(scenario_id="conv-1").model_dump_json() + "\n", encoding="utf-8"
    )
    csv_path = results_dir / "submetrics.csv"
    calls: list[Path] = []

    def _fake_generate_submetric_csv(path: Path) -> Path:
        calls.append(path)
        return csv_path

    monkeypatch.setattr(
        "heams.reporting.submetric_csv.generate_submetric_csv",
        _fake_generate_submetric_csv,
    )

    written = generate_experiment_reports(results_dir, detailed=True)

    assert calls == [results_dir]
    assert written == [exp_a / "report.md", csv_path]


def test_generate_experiment_reports_omits_csv_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results"
    exp_a = results_dir / "exp_a_run1"
    exp_a.mkdir(parents=True)
    (exp_a / "runs.jsonl").write_text(
        _run(scenario_id="conv-1").model_dump_json() + "\n", encoding="utf-8"
    )
    calls: list[Path] = []
    monkeypatch.setattr(
        "heams.reporting.submetric_csv.generate_submetric_csv", calls.append
    )

    written = generate_experiment_reports(results_dir)

    assert calls == []
    assert written == [exp_a / "report.md"]


def test_generate_experiment_reports_detailed_warns_when_csv_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results"
    exp_a = results_dir / "exp_a_run1"
    exp_a.mkdir(parents=True)
    (exp_a / "runs.jsonl").write_text(
        _run(scenario_id="conv-1").model_dump_json() + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "heams.reporting.submetric_csv.generate_submetric_csv", lambda path: None
    )

    written = generate_experiment_reports(results_dir, detailed=True)

    assert written == [exp_a / "report.md"]


def test_generate_experiment_reports_detailed_skips_csv_when_nothing_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No experiment reports means nothing to export either - the CSV
    generator must not even be called."""
    results_dir = tmp_path / "results"
    empty_exp = results_dir / "exp_empty_run1"
    empty_exp.mkdir(parents=True)
    (empty_exp / "runs.jsonl").write_text("", encoding="utf-8")
    calls: list[Path] = []
    monkeypatch.setattr(
        "heams.reporting.submetric_csv.generate_submetric_csv", calls.append
    )

    written = generate_experiment_reports(results_dir, detailed=True)

    assert written == []
    assert calls == []
