"""Criterion table: source-run selection and cell values."""

from __future__ import annotations

from pathlib import Path

import pytest

from heams.results_browser.catalog import ResultsCatalog
from heams.results_browser.table import Pick, build_table, select_sources
from tests.results_browser.fixtures import make_run, write_experiment


def _cells(table: dict, memory: str) -> dict[tuple[str, str], dict]:
    return {
        (row["dimension"], row["criterion"]): row["cells"][memory]
        for row in table["rows"]
        if memory in row["cells"]
    }


def test_table_scores_criteria_from_runs_jsonl(campaign: Path) -> None:
    table = build_table(ResultsCatalog(campaign), wait=True)

    assert table["memory_systems"] == ["in_context", "mem0"]
    assert table["pending"] == []
    cells = _cells(table, "mem0")
    assert cells[("Performance", "Overall")]["value"] == 50.0
    assert cells[("Performance", "temporal")] == {
        "value": 100.0,
        "n": 1,
        "excluded": 0,
        "experiment": "run_mem0/exp_mem0_locomo",
    }
    assert cells[("Performance", "multi-hop")]["value"] == 0.0
    assert cells[("Efficiency", "Duration")]["value"] == 1.5
    assert next(iter(cells)) == ("Performance", "Overall")


def test_unscored_samples_are_excluded_not_counted_as_zero(tmp_path: Path) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="stale",
        memory="mem0",
        runs=[
            make_run("s-1", passed=True, category="x"),
            make_run("s-2", passed=False, category="x", scored=False),
        ],
    )

    cells = _cells(build_table(ResultsCatalog(tmp_path), wait=True), "mem0")

    assert cells[("Performance", "Overall")]["value"] == 100.0
    assert cells[("Performance", "Overall")]["n"] == 1
    assert cells[("Performance", "Overall")]["excluded"] == 1


def test_a_run_without_scored_samples_shows_no_value(tmp_path: Path) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="stale",
        memory="mem0",
        runs=[make_run("s-1", passed=False, category="x", scored=False)],
    )

    cells = _cells(build_table(ResultsCatalog(tmp_path), wait=True), "mem0")

    assert cells[("Performance", "Overall")] == {
        "value": None,
        "n": 0,
        "excluded": 1,
        "experiment": "run/exp",
    }


@pytest.mark.parametrize(
    ("pick", "expected"),
    [
        (Pick.LARGEST, "subset_big/exp"),
        (Pick.LATEST, "smoke_new/exp"),
        (Pick.LATEST_FULL, "full_old/exp"),
    ],
)
def test_pick_modes_choose_one_run(
    mem0_history: Path, pick: Pick, expected: str
) -> None:
    catalog = ResultsCatalog(mem0_history)

    selected = select_sources(list(catalog.experiments.values()), pick)

    assert selected[("mem0", "locomo")].id == expected


def test_table_sources_list_selected_run_and_alternatives(mem0_history: Path) -> None:
    table = build_table(ResultsCatalog(mem0_history), Pick.LARGEST, wait=True)

    [source] = table["sources"]
    assert source["state"] == "complete"
    assert source["selected"] == {
        "experiment": "subset_big/exp",
        "run_dir": "subset_big",
        "started": "2026-01-15 00:00",
        "samples": 5,
        "limit": 5,
        "subset": True,
        "status": "complete",
    }
    assert [alt["experiment"] for alt in source["alternatives"]] == [
        "running_newest/exp",
        "smoke_new/exp",
        "full_old/exp",
    ]


def test_largest_counts_unfinished_runs_and_flags_them(mem0_history: Path) -> None:
    write_experiment(
        mem0_history / "running_biggest",
        "exp",
        benchmark="locomo",
        memory="mem0",
        runs=[make_run("conv-1", passed=True, category="temporal")] * 9,
        run_id="20260105-000000-eeeeeeee",
        complete=False,
    )
    catalog = ResultsCatalog(mem0_history)

    table = build_table(catalog, Pick.LARGEST, wait=True)
    latest = select_sources(list(catalog.experiments.values()), Pick.LATEST)

    [source] = table["sources"]
    assert source["selected"]["experiment"] == "running_biggest/exp"
    assert source["selected"]["samples"] == 9
    assert source["state"] == "incomplete"
    assert table["rows"]
    assert latest[("mem0", "locomo")].id == "smoke_new/exp"


def test_latest_without_finished_run_reports_its_status(tmp_path: Path) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="stale",
        memory="mem0",
        runs=[make_run("s-1", passed=True, category="x")],
        complete=False,
    )

    table = build_table(ResultsCatalog(tmp_path), Pick.LATEST, wait=True)

    assert table["rows"] == []
    assert table["sources"][0]["selected"] is None
    assert table["sources"][0]["state"] == "incomplete"


def test_retention_policy_uses_runs_over_every_policy(tmp_path: Path) -> None:
    runs = [
        make_run(f"RP-{i:02d}", passed=True, category="health") for i in range(1, 4)
    ]
    write_experiment(
        tmp_path / "all_policies",
        "exp",
        benchmark="retention_policy",
        memory="milamem",
        runs=runs,
        run_id="20260101-000000-aaaaaaaa",
    )
    write_experiment(
        tmp_path / "one_policy",
        "exp",
        benchmark="retention_policy",
        memory="milamem",
        runs=runs[:1],
        run_id="20260201-000000-bbbbbbbb",
        policy_ids=["RP-01"],
    )
    write_experiment(
        tmp_path / "legacy_per_policy",
        "exp",
        benchmark="retention_policy",
        memory="milamem_rp01_variant",
        runs=runs[:1],
        policy_ids=["RP-01"],
    )
    catalog = ResultsCatalog(tmp_path)

    table = build_table(catalog, Pick.LATEST, wait=True)

    # The newer single-policy run does not stand for the benchmark, and a
    # memory system seen only in single-policy runs gets no column.
    assert table["memory_systems"] == ["milamem"]
    [source] = table["sources"]
    assert source["selected"]["experiment"] == "all_policies/exp"
    assert source["alternatives"] == []
    assert _cells(table, "milamem")[("Privacy", "Overall")]["n"] == 3


def test_table_indexes_in_background_until_ready(campaign: Path) -> None:
    catalog = ResultsCatalog(campaign)

    first = build_table(catalog)
    assert catalog._worker is not None
    catalog._worker.join(timeout=10)
    second = build_table(catalog)

    assert first["rows"] == [] or not first["pending"]
    assert second["pending"] == []
    assert second["rows"]


def test_rows_follow_declared_sections_sizes_and_labels(tmp_path: Path) -> None:
    for name, benchmark, fields in (
        ("stale", "stale", {}),
        ("beam_1m", "beam", {"split": "1M"}),
        ("ci", "ci", {}),
        ("long_mem_eval", "long_mem_eval", {}),
        ("beam_100k", "beam", {"split": "100K"}),
        ("locomo", "locomo", {}),
        ("new", "someone_elses_benchmark", {}),
    ):
        write_experiment(
            tmp_path / name,
            "exp",
            benchmark=benchmark,
            # No plugin declares this backend, so its label is the id in any
            # environment (installed plugins' names are covered elsewhere).
            memory="someone_elses_memory",
            memory_backend="someone_elses_memory",
            runs=[make_run("s-1", passed=True, category="x")],
            benchmark_fields=fields,
        )

    table = build_table(ResultsCatalog(tmp_path), wait=True)

    overall = [row for row in table["rows"] if row["criterion"] == "Overall"]
    # Sections in the framework's order; within one, smaller declared
    # histories first, undeclared sizes last; undeclared benchmarks at the end.
    assert [(row["dimension"], row["benchmark_label"]) for row in overall] == [
        ("Performance", "LoCoMo"),
        ("Performance", "BEAM 100K"),
        ("Performance", "LongMemEval"),
        ("Performance", "BEAM 1M"),
        ("Performance", "STALE"),
        ("Privacy", "Contextual Integrity"),
        ("Other", "someone_elses_benchmark"),
    ]
    assert table["dimensions"][:3] == ["Performance", "Privacy", "Efficiency"]
    assert table["benchmark_labels"]["beam:1M"] == "BEAM 1M"
    assert table["memory_labels"] == {"someone_elses_memory": "someone_elses_memory"}
