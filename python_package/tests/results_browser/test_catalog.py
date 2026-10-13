"""Experiment discovery, run indexes and bounded run records."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from heams.results_browser import catalog as browser_catalog
from heams.results_browser import records as browser_records
from heams.results_browser.catalog import (
    CACHE_DIRNAME,
    DisplayLimits,
    RecordTooLargeError,
    ResultsCatalog,
    distinct_labels,
)
from tests.results_browser.fixtures import make_run, write_experiment


def test_catalog_discovers_experiments_and_skips_workspaces(campaign: Path) -> None:
    catalog = ResultsCatalog(campaign)

    assert sorted(catalog.experiments) == [
        "run_in_context/exp_in_context_locomo",
        "run_mem0/exp_mem0_locomo",
    ]
    experiment = catalog.experiment("run_mem0/exp_mem0_locomo")
    assert experiment.benchmark == "locomo"
    assert experiment.memory_system == "mem0"
    assert experiment.agent_model == "openai:gpt-test"
    assert experiment.status == "complete"
    assert experiment.has_submetrics


def test_experiment_metadata_records_run_date_size_and_scope(campaign: Path) -> None:
    experiment = ResultsCatalog(campaign).experiment("run_mem0/exp_mem0_locomo")

    assert experiment.run_dir == "run_mem0"
    assert experiment.run_id == "20260101-000000-abcd1234"
    assert experiment.started == "2026-01-01 00:00"
    assert experiment.planned_samples == 2
    assert experiment.recorded_samples == 2
    assert experiment.limit is None
    assert not experiment.subset


@pytest.mark.parametrize(
    ("benchmark", "limit", "policy_ids", "subset"),
    [
        ("locomo", None, None, False),
        ("locomo", 25, None, True),
        ("retention_policy", None, None, False),
        ("retention_policy", None, ["RP-01"], True),
    ],
)
def test_subset_covers_limits_and_policy_selections(
    tmp_path: Path,
    benchmark: str,
    limit: int | None,
    policy_ids: list[str] | None,
    subset: bool,
) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark=benchmark,
        memory="mem0",
        runs=[make_run("s-1", passed=True, category="x")],
        limit=limit,
        policy_ids=policy_ids,
    )

    experiment = ResultsCatalog(tmp_path).experiment("run/exp")

    assert experiment.subset is subset
    assert experiment.to_json()["subset"] is subset


def test_benchmark_variants_come_from_the_benchmark_declaration(tmp_path: Path) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="beam",
        memory="in_context",
        runs=[make_run("BEAM-1", passed=True, category="abstention")],
        benchmark_fields={"split": "1M"},
    )

    experiment = ResultsCatalog(tmp_path).experiment("run/exp")

    assert (experiment.benchmark, experiment.variant) == ("beam", "1M")
    assert experiment.benchmark_key == "beam:1M"
    assert experiment.benchmark_label == "BEAM 1M"


def test_undeclared_benchmarks_are_shown_under_their_type(tmp_path: Path) -> None:
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="someone_elses_benchmark",
        memory="custom_memory",
        memory_backend="custom_memory",
        runs=[make_run("s-1", passed=True, category="x")],
    )

    experiment = ResultsCatalog(tmp_path).experiment("run/exp")

    assert experiment.benchmark_label == "someone_elses_benchmark"
    assert experiment.memory_label == "custom_memory"
    assert not experiment.subset


def test_memory_labels_come_from_plugin_display_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        browser_catalog, "memory_display_names", lambda: {"letta": "Letta"}
    )
    for memory in ("letta_openai", "letta_local"):
        write_experiment(
            tmp_path / memory,
            "exp",
            benchmark="locomo",
            memory=memory,
            memory_backend="letta",
            runs=[make_run("conv-1", passed=True, category="temporal")],
        )
    write_experiment(
        tmp_path / "solo",
        "exp",
        benchmark="locomo",
        memory="hosted_letta",
        memory_backend="unknown_backend",
        runs=[make_run("conv-1", passed=True, category="temporal")],
    )

    catalog = ResultsCatalog(tmp_path)

    labels = {e.memory_system: e.memory_label for e in catalog.experiments.values()}
    assert labels == {
        "letta_local": "Letta",
        "letta_openai": "Letta",
        "hosted_letta": "hosted_letta",
    }
    # Side by side, two configurations of one plugin keep their ids apart.
    assert distinct_labels(labels) == {
        "letta_local": "Letta (letta_local)",
        "letta_openai": "Letta (letta_openai)",
        "hosted_letta": "hosted_letta",
    }


def test_run_index_summarizes_and_caches_runs(campaign: Path) -> None:
    catalog = ResultsCatalog(campaign)
    experiment_id = "run_mem0/exp_mem0_locomo"

    index = catalog.run_index(experiment_id)

    assert [summary["scenario_id"] for summary in index.summaries] == [
        "conv-1",
        "conv-2",
    ]
    assert index.summaries[1]["tags"] == {"category=multi-hop": [0, 1]}
    assert index.summaries[1]["scores"] == {"multi-hop": 0.0}
    assert index.summaries[0]["duration_s"] == 1.5
    assert list((campaign / CACHE_DIRNAME).glob("*.json"))
    # A fresh catalog reuses the on-disk cache instead of rescanning.
    reloaded = ResultsCatalog(campaign).run_index(experiment_id)
    assert reloaded.offsets == index.offsets


def test_cache_dir_keeps_indexes_out_of_results_dir(
    campaign: Path, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    catalog = ResultsCatalog(campaign, cache_dir=cache_dir)

    catalog.run_index("run_mem0/exp_mem0_locomo")

    assert list(cache_dir.rglob("*.json"))
    assert not (campaign / CACHE_DIRNAME).exists()


def test_shared_cache_dir_keeps_results_roots_apart(tmp_path: Path) -> None:
    # Two snapshots with the same experiment id and identical file stats.
    for root, scenario in (("a", "conv-1"), ("b", "conv-2")):
        write_experiment(
            tmp_path / root / "run",
            "exp",
            benchmark="locomo",
            memory="mem0",
            runs=[make_run(scenario, passed=True, category="temporal")],
        )
        os.utime(tmp_path / root / "run" / "exp" / "runs.jsonl", ns=(1, 1))
    cache_dir = tmp_path / "cache"

    first = ResultsCatalog(tmp_path / "a", cache_dir=cache_dir).run_index("run/exp")
    second = ResultsCatalog(tmp_path / "b", cache_dir=cache_dir).run_index("run/exp")

    assert first.summaries[0]["scenario_id"] == "conv-1"
    assert second.summaries[0]["scenario_id"] == "conv-2"


def test_rescan_recounts_and_reindexes_a_growing_run(tmp_path: Path) -> None:
    experiment_dir = write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="locomo",
        memory="mem0",
        runs=[make_run("conv-1", passed=True, category="temporal")],
        complete=False,
    )
    catalog = ResultsCatalog(tmp_path)
    assert len(catalog.run_index("run/exp").offsets) == 1

    runs = experiment_dir / "runs.jsonl"
    extra = make_run("conv-2", passed=False, category="temporal").model_dump_json()
    runs.write_text(runs.read_text() + extra + "\n")
    catalog.refresh()

    assert catalog.experiment("run/exp").recorded_samples == 2
    assert catalog.indexed("run/exp") is None
    assert len(catalog.run_index("run/exp").offsets) == 2


def test_index_of_a_run_that_grew_since_is_not_served(tmp_path: Path) -> None:
    # The index can be published after a rescan that saw no index to drop.
    experiment_dir = write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="locomo",
        memory="mem0",
        runs=[make_run("conv-1", passed=True, category="temporal")],
        complete=False,
    )
    catalog = ResultsCatalog(tmp_path)
    catalog.run_index("run/exp")
    assert catalog.indexed("run/exp") is not None

    runs = experiment_dir / "runs.jsonl"
    extra = make_run("conv-2", passed=False, category="temporal").model_dump_json()
    runs.write_text(runs.read_text() + extra + "\n")

    assert catalog.indexed("run/exp") is None


def test_truncated_tail_of_running_job_is_skipped(campaign: Path) -> None:
    runs = campaign / "run_mem0" / "exp_mem0_locomo" / "runs.jsonl"
    runs.write_bytes(runs.read_bytes() + b'{"scenario_id": "conv-3", "audit": {')

    index = ResultsCatalog(campaign).run_index("run_mem0/exp_mem0_locomo")

    assert len(index.offsets) == 2


def test_run_reads_record_by_offset_and_bounds_large_blocks(campaign: Path) -> None:
    catalog = ResultsCatalog(campaign)
    experiment_id = "run_in_context/exp_in_context_locomo"

    full = catalog.run(experiment_id, 1, limits=None)
    bounded = catalog.run(
        experiment_id,
        1,
        limits=DisplayLimits(max_block_chars=1_000, block_preview_chars=100),
    )

    assert full["run"]["scenario_id"] == "conv-2"
    assert not full["truncated"]
    assert bounded["truncated"]
    output = bounded["run"]["transcript"][0]["tool_results"][0]["output"]
    assert isinstance(output, str)
    assert "characters of JSON truncated" in output
    with pytest.raises(IndexError):
        catalog.raw_run(experiment_id, 5)


@pytest.fixture
def every_record_is_large(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route every record through the in-place walker used for huge ones."""
    monkeypatch.setattr(browser_records, "LARGE_RECORD_BYTES", 0)
    monkeypatch.setattr(browser_catalog, "LARGE_RECORD_BYTES", 0)


def test_large_records_summarize_like_small_ones(
    campaign: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_id = "run_in_context/exp_in_context_locomo"
    small = ResultsCatalog(campaign, cache_dir=tmp_path / "a").run_index(experiment_id)
    monkeypatch.setattr(browser_records, "LARGE_RECORD_BYTES", 0)

    large = ResultsCatalog(campaign, cache_dir=tmp_path / "b").run_index(experiment_id)

    assert large.offsets == small.offsets
    assert large.summaries == small.summaries


@pytest.mark.usefixtures("every_record_is_large")
def test_large_record_is_shown_bounded_and_downloaded_raw(campaign: Path) -> None:
    catalog = ResultsCatalog(campaign)
    experiment_id = "run_in_context/exp_in_context_locomo"

    shown = catalog.run(
        experiment_id,
        1,
        limits=DisplayLimits(max_block_chars=1_000, block_preview_chars=100),
    )
    size, chunks = catalog.raw_run(experiment_id, 1)
    raw = b"".join(chunks)

    assert shown["truncated"]
    assert shown["run"]["scenario_id"] == "conv-2"
    assert shown["run"]["audit"]["failed_keys"] == ["q0"]
    output = shown["run"]["transcript"][0]["tool_results"][0]["output"]
    assert "bytes of JSON truncated" in output
    assert size == len(raw)
    assert json.loads(raw)["scenario_id"] == "conv-2"
    with pytest.raises(RecordTooLargeError):
        catalog.run(experiment_id, 1, limits=None)
