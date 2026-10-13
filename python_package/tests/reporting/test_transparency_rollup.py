"""Artifact-only Transparency rollup tests."""

from __future__ import annotations

from pathlib import Path

from heams.reporting.benchmark_rollup import write_benchmark_rollup_reports


def test_write_benchmark_rollup_returns_empty_without_runs(tmp_path: Path) -> None:
    results_dir = tmp_path / "empty"
    results_dir.mkdir()
    assert write_benchmark_rollup_reports(results_dir) == {}
