"""Shared fixtures for results-browser tests."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from heams.results_browser import server as browser_server
from heams.results_browser.catalog import ResultsCatalog
from tests.results_browser.fixtures import (
    make_run,
    trace_row,
    write_experiment,
    write_submetrics,
)


@pytest.fixture
def campaign(tmp_path: Path) -> Path:
    """Two memory systems on LoCoMo, one with a `submetrics.csv`, plus a
    nested agent workspace that must not count as an experiment."""
    root = tmp_path / "campaign"
    for memory in ("in_context", "mem0"):
        run_dir = root / f"run_{memory}"
        name = f"exp_{memory}_locomo"
        write_experiment(
            run_dir,
            name,
            benchmark="locomo",
            memory=memory,
            runs=[
                make_run("conv-1", passed=True, category="temporal"),
                make_run("conv-2", passed=False, category="multi-hop"),
            ],
        )
        write_submetrics(
            run_dir,
            [
                {
                    "campaign": "c",
                    "experiment": name,
                    "memory_system": memory,
                    "benchmark": "locomo",
                    "scope": "headline",
                    "dimension": "Performance",
                    "criterion": "Overall",
                    "metric_name": "Task accuracy",
                    "aggregation": "mean",
                    "metric_value": "50",
                    "unit": "%",
                    "sample_count": "2",
                    "excluded_count": "0",
                },
            ],
        )
    write_experiment(
        root / "run_mem0" / "exp_mem0_locomo" / "workspace",
        "nested",
        benchmark="locomo",
        memory="mem0",
        runs=[],
    )
    return root


@pytest.fixture
def mem0_history(tmp_path: Path) -> Path:
    """Four Mem0 LoCoMo runs: full but old, a big subset, a new smoke run,
    and the newest one still running."""
    runs = [make_run("conv-1", passed=True, category="temporal")]
    for name, run_id, limit, size, complete in (
        ("full_old", "20260101-000000-aaaaaaaa", None, 2, True),
        ("subset_big", "20260115-000000-bbbbbbbb", 5, 5, True),
        ("smoke_new", "20260201-000000-cccccccc", 1, 1, True),
        ("running_newest", "20260301-000000-dddddddd", None, 1, False),
    ):
        write_experiment(
            tmp_path / name,
            "exp",
            benchmark="locomo",
            memory="mem0",
            runs=runs * size,
            run_id=run_id,
            limit=limit,
            complete=complete,
        )
    return tmp_path


@pytest.fixture
def traced(tmp_path: Path) -> Path:
    """Samples joined to a trace sidecar through `trace_id`: one with a trace,
    one whose trace row is missing, one without a trace."""
    experiment_dir = write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="locomo",
        memory="mem0",
        runs=[
            make_run("conv-1", passed=True, category="temporal"),
            make_run("conv-2", passed=False, category="temporal"),
            make_run("conv-3", passed=True, category="temporal"),
        ],
    )
    runs_path = experiment_dir / "runs.jsonl"
    runs = [json.loads(line) for line in runs_path.read_text().splitlines()]
    runs[0]["trace_id"] = "trace-a"
    runs[1]["trace_id"] = "trace-missing"
    runs_path.write_text("".join(f"{json.dumps(run)}\n" for run in runs))
    rows = [
        trace_row("trace-a", "conv-1"),
        {**trace_row("trace-a", "duplicate"), "outcome": "failed"},
    ]
    (experiment_dir / "execution_traces.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows)
        + '{"schema_version": 1, "trace_id": "trace-b", "events": ['
    )
    return tmp_path


@pytest.fixture
def serve() -> Iterator[Callable[[Path], str]]:
    """Start `heams browse` servers on free ports; returns their base URLs."""
    servers = []

    def start(results_dir: Path) -> str:
        server = browser_server.make_server(ResultsCatalog(results_dir), port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()
