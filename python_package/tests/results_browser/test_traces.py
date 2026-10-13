"""Execution trace sidecars: joining, timeline rows and event detail."""

from __future__ import annotations

from pathlib import Path

import pytest

from heams.results_browser import records as browser_records
from heams.results_browser.catalog import (
    DisplayLimits,
    ResultsCatalog,
    TraceNotFoundError,
)


def test_trace_index_joins_runs_through_trace_id(traced: Path) -> None:
    catalog = ResultsCatalog(traced)

    summaries = catalog.run_index("run/exp").summaries
    traces = catalog.trace_index("run/exp")

    assert catalog.experiment("run/exp").to_json()["has_traces"]
    assert [summary["trace_id"] for summary in summaries] == [
        "trace-a",
        "trace-missing",
        None,
    ]
    # The first durable row wins; the truncated tail is ignored.
    assert list(traces) == ["trace-a"]
    assert traces["trace-a"]["outcome"] == "completed"
    assert traces["trace-a"]["events"] == 4


@pytest.mark.parametrize("large", [False, True])
def test_trace_timeline_lists_every_event_in_order(
    traced: Path, monkeypatch: pytest.MonkeyPatch, large: bool
) -> None:
    if large:
        monkeypatch.setattr(browser_records, "LARGE_RECORD_BYTES", 0)
    catalog = ResultsCatalog(traced)

    trace = catalog.trace("run/exp", 0)

    assert trace["trace_id"] == "trace-a"
    assert trace["outcome"] == "completed"
    events = trace["events"]
    assert [(event["type"], event["name"]) for event in events] == [
        ("memory_priming", "prime_memory_entries"),
        ("user_stimulus", "user_stimulus"),
        ("llm_call", "gpt-test"),
        ("tool_call", "recall"),
    ]
    assert [event["start_offset_ms"] for event in events] == [0.0, 5.0, 6.0, 50.0]
    llm, tool = events[2], events[3]
    assert llm["input_messages"] == 2
    assert llm["usage"] == {"input_tokens": 120, "output_tokens": 8}
    assert "input_messages" not in tool
    assert tool["arguments_preview"] == "query: sunrise"
    assert tool["status"] == "error"
    assert tool["error"] == "timeout after retry"
    assert tool["stimulus_index"] == 0


def test_trace_event_returns_one_event_in_full(traced: Path) -> None:
    catalog = ResultsCatalog(traced)

    detail = catalog.trace_event("run/exp", 0, 2)
    short = catalog.trace_event(
        "run/exp", 0, 2, limits=DisplayLimits(max_chars=100, max_block_chars=200)
    )

    assert detail["event"]["input_messages"][1]["content"] == "x" * 5_000
    assert not detail["truncated"]
    assert short["truncated"]
    assert len(short["event"]["input_messages"][1]["content"]) < 200
    with pytest.raises(IndexError):
        catalog.trace_event("run/exp", 0, 9)


def test_samples_without_a_trace_row_report_why(traced: Path) -> None:
    catalog = ResultsCatalog(traced)

    with pytest.raises(TraceNotFoundError, match="missing"):
        catalog.trace("run/exp", 1)
    with pytest.raises(TraceNotFoundError, match="no execution trace"):
        catalog.trace("run/exp", 2)
