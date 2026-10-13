"""Read ``execution_traces.jsonl`` sidecars for the sample timeline.

Each line is one sample's execution trace (``ExecutionTraceRecord``, schema
version 1): ordered framework and agent events — priming, manipulations,
user stimuli, memory operations, LLM calls and tool calls — joined to
``runs.jsonl`` through ``RunResult.trace_id``. LLM-call events carry their
full input messages, so a trace can be far larger than its run record; lines
are walked in place (see :mod:`heams.results_browser.records`) and events are
summarized without loading their large fields.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from heams.execution_traces import TraceEventKind, trace_event_kind
from heams.results_browser.records import RecordFile, Span, Walker

TRACES_JSONL_FILENAME = "execution_traces.jsonl"

#: Header and event fields up to this size are decoded; larger ones are
#: previewed from their raw JSON.
_LOAD_BYTES = 64 * 1024
_HEADER_LOAD_BYTES = 1 << 20
_PREVIEW_CHARS = 300
_EVENT_FIELDS = frozenset(
    {
        "sequence_index",
        "phase",
        "type",
        "name",
        "start_offset_ms",
        "duration_ms",
        "status",
        "error",
        "source",
        "step_index",
        "stimulus_index",
        "usage",
    }
)
_PREVIEWED_FIELDS = ("content", "arguments", "result")


def _header(walker: Walker, span: Span) -> tuple[dict[str, Any], Span | None]:
    """Top-level fields other than ``events``, plus the events array span."""
    header: dict[str, Any] = {}
    events: Span | None = None
    for key, value in walker.members(walker.skip_ws(span.start)):
        if key == "events":
            events = value
        elif value.size <= _HEADER_LOAD_BYTES:
            header[key] = walker.load(value)
    return header, events


def index_traces(path: Path) -> dict[str, dict[str, Any]]:
    """``{trace_id: {"offset", "outcome", "events"}}`` for one sidecar.

    A truncated tail (a trace still being written) is skipped; for a
    duplicated ID the first durable row wins, as in the trace loader.
    """
    traces: dict[str, dict[str, Any]] = {}
    with RecordFile(path) as records:
        for span in records.spans():
            walker = Walker(records.buffer, span.end)
            try:
                header, events = _header(walker, span)
                count = (
                    sum(1 for _ in walker.items(events.start))
                    if events is not None and records.buffer[events.start] == ord("[")
                    else 0
                )
            except ValueError:
                continue
            finally:
                records.release(span)
            trace_id = header.get("trace_id")
            if isinstance(trace_id, str) and trace_id not in traces:
                traces[trace_id] = {
                    "offset": span.start,
                    "outcome": header.get("outcome"),
                    "events": count,
                }
    return traces


def _leaves(value: Any, key: str | None = None) -> Iterator[str]:
    """Scalar values of a structure as ``key: value``, depth first.

    A preview of ``{"memory_calls": [{"arguments": {"content": "..."}}]}``
    then reads ``content: ...`` instead of the JSON around it.
    """
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _leaves(child, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _leaves(child, key)
    elif value is not None and value != "":
        yield str(value) if key is None else f"{key}: {value}"


def _preview(walker: Walker, value: Span) -> str:
    if value.size <= _LOAD_BYTES:
        loaded = walker.load(value)
        if loaded is None or loaded == {} or loaded == []:
            return ""
        text = " · ".join(_leaves(loaded))
    else:
        text = walker.text_preview(value, 2 * _PREVIEW_CHARS)
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else f"{text[: _PREVIEW_CHARS - 1]}…"


def event_overview(walker: Walker, span: Span, position: int) -> dict[str, Any]:
    """Timeline row for one event: small fields plus previews and sizes."""
    row: dict[str, Any] = {"i": position}
    for key, value in walker.members(span.start):
        if key in _EVENT_FIELDS:
            row[key] = (
                walker.load(value)
                if value.size <= _LOAD_BYTES
                else walker.text_preview(value, _PREVIEW_CHARS)
            )
        elif key in _PREVIEWED_FIELDS:
            row[f"{key}_preview"] = _preview(walker, value)
            row[f"{key}_bytes"] = value.size
        elif key == "input_messages":
            row["input_messages"] = (
                sum(1 for _ in walker.items(value.start))
                if walker.buffer[value.start] == ord("[")
                else 0
            )
            row["input_messages_bytes"] = value.size
    row["kind"] = trace_event_kind(str(row.get("type", ""))).value
    return row


def trace_overview(path: Path, offset: int) -> dict[str, Any]:
    """Trace header and one :func:`event_overview` row per event."""
    with RecordFile(path) as records:
        span = records.record_at(offset)
        walker = Walker(records.buffer, span.end)
        header, events = _header(walker, span)
        rows: list[dict[str, Any]] = []
        if events is not None and records.buffer[events.start] == ord("["):
            rows = [
                event_overview(walker, item, position)
                for position, item in enumerate(walker.items(events.start))
            ]
        records.release(span)
    header["events"] = rows
    header["bytes"] = span.size
    header["kinds"] = {kind.value: kind.label for kind in TraceEventKind}
    return header


def event_span(records: RecordFile, offset: int, position: int) -> tuple[Walker, Span]:
    """Locate the `position`-th event of the trace starting at `offset`."""
    span = records.record_at(offset)
    walker = Walker(records.buffer, span.end)
    _, events = _header(walker, span)
    if events is None or records.buffer[events.start] != ord("["):
        raise IndexError(position)
    for index, item in enumerate(walker.items(events.start)):
        if index == position:
            return walker, item
    raise IndexError(position)


__all__ = [
    "TRACES_JSONL_FILENAME",
    "event_overview",
    "event_span",
    "index_traces",
    "trace_overview",
]
