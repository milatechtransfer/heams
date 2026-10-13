"""Read ``runs.jsonl`` records without loading huge ones into memory.

Most records are a few megabytes and are parsed with :func:`json.loads`.
Some are not: an EvoMemory run that replays its whole history through tool
outputs can write a single record of several gigabytes, which would need
several times that in memory once decoded. Records above
:data:`LARGE_RECORD_BYTES` are instead walked structurally over a memory map
of the file: the walker finds where each JSON value starts and ends without
decoding it, and only small values are materialized.
"""

from __future__ import annotations

import json
import mmap
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

#: Records up to this size are parsed whole; larger ones are walked in place.
LARGE_RECORD_BYTES = 64 * 1024 * 1024
#: Largest single value materialized while walking a large record.
_MAX_VALUE_BYTES = 8 * 1024 * 1024
_QUESTION_PREVIEW_BYTES = 4096

_STRUCTURAL = re.compile(rb'["\[\]{}]')
_NON_WHITESPACE = re.compile(rb"[^ \t\r\n]")
_SCALAR_END = re.compile(rb"[,\]} \t\r\n]")
_QUOTE, _BACKSLASH, _COLON, _COMMA = 0x22, 0x5C, 0x3A, 0x2C
_OPEN, _CLOSE = frozenset(b"{["), frozenset(b"}]")

#: Top-level ``RunResult`` arrays that can hold most of a record's size.
STREAMED_KEYS = frozenset(
    {
        "transcript",
        "memory_ledger",
        "component_metrics",
        "component_performance_metrics",
    }
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start


class Walker:
    """Locate JSON values in a byte buffer (``bytes`` or ``mmap``)."""

    def __init__(self, buffer: Any, end: int) -> None:
        self.buffer = buffer
        self.end = end

    def skip_ws(self, position: int) -> int:
        match = _NON_WHITESPACE.search(self.buffer, position, self.end)
        if match is None:
            raise ValueError("unexpected end of record")
        return match.start()

    def string_end(self, position: int) -> int:
        """Index just after the string opening at `position`."""
        cursor = position + 1
        while True:
            quote = self.buffer.find(b'"', cursor, self.end)
            if quote < 0:
                raise ValueError("unterminated string")
            backslashes = 0
            back = quote - 1
            while back > position and self.buffer[back] == _BACKSLASH:
                backslashes += 1
                back -= 1
            if backslashes % 2 == 0:
                return quote + 1
            cursor = quote + 1

    def value_end(self, position: int) -> int:
        first = self.buffer[position]
        if first == _QUOTE:
            return self.string_end(position)
        if first not in _OPEN:
            match = _SCALAR_END.search(self.buffer, position, self.end)
            return match.start() if match else self.end
        depth = 0
        cursor = position
        while True:
            match = _STRUCTURAL.search(self.buffer, cursor, self.end)
            if match is None:
                raise ValueError("unterminated container")
            index = match.start()
            char = self.buffer[index]
            if char == _QUOTE:
                cursor = self.string_end(index)
                continue
            depth += 1 if char in _OPEN else -1
            cursor = index + 1
            if depth == 0:
                return cursor

    def members(self, position: int) -> Iterator[tuple[str, Span]]:
        """``(key, value span)`` for the object opening at `position`."""
        cursor = self.skip_ws(position + 1)
        if self.buffer[cursor] in _CLOSE:
            return
        while True:
            key_end = self.string_end(cursor)
            key = json.loads(self.buffer[cursor:key_end])
            cursor = self.skip_ws(key_end)
            if self.buffer[cursor] != _COLON:
                raise ValueError("expected ':'")
            start = self.skip_ws(cursor + 1)
            end = self.value_end(start)
            yield key, Span(start, end)
            cursor = self.skip_ws(end)
            if self.buffer[cursor] != _COMMA:
                return
            cursor = self.skip_ws(cursor + 1)

    def items(self, position: int) -> Iterator[Span]:
        """Element spans of the array opening at `position`."""
        cursor = self.skip_ws(position + 1)
        if self.buffer[cursor] in _CLOSE:
            return
        while True:
            end = self.value_end(cursor)
            yield Span(cursor, end)
            cursor = self.skip_ws(end)
            if self.buffer[cursor] != _COMMA:
                return
            cursor = self.skip_ws(cursor + 1)

    def load(self, span: Span) -> Any:
        return json.loads(self.buffer[span.start : span.end])

    def text_preview(self, span: Span, limit: int) -> str:
        """Start of a string value (or raw JSON for anything else)."""
        start = span.start + 1 if self.buffer[span.start] == _QUOTE else span.start
        raw = bytes(self.buffer[start : min(span.end, start + limit)])
        return raw.decode("utf-8", errors="replace")


class RecordFile:
    """Memory-mapped ``runs.jsonl`` with record offsets."""

    def __init__(self, path: Path) -> None:
        self._file = path.open("rb")
        size = path.stat().st_size
        self.buffer: Any = (
            mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ) if size else b""
        )
        self.size = size

    def close(self) -> None:
        if isinstance(self.buffer, mmap.mmap):
            self.buffer.close()
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def spans(self) -> Iterator[Span]:
        """Non-blank lines; JSON escapes newlines, so they delimit records."""
        position = 0
        while position < self.size:
            newline = self.buffer.find(b"\n", position)
            end = self.size if newline < 0 else newline
            start = _NON_WHITESPACE.search(self.buffer, position, end)
            if start is not None:
                yield Span(position, end)
            position = end + 1

    def release(self, span: Span) -> None:
        """Drop a read record's pages from this process (they stay cached).

        Mapped pages count towards the job's memory while resident; letting
        them go after each record keeps a multi-gigabyte scan flat.
        """
        advice = getattr(mmap, "MADV_DONTNEED", None)
        if advice is None or not isinstance(self.buffer, mmap.mmap):
            return
        start = span.start - span.start % mmap.PAGESIZE
        if span.end > start:
            self.buffer.madvise(advice, start, span.end - start)

    def record_at(self, offset: int) -> Span:
        newline = self.buffer.find(b"\n", offset)
        return Span(offset, self.size if newline < 0 else newline)

    def iter_bytes(self, span: Span, chunk: int = 1 << 20) -> Iterator[bytes]:
        for start in range(span.start, span.end, chunk):
            yield bytes(self.buffer[start : min(span.end, start + chunk)])


def summary_fields(record: RecordFile, span: Span) -> dict[str, Any]:
    """The fields a run summary needs.

    Small records are parsed whole. For large ones, top-level values up to
    :data:`_MAX_VALUE_BYTES` are loaded, streamed arrays become
    ``{"count": n, "first": {"question": ...}}``, and anything else is
    skipped.
    """
    if span.size <= LARGE_RECORD_BYTES:
        return json.loads(record.buffer[span.start : span.end])
    walker = Walker(record.buffer, span.end)
    fields: dict[str, Any] = {}
    for key, value in walker.members(walker.skip_ws(span.start)):
        if key in STREAMED_KEYS:
            count, first = 0, None
            for item in walker.items(value.start):
                if count == 0 and key == "transcript":
                    first = _turn_question(walker, item)
                count += 1
            fields[key] = {"count": count, "first": first}
        elif value.size <= _MAX_VALUE_BYTES:
            fields[key] = walker.load(value)
    return fields


def _turn_question(walker: Walker, turn: Span) -> dict[str, str]:
    for key, value in walker.members(turn.start):
        if key == "question":
            if value.size <= _MAX_VALUE_BYTES:
                return {"question": walker.load(value)}
            return {"question": walker.text_preview(value, _QUESTION_PREVIEW_BYTES)}
    return {}


__all__ = [
    "LARGE_RECORD_BYTES",
    "STREAMED_KEYS",
    "RecordFile",
    "Span",
    "Walker",
    "summary_fields",
]
