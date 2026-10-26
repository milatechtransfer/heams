"""Durable, normalized execution traces for benchmark samples."""

from __future__ import annotations

import ast
import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from heams.memory_systems.schemas import MemoryOperationEvent

TracePhase = Literal["priming", "execution", "finalization"]
TraceStatus = Literal["unset", "ok", "error", "skipped"]


class TraceEventKind(StrEnum):
    """Family of an event ``type``, for summaries and timelines.

    Event types are open-ended strings; every type the framework emits is
    mapped in :data:`TRACE_EVENT_KINDS`, and any other type is ``OTHER``.
    """

    SETUP = "setup"
    MANIPULATION = "manipulation"
    INPUT = "input"
    LLM = "llm"
    TOOL = "tool"
    MEMORY = "memory"
    OTHER = "other"

    @property
    def label(self) -> str:
        return _TRACE_EVENT_KIND_LABELS[self]


_TRACE_EVENT_KIND_LABELS = {
    TraceEventKind.SETUP: "Setup",
    TraceEventKind.MANIPULATION: "Manipulations",
    TraceEventKind.INPUT: "User inputs",
    TraceEventKind.LLM: "LLM calls",
    TraceEventKind.TOOL: "Tool calls",
    TraceEventKind.MEMORY: "Memory operations",
    TraceEventKind.OTHER: "Other events",
}

#: Event types emitted by the framework, by kind.
TRACE_EVENT_KINDS: dict[str, TraceEventKind] = {
    "environment_priming": TraceEventKind.SETUP,
    "memory_priming": TraceEventKind.SETUP,
    "agent_manipulation": TraceEventKind.MANIPULATION,
    "environment_manipulation": TraceEventKind.MANIPULATION,
    "memory_manipulation": TraceEventKind.MANIPULATION,
    "user_stimulus": TraceEventKind.INPUT,
    "llm_call": TraceEventKind.LLM,
    "tool_call": TraceEventKind.TOOL,
    "memory_operation": TraceEventKind.MEMORY,
}


def trace_event_kind(event_type: str) -> TraceEventKind:
    return TRACE_EVENT_KINDS.get(event_type, TraceEventKind.OTHER)


class ExecutionTraceEvent(BaseModel):
    """One ordered framework or agent action."""

    sequence_index: int = Field(ge=0)
    phase: TracePhase
    type: str
    name: str
    content: str | None = Field(default=None, exclude_if=lambda value: value is None)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = Field(default=None, exclude_if=lambda value: value is None)
    input_messages: list[dict[str, Any]] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    usage: dict[str, float | int] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    start_offset_ms: float = Field(ge=0.0)
    duration_ms: float | None = Field(
        default=None, ge=0.0, exclude_if=lambda value: value is None
    )
    status: TraceStatus = "unset"
    error: str | None = Field(default=None, exclude_if=lambda value: value is None)
    source: str | None = Field(default=None, exclude_if=lambda value: value is None)
    step_index: int | None = Field(default=None, exclude_if=lambda value: value is None)
    stimulus_index: int | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ExecutionTraceRecord(BaseModel):
    """One complete benchmark-sample execution trace."""

    schema_version: Literal[1] = 1
    trace_id: str
    run_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    scenario_id: str
    agent_index: int
    memory_index: int
    evaluator_index: int
    sample_index: int
    outcome: Literal["completed", "failed"]
    error: str | None = Field(default=None, exclude_if=lambda value: value is None)
    events: list[ExecutionTraceEvent] = Field(default_factory=list)


class ExecutionTraceArtifactError(ValueError):
    """Raised when an execution trace artifact cannot be trusted."""


@dataclass
class _CapturedEvent:
    order: int
    start_time_ns: int
    end_time_ns: int | None
    phase: TracePhase
    type: str
    name: str
    content: str | None
    arguments: dict[str, Any]
    result: Any
    input_messages: list[dict[str, Any]] | None
    usage: dict[str, float | int] | None
    status: TraceStatus
    error: str | None
    source: str | None
    step_index: int | None
    stimulus_index: int | None


class ExecutionTraceCollector:
    """Collect framework, memory, and TinyAgent events for one sample."""

    def __init__(
        self,
        *,
        run_id: str | None,
        scenario_id: str,
        agent_index: int,
        memory_index: int,
        evaluator_index: int,
        sample_index: int,
    ) -> None:
        self.trace_id = uuid4().hex
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.agent_index = agent_index
        self.memory_index = memory_index
        self.evaluator_index = evaluator_index
        self.sample_index = sample_index
        self._created_ns = time.time_ns()
        self._events: list[_CapturedEvent] = []
        self._context: ContextVar[dict[str, Any]] = ContextVar(
            f"execution_trace_context_{self.trace_id}", default={}
        )

    @contextmanager
    def bind_context(
        self,
        *,
        phase: TracePhase,
        step_index: int | None = None,
        stimulus_index: int | None = None,
    ) -> Iterator[None]:
        token: Token[dict[str, Any]] = self._context.set(
            {
                "phase": phase,
                "step_index": step_index,
                "stimulus_index": stimulus_index,
            }
        )
        try:
            yield
        finally:
            self._context.reset(token)

    def add_event(
        self,
        *,
        phase: TracePhase,
        type: str,
        name: str,
        start_time_ns: int,
        end_time_ns: int | None = None,
        content: str | None = None,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        input_messages: list[dict[str, Any]] | None = None,
        usage: dict[str, float | int] | None = None,
        status: TraceStatus = "unset",
        error: str | None = None,
        source: str | None = None,
        step_index: int | None = None,
        stimulus_index: int | None = None,
    ) -> None:
        self._events.append(
            _CapturedEvent(
                order=len(self._events),
                start_time_ns=start_time_ns,
                end_time_ns=end_time_ns,
                phase=phase,
                type=type,
                name=name,
                content=content,
                arguments=_json_safe(arguments or {}),
                result=_json_safe(result),
                input_messages=_json_safe(input_messages),
                usage=usage,
                status=status,
                error=error,
                source=source,
                step_index=step_index,
                stimulus_index=stimulus_index,
            )
        )

    def add_memory_event(self, event: MemoryOperationEvent) -> None:
        context = self._context.get()
        run_context = event.payload.get("run_context", {})
        if not isinstance(run_context, dict):
            run_context = {}
        end_time_ns = time.time_ns()
        duration_ms = event.duration_ms or 0.0
        start_time_ns = end_time_ns - int(duration_ms * 1_000_000)
        result = event.result
        content = _memory_event_content(result)
        self.add_event(
            phase=context.get(
                "phase",
                "execution"
                if run_context.get("stimulus_index") is not None
                else "finalization",
            ),
            type="memory_operation",
            name=event.operation_type.value,
            content=content,
            arguments={
                key: value
                for key, value in event.payload.items()
                if key != "run_context"
            },
            result=result,
            start_time_ns=start_time_ns,
            end_time_ns=end_time_ns,
            status=_memory_status(event),
            error=event.error,
            source=event.source.value,
            step_index=_optional_int(
                context.get("step_index", run_context.get("step_index"))
            ),
            stimulus_index=_optional_int(
                context.get("stimulus_index", run_context.get("stimulus_index"))
            ),
        )

    def add_agent_trace(
        self,
        raw_trace: Any,
        *,
        step_index: int,
        stimulus_index: int,
    ) -> None:
        for action in _agent_actions(raw_trace):
            self.add_event(
                phase="execution",
                step_index=step_index,
                stimulus_index=stimulus_index,
                **action,
            )

    def finalize(
        self,
        *,
        outcome: Literal["completed", "failed"],
        error: str | None = None,
    ) -> ExecutionTraceRecord:
        ordered = sorted(
            self._events,
            key=lambda event: (event.start_time_ns, event.order),
        )
        origin_ns = min(
            (event.start_time_ns for event in ordered), default=self._created_ns
        )
        events = [
            ExecutionTraceEvent(
                sequence_index=index,
                phase=event.phase,
                type=event.type,
                name=event.name,
                content=event.content,
                arguments=event.arguments,
                result=event.result,
                input_messages=event.input_messages,
                usage=event.usage,
                start_offset_ms=(event.start_time_ns - origin_ns) / 1_000_000,
                duration_ms=(
                    max(event.end_time_ns - event.start_time_ns, 0) / 1_000_000
                    if event.end_time_ns is not None
                    else None
                ),
                status=event.status,
                error=event.error,
                source=event.source,
                step_index=event.step_index,
                stimulus_index=event.stimulus_index,
            )
            for index, event in enumerate(ordered)
        ]
        return ExecutionTraceRecord(
            trace_id=self.trace_id,
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            agent_index=self.agent_index,
            memory_index=self.memory_index,
            evaluator_index=self.evaluator_index,
            sample_index=self.sample_index,
            outcome=outcome,
            error=error,
            events=events,
        )


class ExecutionTraceLedger:
    """Append-only JSONL storage for completed sample traces."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._trace_ids: set[str] = set()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._load_existing()

    def append(self, trace: ExecutionTraceRecord) -> None:
        if trace.trace_id in self._trace_ids:
            raise ExecutionTraceArtifactError(
                f"Duplicate execution trace ID {trace.trace_id!r} in {self.path}"
            )
        if self._closed:
            raise ValueError(f"Execution trace ledger {self.path} is closed")
        try:
            serialized = trace.model_dump_json()
            with self.path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(serialized + "\n")
                trace_file.flush()
                os.fsync(trace_file.fileno())
        except (TypeError, ValueError, OSError) as exc:
            raise ExecutionTraceArtifactError(
                f"Execution trace could not be written to {self.path}."
            ) from exc
        self._trace_ids.add(trace.trace_id)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> ExecutionTraceLedger:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def _load_existing(self) -> None:
        data = self.path.read_bytes()
        if data and not data.endswith(b"\n"):
            durable_end = data.rfind(b"\n") + 1
            data = data[:durable_end]
            with self.path.open("r+b") as trace_file:
                trace_file.truncate(durable_end)
        for line_number, raw_line in enumerate(data.splitlines(), start=1):
            if not raw_line.strip():
                raise ExecutionTraceArtifactError(
                    f"Malformed empty execution trace at {self.path}:{line_number}"
                )
            try:
                trace = ExecutionTraceRecord.model_validate_json(raw_line)
            except (TypeError, ValueError) as exc:
                raise ExecutionTraceArtifactError(
                    f"Malformed execution trace at {self.path}:{line_number}: {exc}"
                ) from exc
            if trace.trace_id in self._trace_ids:
                raise ExecutionTraceArtifactError(
                    f"Duplicate execution trace ID {trace.trace_id!r} "
                    f"at {self.path}:{line_number}"
                )
            self._trace_ids.add(trace.trace_id)


def load_execution_traces(
    path: Path,
) -> tuple[dict[str, ExecutionTraceRecord], list[str]]:
    """Load complete trace rows for offline reporting."""
    if not path.is_file():
        return {}, []

    data = path.read_bytes()
    warnings: list[str] = []
    if data and not data.endswith(b"\n"):
        durable_end = data.rfind(b"\n") + 1
        warnings.append(f"Ignoring truncated execution trace tail at {path}.")
        data = data[:durable_end]

    traces: dict[str, ExecutionTraceRecord] = {}
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            warnings.append(f"Skipping empty execution trace at {path}:{line_number}.")
            continue
        try:
            trace = ExecutionTraceRecord.model_validate_json(raw_line)
        except (TypeError, ValueError) as exc:
            warnings.append(
                f"Skipping malformed execution trace at {path}:{line_number}: {exc}"
            )
            continue
        if trace.trace_id in traces:
            warnings.append(
                f"Skipping duplicate execution trace ID {trace.trace_id!r} "
                f"at {path}:{line_number}."
            )
            continue
        traces[trace.trace_id] = trace
    return traces, warnings


def normalized_agent_events(raw_trace: Any) -> list[ExecutionTraceEvent]:
    """Return only TinyAgent LLM/tool events, ordered by span start."""
    actions = sorted(
        _agent_actions(raw_trace),
        key=lambda action: action["start_time_ns"],
    )
    origin_ns = actions[0]["start_time_ns"] if actions else time.time_ns()
    return [
        ExecutionTraceEvent(
            sequence_index=index,
            phase="execution",
            type=action["type"],
            name=action["name"],
            content=action["content"],
            arguments=action["arguments"],
            result=action["result"],
            input_messages=action["input_messages"],
            usage=action["usage"],
            start_offset_ms=(action["start_time_ns"] - origin_ns) / 1_000_000,
            duration_ms=(
                max(action["end_time_ns"] - action["start_time_ns"], 0) / 1_000_000
                if action["end_time_ns"] is not None
                else None
            ),
            status=action["status"],
            error=action["error"],
            source="tinyagent",
        )
        for index, action in enumerate(actions)
    ]


def _agent_actions(raw_trace: Any) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for span in getattr(raw_trace, "spans", None) or []:
        attributes = getattr(span, "attributes", None) or {}
        operation = attributes.get("gen_ai.operation.name")
        if operation not in {"call_llm", "execute_tool"}:
            continue
        start_time_ns = getattr(span, "start_time", None)
        if not isinstance(start_time_ns, int):
            start_time_ns = time.time_ns()
        end_time_ns = getattr(span, "end_time", None)
        if not isinstance(end_time_ns, int):
            end_time_ns = None
        status, error = _span_status(span)
        raw_output = _coerce_str(attributes.get("gen_ai.output"))
        if operation == "call_llm":
            messages = _parse_messages(attributes.get("gen_ai.input.messages"))
            usage = {
                key: value
                for key, attribute_name in {
                    "input_tokens": "gen_ai.usage.input_tokens",
                    "output_tokens": "gen_ai.usage.output_tokens",
                    "input_cost": "gen_ai.usage.input_cost",
                    "output_cost": "gen_ai.usage.output_cost",
                }.items()
                if isinstance((value := attributes.get(attribute_name)), (int, float))
                and not isinstance(value, bool)
            }
            actions.append(
                {
                    "type": "llm_call",
                    "name": _coerce_str(attributes.get("gen_ai.request.model"))
                    or "unknown",
                    "content": raw_output or None,
                    "arguments": {},
                    "result": None,
                    "input_messages": messages,
                    "usage": usage or None,
                    "start_time_ns": start_time_ns,
                    "end_time_ns": end_time_ns,
                    "status": status,
                    "error": error,
                    "source": "tinyagent",
                }
            )
        else:
            raw_arguments = _coerce_str(attributes.get("gen_ai.tool.args")) or "{}"
            arguments = _safe_parse(raw_arguments)
            actions.append(
                {
                    "type": "tool_call",
                    "name": _coerce_str(attributes.get("gen_ai.tool.name"))
                    or "unknown",
                    "content": raw_output or None,
                    "arguments": arguments if isinstance(arguments, dict) else {},
                    "result": _safe_parse(raw_output),
                    "input_messages": None,
                    "usage": None,
                    "start_time_ns": start_time_ns,
                    "end_time_ns": end_time_ns,
                    "status": status,
                    "error": error,
                    "source": "tinyagent",
                }
            )
    return actions


def _span_status(span: Any) -> tuple[TraceStatus, str | None]:
    status = getattr(span, "status", None)
    code = getattr(status, "status_code", None)
    value = getattr(code, "value", code)
    name = getattr(code, "name", None)
    normalized = str(name or value or "unset").lower()
    if normalized not in {"ok", "error", "unset"}:
        normalized = "unset"
    description = getattr(status, "description", None)
    return normalized, str(description) if description else None  # type: ignore[return-value]


def _memory_status(event: MemoryOperationEvent) -> TraceStatus:
    if event.status.value == "success":
        return "ok"
    if event.status.value == "dropped":
        # The operation was declined, not performed; the memory ledger keeps the
        # exact `dropped` status.
        return "skipped"
    return event.status.value  # type: ignore[return-value]


def _parse_messages(value: Any) -> list[dict[str, Any]] | None:
    if not value:
        return None
    parsed = _safe_parse(_coerce_str(value))
    if not isinstance(parsed, list):
        return None
    return [
        _json_safe(message) for message in parsed if isinstance(message, dict)
    ] or None


def _safe_parse(payload: str) -> Any:
    if not isinstance(payload, str) or not payload:
        return payload
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        pass
    try:
        return ast.literal_eval(payload)
    except (ValueError, SyntaxError):
        return payload


def _coerce_str(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _memory_event_content(result: Any) -> str | None:
    if isinstance(result, dict):
        for key in ("summary", "message", "returned_items"):
            if result.get(key) is not None:
                return _display_content(result[key])
    return _display_content(result)


def _display_content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
