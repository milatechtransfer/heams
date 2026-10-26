from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from heams.execution_traces import (
    ExecutionTraceArtifactError,
    ExecutionTraceCollector,
    ExecutionTraceLedger,
    TRACE_EVENT_KINDS,
    TraceEventKind,
    load_execution_traces,
    normalized_agent_events,
    trace_event_kind,
)
from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationSource,
    MemoryOperationStatus,
    MemoryOperationType,
)


def _span(
    operation: str,
    *,
    start: int,
    end: int,
    **attributes: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        attributes={"gen_ai.operation.name": operation, **attributes},
        start_time=start,
        end_time=end,
        status=SimpleNamespace(status_code="ok", description=None),
    )


def test_normalized_agent_events_preserve_action_content_and_order() -> None:
    trace = SimpleNamespace(
        spans=[
            _span(
                "execute_tool",
                start=300,
                end=350,
                **{
                    "gen_ai.tool.name": "recall",
                    "gen_ai.tool.args": '{"query": "fox"}',
                    "gen_ai.output": '["memory"]',
                },
            ),
            _span(
                "call_llm",
                start=100,
                end=200,
                **{
                    "gen_ai.request.model": "test-model",
                    "gen_ai.input.messages": (
                        '[{"role": "user", "content": "Where is the fox?"}]'
                    ),
                    "gen_ai.output": "I will recall.",
                    "gen_ai.usage.input_tokens": 12,
                    "gen_ai.usage.output_tokens": 4,
                },
            ),
        ]
    )

    events = normalized_agent_events(trace)

    assert [event.type for event in events] == ["llm_call", "tool_call"]
    assert events[0].content == "I will recall."
    assert events[0].input_messages == [
        {"role": "user", "content": "Where is the fox?"}
    ]
    assert events[0].usage == {"input_tokens": 12, "output_tokens": 4}
    assert events[1].name == "recall"
    assert events[1].arguments == {"query": "fox"}
    assert events[1].content == '["memory"]'
    assert events[1].result == ["memory"]


def test_collector_merges_framework_memory_and_agent_events() -> None:
    collector = ExecutionTraceCollector(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
    )
    collector.add_event(
        phase="priming",
        type="memory_priming",
        name="prime_memory_entries",
        content="1 entries",
        start_time_ns=100,
        end_time_ns=200,
        status="ok",
    )
    with collector.bind_context(phase="execution", step_index=2):
        collector.add_memory_event(
            MemoryOperationEvent(
                operation_type=MemoryOperationType.RETAIN,
                status=MemoryOperationStatus.SUCCESS,
                source=MemoryOperationSource.RUNTIME,
                duration_ms=1.0,
                payload={"content": "remember me"},
                result={"summary": "retained"},
            )
        )
    collector.add_agent_trace(
        SimpleNamespace(
            spans=[
                _span(
                    "call_llm",
                    start=300,
                    end=400,
                    **{
                        "gen_ai.request.model": "test-model",
                        "gen_ai.output": "answer",
                    },
                )
            ]
        ),
        step_index=3,
        stimulus_index=3,
    )

    record = collector.finalize(outcome="completed")

    assert record.run_id == "run-1"
    assert [event.type for event in record.events] == [
        "memory_priming",
        "llm_call",
        "memory_operation",
    ]
    memory_event = record.events[-1]
    assert memory_event.step_index == 2
    assert memory_event.content == "retained"
    assert memory_event.arguments == {"content": "remember me"}


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            {"message": "Successfully saved to memory.", "write_count": 1},
            "Successfully saved to memory.",
        ),
        ({"returned_items": ["a", "b"], "write_count": 0}, '["a", "b"]'),
        (
            {"write_count": 1, "stored_count": 2},
            '{"stored_count": 2, "write_count": 1}',
        ),
    ],
)
def test_memory_event_content_prefers_display_keys(
    result: object, expected: str
) -> None:
    collector = ExecutionTraceCollector(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
    )
    collector.add_memory_event(
        MemoryOperationEvent(
            operation_type=MemoryOperationType.RECALL,
            status=MemoryOperationStatus.SUCCESS,
            source=MemoryOperationSource.RUNTIME,
            duration_ms=1.0,
            payload={"query": "fox"},
            result=result,
        )
    )
    record = collector.finalize(outcome="completed")
    assert record.events[0].content == expected


@pytest.mark.parametrize(
    ("status", "operation_type", "expected"),
    [
        (MemoryOperationStatus.SUCCESS, MemoryOperationType.RETAIN, "ok"),
        (MemoryOperationStatus.DROPPED, MemoryOperationType.RETAIN, "skipped"),
        (
            MemoryOperationStatus.SKIPPED,
            MemoryOperationType.SET_RETENTION_POLICY,
            "skipped",
        ),
        (MemoryOperationStatus.ERROR, MemoryOperationType.RECALL, "error"),
    ],
)
def test_memory_event_status_maps_onto_trace_status(
    status: MemoryOperationStatus,
    operation_type: MemoryOperationType,
    expected: str,
) -> None:
    collector = ExecutionTraceCollector(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
    )
    collector.add_memory_event(
        MemoryOperationEvent(
            operation_type=operation_type,
            status=status,
            source=MemoryOperationSource.RUNTIME,
            duration_ms=1.0,
        )
    )
    record = collector.finalize(outcome="completed")
    assert record.events[0].status == expected
    assert record.events[0].name == operation_type.value


def test_execution_trace_ledger_recovers_partial_tail_and_rejects_duplicates(
    tmp_path: Path,
) -> None:
    collector = ExecutionTraceCollector(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
    )
    record = collector.finalize(outcome="completed")
    path = tmp_path / "execution_traces.jsonl"
    path.write_text(record.model_dump_json() + "\n" + '{"schema_version":1', "utf-8")

    with ExecutionTraceLedger(path) as ledger:
        with pytest.raises(ExecutionTraceArtifactError, match="Duplicate"):
            ledger.append(record)

    payloads = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert payloads == [json.loads(record.model_dump_json())]


def test_report_loader_warns_and_skips_malformed_and_partial_rows(
    tmp_path: Path,
) -> None:
    collector = ExecutionTraceCollector(
        run_id="run-1",
        scenario_id="scenario-1",
        agent_index=0,
        memory_index=0,
        evaluator_index=0,
        sample_index=0,
    )
    record = collector.finalize(outcome="completed")
    path = tmp_path / "execution_traces.jsonl"
    path.write_text(
        record.model_dump_json() + "\n" + "{bad json}\n" + '{"schema_version":1',
        "utf-8",
    )

    traces, warnings = load_execution_traces(path)

    assert traces == {record.trace_id: record}
    assert any("malformed execution trace" in warning for warning in warnings)
    assert any("truncated execution trace tail" in warning for warning in warnings)


def test_every_emitted_event_type_has_a_kind() -> None:
    """Event types written anywhere in the framework are mapped to a kind."""
    source_root = Path(__file__).resolve().parents[1] / "src" / "heams"
    emitted = {
        match
        for path in source_root.rglob("*.py")
        for match in re.findall(r'\btype="([a-z_]+)"', path.read_text(encoding="utf-8"))
        if path.parts[-2] in {"orchestration"} or path.name == "execution_traces.py"
    } | {"llm_call", "tool_call"}

    assert emitted <= set(TRACE_EVENT_KINDS), emitted - set(TRACE_EVENT_KINDS)
    assert trace_event_kind("something_new") is TraceEventKind.OTHER
    assert all(kind.label for kind in TraceEventKind)
