from pathlib import Path

from heams.memory_systems.schemas import (
    MemoryOperationEvent,
    MemoryOperationSource,
    MemoryOperationStatus,
    MemoryOperationType,
)
from heams.memory_systems import create_memory
from heams_memory_example.plugin import ExampleMemoryConfig
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)


class StubMemorySystem(BaseMemorySystem):
    def _retain_impl(self, content: str, metadata=None) -> MemoryOperationResult:
        return MemoryOperationResult(value=content)

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        return MemoryOperationResult(value=[query])


def test_log_memory_event_appends_to_ledger_and_attaches_run_context() -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = StubMemorySystem(
        BaseMemoryConfig(type="stub", name="stub-memory"),
        storage_dir=Path(),
        event_sink=ledger,
        run_context={"scenario_id": "scenario-a", "sample_index": 0},
    )

    event = memory._log_memory_event(
        operation_type=MemoryOperationType.RECALL,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.TINY_AGENT,
        payload={"query": "favorite color"},
        result={"items_returned": 1},
    )

    assert ledger == [event]
    assert event.component == "stub"
    assert event.payload == {
        "query": "favorite color",
        "run_context": {"scenario_id": "scenario-a", "sample_index": 0},
    }
    assert event.result == {"items_returned": 1}


def test_log_memory_event_accepts_prebuilt_event_and_callable_sink() -> None:
    captured: list[MemoryOperationEvent] = []
    memory = StubMemorySystem(BaseMemoryConfig(type="stub"), storage_dir=Path())
    memory.attach_event_sink(captured.append)
    memory.set_run_context({"scenario_id": "scenario-b"})

    original = MemoryOperationEvent(
        operation_type=MemoryOperationType.RETAIN,
        status=MemoryOperationStatus.ERROR,
        source=MemoryOperationSource.TOOL_CALL,
        component="custom-component",
        payload={"content_length": 12},
        error="boom",
    )

    emitted = memory._log_memory_event(original)

    assert captured == [emitted]
    assert emitted.component == "custom-component"
    assert emitted.payload == {
        "content_length": 12,
        "run_context": {"scenario_id": "scenario-b"},
    }
    assert original.payload == {"content_length": 12}


def test_create_memory_constructor_paths_remain_compatible(tmp_path) -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = create_memory(
        ExampleMemoryConfig(name="t", top_k=2),
        storage_dir=tmp_path,
        event_sink=ledger,
        run_context={"scenario_id": "scenario-c"},
    )

    assert memory.config.name == "t"
    assert memory.recall("") == []
    memory.retain("hello")
    assert len(ledger) == 2
    assert ledger[1].operation_type == MemoryOperationType.RETAIN
    assert ledger[1].payload["run_context"] == {"scenario_id": "scenario-c"}


def test_log_memory_event_accepts_explicit_run_context() -> None:
    ledger: list[MemoryOperationEvent] = []
    memory = StubMemorySystem(
        BaseMemoryConfig(type="stub", name="stub-memory"),
        storage_dir=Path(),
        event_sink=ledger,
        run_context={"scenario_id": "scenario-a", "sample_index": 0},
    )

    memory._log_memory_event(
        operation_type=MemoryOperationType.RECALL,
        status=MemoryOperationStatus.SUCCESS,
        source=MemoryOperationSource.RUNTIME,
        payload={"query": "favorite color"},
        run_context={"stimulus_index": 3, "question_id": "q3"},
    )

    assert ledger[0].payload["run_context"] == {
        "scenario_id": "scenario-a",
        "sample_index": 0,
        "stimulus_index": 3,
        "question_id": "q3",
    }
