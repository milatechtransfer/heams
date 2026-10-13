import pytest

from heams.memory_systems.api import MemoryOperationStatus, MemoryOperationType
from heams_memory_example.plugin import ExampleMemoryConfig, ExampleMemorySystem


def _memory(tmp_path, **kwargs) -> ExampleMemorySystem:
    return ExampleMemorySystem(
        config=ExampleMemoryConfig(top_k=10),
        storage_dir=tmp_path,
        **kwargs,
    )


def test_base_memory_system_emits_events_for_every_operation(tmp_path) -> None:
    ledger = []
    memory = _memory(tmp_path)
    memory.attach_event_sink(ledger)

    memory.retain("Hello!", metadata={"role": "user"})
    recall_results = memory.recall(query="hello")
    memory.delete("hello")
    memory.consolidate(context="maintenance", metadata={"source": "test"})

    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.RECALL,
        MemoryOperationType.DELETE,
        MemoryOperationType.CONSOLIDATE,
    ]
    assert all(event.status == MemoryOperationStatus.SUCCESS for event in ledger)
    assert ledger[0].payload["content"] == "Hello!"
    assert ledger[0].payload["metadata"] == {"role": "user"}
    assert ledger[1].payload["query"] == "hello"
    assert ledger[1].result["returned_items"] == recall_results
    assert ledger[2].payload["query"] == "hello"
    assert ledger[3].payload["context"] == "maintenance"
    assert ledger[3].payload["metadata"] == {"source": "test"}
    assert all(event.duration_ms is not None for event in ledger)


def test_delete_and_consolidate_reach_the_backend(tmp_path) -> None:
    ledger = []
    memory = _memory(tmp_path, event_sink=ledger)
    memory.retain("The user likes tea.")
    memory.retain("The user likes tea.")
    memory.retain("The user lives in Montreal.")

    memory.consolidate()
    memory.delete("montreal")

    remaining = memory.recall("")
    assert len(remaining) == 1
    assert "The user likes tea." in remaining[0]
    assert ledger[-3].result["changed_count"] == 1
    assert ledger[-2].result["deleted_count"] == 1


def test_base_memory_system_accepts_event_sink_and_run_context_at_construction(
    tmp_path,
) -> None:
    ledger = []
    memory = _memory(
        tmp_path,
        event_sink=ledger,
        run_context={"scenario_id": "test-scenario"},
    )

    memory.retain("Hello!")

    assert len(ledger) == 1
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].payload["run_context"] == {"scenario_id": "test-scenario"}


def test_base_memory_system_records_backend_errors(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = []
    memory = _memory(tmp_path, event_sink=ledger)

    def _boom(*args, **kwargs):
        raise RuntimeError("backend exploded")

    monkeypatch.setattr(memory, "_consolidate_impl", _boom)

    memory.consolidate()

    assert ledger[-1].operation_type == MemoryOperationType.CONSOLIDATE
    assert ledger[-1].status == MemoryOperationStatus.ERROR
    assert "backend exploded" in (ledger[-1].error or "")
