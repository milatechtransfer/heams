from __future__ import annotations

from heams.memory_systems import NoMemoryConfig, create_memory
from heams.memory_systems.shared_types import MemoryEntry
from heams.memory_systems.schemas import MemoryOperationStatus, MemoryOperationType


def test_runtime_storage_is_no_op_for_no_memory_config(tmp_path) -> None:
    config = NoMemoryConfig(name="no-storage", top_k=7)
    original_config = config.model_dump()

    resolved_config = config.with_runtime_storage(tmp_path / "runtime-storage")

    assert resolved_config is config
    assert resolved_config.model_dump() == original_config


def test_prime_memory_entries_is_a_no_op(tmp_path) -> None:
    memory = create_memory(
        NoMemoryConfig(),
        storage_dir=tmp_path,
    )

    memory.prime_memory_entries(
        [
            MemoryEntry(content="first", metadata={"role": "user"}),
            MemoryEntry(content="second", metadata={"role": "assistant"}),
        ]
    )

    assert memory.recall("") == []


def test_retain_skips_and_does_not_persist(tmp_path) -> None:
    ledger = []
    memory = create_memory(
        NoMemoryConfig(),
        storage_dir=tmp_path,
        event_sink=ledger,
    )

    message = memory.retain("remember this", metadata={"topic": "test"})

    assert message == "No-memory backend does not store memories."
    assert memory.recall("") == []
    assert len(ledger) == 2
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].status == MemoryOperationStatus.SKIPPED
    assert ledger[0].result["write_count"] == 0
    assert ledger[1].operation_type == MemoryOperationType.RECALL
    assert ledger[1].result["returned_items"] == []


def test_recall_always_returns_empty_list(tmp_path) -> None:
    memory = create_memory(
        NoMemoryConfig(top_k=7),
        storage_dir=tmp_path,
    )

    assert memory.recall("anything") == []
    assert memory.recall("anything", top_k=1) == []


def test_delete_and_consolidate_report_skipped(tmp_path) -> None:
    ledger = []
    memory = create_memory(
        NoMemoryConfig(),
        storage_dir=tmp_path,
        event_sink=ledger,
    )

    delete_message = memory.delete("anything")
    consolidate_message = memory.consolidate("recent context")

    assert delete_message == "No-memory backend has nothing to delete."
    assert consolidate_message == "No-memory backend has no consolidation step."
    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.DELETE,
        MemoryOperationType.CONSOLIDATE,
    ]
    assert all(event.status == MemoryOperationStatus.SKIPPED for event in ledger)
