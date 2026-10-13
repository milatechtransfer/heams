"""Tests for Mem0's implementation of the public observability contract."""

import json
from unittest.mock import MagicMock

import pytest

from heams.memory_systems.api import (
    MemoryEntry,
    MemoryOperationStatus,
    MemoryOperationType,
)
from heams_memory_mem0 import Mem0MemoryConfig, Mem0MemorySystem


@pytest.fixture()
def mock_mem0_class(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_class.from_config.return_value = mock_instance
    monkeypatch.setattr("heams_memory_mem0.plugin.Memory", mock_class)
    return mock_class


def test_mem0_emits_retain_recall_consolidate_events_without_duplicate_retain(
    mock_mem0_class: MagicMock,
    tmp_path,
) -> None:
    ledger = []
    mem0_instance = mock_mem0_class.return_value
    mem0_instance.add.return_value = {"results": [{"id": "retain-1"}]}
    mem0_instance.search.return_value = {
        "results": [
            {"memory": "fact with metadata", "metadata": {"tag": "x"}, "score": 0.8}
        ]
    }

    memory = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="test_user", top_k=3),
        storage_dir=tmp_path,
    )
    memory.attach_event_sink(ledger)

    retained = memory.retain("Alice lives in Berlin.", metadata={"role": "user"})
    recalled = memory.recall("where does Alice live", top_k=5)
    consolidated = memory.consolidate(
        context="summary",
        metadata={"source": "test"},
    )

    assert retained == "Memory retained successfully."
    parsed = json.loads(recalled[0])
    assert parsed["content"] == "fact with metadata"
    assert parsed["metadata"] == {"tag": "x"}
    assert consolidated == "This memory system has no consolidation step."

    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.RECALL,
        MemoryOperationType.CONSOLIDATE,
    ]
    assert ledger[0].result["write_count"] == 1
    assert ledger[0].payload["metadata"] == {"role": "user"}
    assert ledger[1].payload["requested_top_k"] == 5
    assert ledger[1].payload["effective_top_k"] == 5
    assert ledger[1].result["returned_items"] == recalled
    assert ledger[1].result["scores"] == [0.8]
    assert ledger[2].status == MemoryOperationStatus.SKIPPED
    assert ledger[2].payload["context"] == "summary"
    assert ledger[2].payload["metadata"] == {"source": "test"}
    assert ledger[2].result["summary"] == consolidated
    assert ledger[2].result["supported"] is False
    assert ledger[2].result["changed_count"] == 0


def test_mem0_accepts_event_sink_at_construction_time(
    mock_mem0_class: MagicMock,
    tmp_path,
) -> None:
    ledger = []
    mem0_instance = mock_mem0_class.return_value
    mem0_instance.add.return_value = {"results": [{"id": "r1"}]}

    memory = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="test_user"),
        storage_dir=tmp_path,
        event_sink=ledger,
        run_context={"scenario_id": "test-scenario"},
    )

    memory.retain("Hello!")

    assert len(ledger) == 1
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].payload["run_context"] == {"scenario_id": "test-scenario"}


def test_mem0_priming_emits_retain_event_for_deterministic_startup(
    mock_mem0_class: MagicMock,
    tmp_path,
) -> None:
    ledger = []
    mem0_instance = mock_mem0_class.return_value
    mem0_instance.add.return_value = {"results": [{"id": "prime-1"}]}

    memory = Mem0MemorySystem(
        config=Mem0MemoryConfig(
            user_id="test_user",
            top_k=3,
            temporal_prompt_enabled=True,
        ),
        storage_dir=tmp_path,
    )
    memory.attach_event_sink(ledger)

    memory.prime_memory_entries(
        [
            MemoryEntry(
                content="Alice: I moved to Montreal last spring.",
                metadata={
                    "source": "locomo",
                    "source_benchmark": "locomo",
                    "session_id": "session_1",
                    "timestamp": "2024-03-01 10:00:00",
                    "ingest_group_id": "group-1",
                    "ingest_order": 0,
                    "ingest_role": "user",
                    "ingest_timestamp_epoch": 1709287200,
                },
            )
        ]
    )

    assert len(ledger) == 1
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].status == MemoryOperationStatus.SUCCESS
    assert ledger[0].payload["is_priming"] is True
    assert ledger[0].payload["entry_count"] == 1
    assert ledger[0].payload["input_entries"][0]["content"] == (
        "Alice: I moved to Montreal last spring."
    )
    assert ledger[0].payload["input_entries"][0]["metadata"] == {
        "source": "locomo",
        "source_benchmark": "locomo",
        "session_id": "session_1",
        "timestamp": "2024-03-01 10:00:00",
        "ingest_group_id": "group-1",
        "ingest_order": 0,
        "ingest_role": "user",
        "ingest_timestamp_epoch": 1709287200,
    }
    assert ledger[0].result["memory"] == [
        {
            "content": [
                    {
                        "role": "system",
                        "content": (
                            "Temporal context for extraction only; do not treat this "
                            "message as source conversation content. Today is "
                            "2024-03-01 for this conversation. Use this as the "
                            "authoritative reference date when resolving relative "
                            "temporal expressions, rather than the current system date. "
                            "Express resolved dates explicitly in extracted memories "
                            "without rewriting the input messages."
                        ),
                    },
                {
                    "role": "user",
                    "content": "Alice: I moved to Montreal last spring.",
                }
            ],
            "metadata": {
                "source": "locomo",
                "source_benchmark": "locomo",
                "session_id": "session_1",
                "timestamp": "2024-03-01 10:00:00",
            },
            "entry_count": 1,
            "timestamp": 1709287200,
        }
    ]
    assert ledger[0].result["write_count"] == 1
    assert ledger[0].result["saved_content"] == [
        {
            "content": [
                {
                    "role": "system",
                    "content": (
                        "Temporal context for extraction only; do not treat this "
                        "message as source conversation content. Today is "
                        "2024-03-01 for this conversation. Use this as the "
                        "authoritative reference date when resolving relative "
                        "temporal expressions, rather than the current system date. "
                        "Express resolved dates explicitly in extracted memories "
                        "without rewriting the input messages."
                    ),
                },
                {
                    "role": "user",
                    "content": "Alice: I moved to Montreal last spring.",
                }
            ],
            "metadata": {
                "source": "locomo",
                "source_benchmark": "locomo",
                "session_id": "session_1",
                "timestamp": "2024-03-01 10:00:00",
            },
            "entry_count": 1,
            "timestamp": 1709287200,
        }
    ]
