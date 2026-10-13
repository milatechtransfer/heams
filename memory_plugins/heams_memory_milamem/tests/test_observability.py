import json

import pytest

from heams_memory_milamem.basic_complete_memory_system import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
)
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
)
from heams_memory_milamem.dedup_strategies import DedupStrategyConfig, DedupStrategyKind
from heams_memory_milamem.design.delete.strategies import (
    llm_judge as delete_strategies_module,
)
from heams_memory_milamem.design.delete.strategies import DeleteDecision
from heams.memory_systems.api import MemoryOperationStatus, MemoryOperationType


def _decode_recall_item(raw: str) -> tuple[str, dict[str, object]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {}

    if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
        metadata = parsed.get("metadata")
        if isinstance(metadata, dict):
            return parsed["content"], metadata
    return raw, {}


def test_basic_canonical_retain_event_includes_temporal_summary_when_enabled(
    tmp_path,
) -> None:
    ledger = []
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="basic-canonical-memory-system",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
            temporal_metadata={"enabled": True},
        ),
        storage_dir=tmp_path,
        event_sink=ledger,
    )

    memory.retain("Timestamped fact", {"timestamp": "2024-03-01 10:00:00"})

    assert len(ledger) == 1
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].result["raw_write_count"] == 1
    assert ledger[0].result["canonical_write_count"] == 1
    assert ledger[0].result["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert ledger[0].result["temporal_source"] == "metadata.timestamp"


def test_basic_canonical_retain_event_omits_temporal_summary_when_disabled(
    tmp_path,
) -> None:
    ledger = []
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="basic-canonical-memory-system",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        ),
        storage_dir=tmp_path,
        event_sink=ledger,
    )

    memory.retain("Timestamped fact", {"timestamp": "2024-03-01 10:00:00"})

    assert len(ledger) == 1
    assert ledger[0].result["raw_write_count"] == 1
    assert ledger[0].result["canonical_write_count"] == 1
    assert "temporal_received_at" not in ledger[0].result
    assert "temporal_source" not in ledger[0].result


def test_basic_canonical_delete_observability_reports_success_and_skipped(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = []
    memory = MilaMemMemorySystem(
        config=MilaMemConfig(
            name="basic-canonical-memory-system",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            dedup_strategy=DedupStrategyConfig(kind=DedupStrategyKind.SEMANTIC),
        ),
        storage_dir=tmp_path,
        event_sink=ledger,
    )
    memory.retain("The user likes jasmine tea.")
    target_id = memory._load_canonical_store().records[0].id
    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        lambda model, messages: DeleteDecision(
            selected_ids=[target_id],
            reasoning="The candidate is directly targeted.",
            ambiguous=False,
        ),
    )

    deleted = memory.delete("Forget jasmine tea.")
    skipped = memory.delete("")

    assert deleted == "Logically deleted 1 canonical memory record(s)."
    assert skipped == "Delete request requires a non-empty query."
    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.DELETE,
        MemoryOperationType.DELETE,
    ]
    success_event = ledger[1]
    assert success_event.status == MemoryOperationStatus.SUCCESS
    assert success_event.payload["query"] == "Forget jasmine tea."
    assert success_event.result["supported"] is True
    assert success_event.result["strategy"] == "exact_then_top_k_llm"
    assert success_event.result["planning_mode"] == "none"
    assert success_event.result["selection_mode"] == "exact_then_top_k_llm"
    assert success_event.result["target_text"] == "jasmine tea."
    assert success_event.result["cardinality"] == "one"
    assert success_event.result["exclusions"] == []
    assert success_event.result["selected_ids"] == [target_id]
    assert success_event.result["deleted_count"] == 1
    assert success_event.result["verification_status"] == "verified"
    skipped_event = ledger[2]
    assert skipped_event.status == MemoryOperationStatus.SKIPPED
    assert skipped_event.payload["query"] == ""
    assert skipped_event.result["supported"] is True
    assert skipped_event.result["selected_ids"] == []
    assert skipped_event.result["deleted_count"] == 0
    assert skipped_event.result["verification_status"] == "not_run"


def test_basic_complete_memory_system_emits_retain_recall_consolidate_events(
    tmp_path,
) -> None:
    ledger = []
    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="basic-complete-memory-system",
            top_k=1,
            file_path=str(tmp_path / "memory.json"),
        ),
        storage_dir=tmp_path,
    )
    memory.attach_event_sink(ledger)

    retained = memory.retain("Hello!", metadata={"role": "user"})
    recall_results = memory.recall(query="hello")
    consolidated = memory.consolidate()

    assert retained == "Successfully saved to memory."
    assert _decode_recall_item(recall_results[0]) == ("Hello!", {"role": "user"})
    assert consolidated == "No active memories required consolidation."

    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.RECALL,
        MemoryOperationType.CONSOLIDATE,
    ]
    assert ledger[0].result["write_count"] == 1
    assert ledger[1].payload["query"] == "hello"
    assert ledger[1].result["returned_items"] == recall_results
    assert ledger[1].result["scores"][0] is not None
    assert ledger[2].status == MemoryOperationStatus.SUCCESS
    assert ledger[2].result["summary"] == consolidated
    assert ledger[2].result["supported"] is True
    assert ledger[2].result["changed_count"] == 0
    assert ledger[2].result["deduplicated_count"] == 0
    assert ledger[2].result["conflict_resolved_count"] == 0


def test_basic_complete_memory_system_delete_emits_expected_event_shape(
    tmp_path,
) -> None:
    ledger = []
    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="basic-complete-memory-system",
            top_k=10,
            file_path=str(tmp_path / "memory.json"),
        ),
        storage_dir=tmp_path,
    )
    memory.attach_event_sink(ledger)

    memory.retain("Employer is State University", {"ts": 10})
    deleted = memory.delete("state university")
    ambiguous = memory.delete("state")

    assert deleted == "Deleted 1 memory record."
    assert ambiguous == "Delete request did not match any active memory."
    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.DELETE,
        MemoryOperationType.DELETE,
    ]
    assert ledger[1].status == MemoryOperationStatus.SUCCESS
    assert ledger[1].payload["query"] == "state university"
    assert ledger[1].result["deleted_count"] == 1
    assert ledger[1].result["matched_content_preview"] == "Employer is State University"
    assert ledger[2].status == MemoryOperationStatus.SKIPPED
    assert ledger[2].result["deleted_count"] == 0
    assert ledger[2].result["supported"] is True
    assert ledger[2].result["summary"] == ambiguous


def test_basic_complete_memory_system_consolidate_emits_expected_event_shape(
    tmp_path,
) -> None:
    ledger = []
    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            name="basic-complete-memory-system",
            top_k=10,
            file_path=str(tmp_path / "memory.json"),
        ),
        storage_dir=tmp_path,
    )
    memory.attach_event_sink(ledger)

    memory.retain("Alice works at State University", {"ts": 10})
    memory.retain("Bob works at Mila", {"ts": 11})
    consolidated = memory.consolidate(
        context="maintenance", metadata={"source": "test"}
    )

    assert consolidated == "No active memories required consolidation."
    assert [event.operation_type for event in ledger] == [
        MemoryOperationType.RETAIN,
        MemoryOperationType.RETAIN,
        MemoryOperationType.CONSOLIDATE,
    ]
    assert ledger[2].status == MemoryOperationStatus.SUCCESS
    assert ledger[2].payload["context"] == "maintenance"
    assert ledger[2].payload["metadata"] == {"source": "test"}
    assert ledger[2].result["summary"] == consolidated
    assert ledger[2].result["supported"] is True
    assert ledger[2].result["changed_count"] == 0
    assert ledger[2].result["deduplicated_count"] == 0
    assert ledger[2].result["conflict_resolved_count"] == 0


def test_basic_complete_memory_system_accepts_event_sink_at_construction_time(
    tmp_path,
) -> None:
    ledger = []
    memory = BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            file_path=str(tmp_path / "memory.json"),
        ),
        storage_dir=tmp_path,
        event_sink=ledger,
        run_context={"scenario_id": "test-scenario"},
    )

    memory.retain("Hello!")

    assert len(ledger) == 1
    assert ledger[0].operation_type == MemoryOperationType.RETAIN
    assert ledger[0].payload["run_context"] == {"scenario_id": "test-scenario"}
