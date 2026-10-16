import pytest

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
