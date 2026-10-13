from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.milamem import (
    MilaMemMemorySystem,
    MilaMemConfig,
    CanonicalMemoryStoreState,
    RawMemoryStoreState,
)
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)
from heams_memory_milamem.design.delete.config import SelectionMode
from heams_memory_milamem.design.delete.models import DeleteQueryPlan
from heams_memory_milamem.design.delete.strategies import (
    BaseDeleteStrategy,
    DeleteCandidate,
    DeleteDecision,
    DeleteStrategyConfig,
)
from heams_memory_milamem.design.delete.strategies import (
    llm_judge as delete_strategies_module,
)
from heams_memory_milamem.design.storage_type import (
    CanonicalMemoryRecord,
    RawMemoryRecord,
)
from heams_memory_milamem.recall_strategies import (
    RecallStrategyConfig,
    RecallStrategyKind,
)


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        vectors.append(
            [int(digest[index : index + 2], 16) / 255.0 for index in range(0, 16, 2)]
        )
    return vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


class FixedDeleteStrategy(BaseDeleteStrategy):
    def __init__(self, decision: DeleteDecision):
        self.decision = decision
        self.calls: list[tuple[DeleteQueryPlan, list[DeleteCandidate]]] = []

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        self.calls.append((query_plan, candidates))
        return self.decision


def _memory(
    tmp_path: Path,
    *,
    max_delete_count: int = 1,
) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            name="canonical",
            raw_store_path=str(tmp_path / "raw.json"),
            canonical_store_path=str(tmp_path / "canonical.json"),
            recall_strategy=RecallStrategyConfig(kind=RecallStrategyKind.BM25),
            dedup_strategy=DedupStrategyConfig(
                enabled=False,
                kind=DedupStrategyKind.BM25,
            ),
            delete_strategy=DeleteStrategyConfig(
                max_delete_count=max_delete_count,
            ),
        ),
        storage_dir=tmp_path,
    )


def _canonical_record(
    memory_id: str,
    sequence: int,
    *,
    status: str = "active",
    deleted_at: float | None = None,
) -> CanonicalMemoryRecord:
    raw_id = f"raw-{memory_id}"
    return CanonicalMemoryRecord(
        id=memory_id,
        raw_source_id=raw_id,
        content=f"Content for {memory_id}",
        sequence=sequence,
        provenance={"raw_record_id": raw_id},
        status=status,
        deleted_at=deleted_at,
        superseded_by="active-2" if status == "superseded" else None,
        consolidation_reason="conflict_resolved" if status == "superseded" else None,
    )


def _raw_record(memory_id: str, sequence: int) -> RawMemoryRecord:
    return RawMemoryRecord(
        id=f"raw-{memory_id}",
        content=f"Content for {memory_id}",
        sequence=sequence,
    )


def _delete_request_records(
    memory: MilaMemMemorySystem,
) -> list[RawMemoryRecord]:
    return [
        record
        for record in memory._load_raw_store().records
        if record.metadata.get("record_type") == "delete_request"
    ]


def _candidate(memory_id: str, score: float = 1.0) -> DeleteCandidate:
    return DeleteCandidate(
        memory_id=memory_id,
        content=f"Content for {memory_id}",
        score=score,
    )


def _set_delete_strategy(
    memory: MilaMemMemorySystem,
    strategy: BaseDeleteStrategy,
) -> None:
    delete_config = memory.config.delete_strategy.model_copy(
        update={"selection_mode": SelectionMode.TOP_K_LLM}
    )
    memory.config.delete_strategy = delete_config
    memory._delete_lifecycle.config = delete_config
    memory._delete_strategy = strategy
    memory._delete_lifecycle.strategy = strategy


def test_delete_impl_skips_empty_query_without_selection_or_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_canonical_record("active-1", 1)])
    )
    canonical_path = Path(memory.config.canonical_store_path)
    before = canonical_path.read_bytes()
    strategy = FixedDeleteStrategy(
        DeleteDecision(selected_ids=["active-1"], reasoning="unused", ambiguous=False)
    )
    _set_delete_strategy(memory, strategy)
    monkeypatch.setattr(
        memory,
        "_load_canonical_store",
        lambda: pytest.fail("Canonical store must not load for an empty query"),
    )

    result = memory._delete_impl("   ")

    assert result.status == "skipped"
    assert result.result == {
        "summary": "Delete request requires a non-empty query.",
        "message": "Delete request requires a non-empty query.",
        "supported": True,
        "strategy": "top_k_llm",
        "planning_mode": "none",
        "selection_mode": "top_k_llm",
        "target_text": "",
        "cardinality": "one",
        "exclusions": [],
        "candidate_count": 0,
        "selected_ids": [],
        "deleted_count": 0,
        "ambiguous": False,
        "reasoning": "The deletion query was empty.",
        "raw_erasure": False,
        "verification_status": "not_run",
    }
    assert strategy.calls == []
    assert canonical_path.read_bytes() == before


@pytest.mark.parametrize(
    "decision",
    [
        DeleteDecision(
            selected_ids=[],
            reasoning="Multiple interpretations are plausible.",
            ambiguous=True,
        ),
        DeleteDecision(
            selected_ids=[],
            reasoning="No candidate is directly targeted.",
            ambiguous=False,
        ),
    ],
)
def test_delete_impl_skips_ambiguous_or_empty_selection_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: DeleteDecision,
) -> None:
    memory = _memory(tmp_path)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_canonical_record("active-1", 1)])
    )
    canonical_path = Path(memory.config.canonical_store_path)
    before = canonical_path.read_bytes()
    strategy = FixedDeleteStrategy(decision)
    _set_delete_strategy(memory, strategy)
    save_calls: list[CanonicalMemoryStoreState] = []
    monkeypatch.setattr(
        memory,
        "_save_canonical_store",
        lambda store: save_calls.append(store),
    )

    result = memory._delete_impl("  forget this fact  ")

    assert result.status == "skipped"
    assert result.result["supported"] is True
    assert result.result["candidate_count"] == 1
    assert result.result["selected_ids"] == []
    assert result.result["deleted_count"] == 0
    assert result.result["ambiguous"] is decision.ambiguous
    assert result.result["reasoning"] == decision.reasoning
    assert result.result["raw_erasure"] is False
    assert result.result["verification_status"] == "not_run"
    assert strategy.calls[0][0].original_query == "forget this fact"
    assert [candidate.memory_id for candidate in strategy.calls[0][1]] == ["active-1"]
    assert save_calls == []
    assert canonical_path.read_bytes() == before


def test_delete_impl_tombstones_selected_active_records_and_appends_raw_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path, max_delete_count=2)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(
            records=[
                _canonical_record("active-1", 1),
                _canonical_record("superseded-1", 2, status="superseded"),
                _canonical_record("active-2", 3),
                _canonical_record("deleted-1", 4, status="deleted", deleted_at=10.0),
            ]
        )
    )
    memory._save_raw_store(
        RawMemoryStoreState(
            records=[
                _raw_record("active-1", 1),
                _raw_record("superseded-1", 2),
                _raw_record("active-2", 3),
                _raw_record("deleted-1", 4),
            ]
        )
    )
    raw_before = memory._load_raw_store().records
    cache_path = tmp_path / "canonical.embeddings.json"
    cache_path.write_text('{"existing": [1.0]}', encoding="utf-8")
    cache_before = cache_path.read_bytes()
    reasoning = "Both active facts are directly targeted."
    strategy = FixedDeleteStrategy(
        DeleteDecision(
            selected_ids=["active-1", "active-2"],
            reasoning=reasoning,
            ambiguous=False,
        )
    )
    _set_delete_strategy(memory, strategy)
    original_save = memory._save_canonical_store
    save_calls: list[CanonicalMemoryStoreState] = []

    def recording_save(store: CanonicalMemoryStoreState) -> None:
        save_calls.append(store)
        original_save(store)

    monkeypatch.setattr(memory, "_save_canonical_store", recording_save)

    result = memory._delete_impl("forget both active facts")

    assert result.status == "success"
    assert result.result == {
        "summary": "Logically deleted 2 canonical memory record(s).",
        "message": "Logically deleted 2 canonical memory record(s).",
        "supported": True,
        "strategy": "top_k_llm",
        "planning_mode": "none",
        "selection_mode": "top_k_llm",
        "target_text": "both active facts",
        "cardinality": "one",
        "exclusions": [],
        "candidate_count": 2,
        "selected_ids": ["active-1", "active-2"],
        "deleted_count": 2,
        "ambiguous": False,
        "reasoning": reasoning,
        "raw_erasure": False,
        "verification_status": "verified",
    }
    assert len(save_calls) == 1
    records = {record.id: record for record in memory._load_canonical_store().records}
    assert records["active-1"].status == "deleted"
    assert records["active-2"].status == "deleted"
    assert records["active-1"].deleted_at is not None
    assert records["active-1"].deleted_at == records["active-2"].deleted_at
    assert records["superseded-1"].status == "superseded"
    assert records["superseded-1"].superseded_by == "active-2"
    assert records["deleted-1"].deleted_at == 10.0
    raw_after = memory._load_raw_store().records
    assert raw_after[: len(raw_before)] == raw_before
    assert [record.content for record in _delete_request_records(memory)] == [
        "forget both active facts"
    ]
    assert cache_path.read_bytes() == cache_before


def test_delete_impl_uses_matching_message_for_broad_deletion(tmp_path: Path) -> None:
    memory = _memory(tmp_path, max_delete_count=2)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(
            records=[
                _canonical_record("project-1", 1).model_copy(
                    update={"content": "Project Titan"}
                ),
                _canonical_record("project-2", 2).model_copy(
                    update={"content": "Project Titan"}
                ),
            ]
        )
    )

    result = memory._delete_impl("Delete everything about Project Titan")

    assert result.value == "Logically deleted 2 matching canonical memory record(s)."
    assert result.result["cardinality"] == "all"
    assert result.result["target_text"] == "Project Titan"
    assert result.result["deleted_count"] == 2


def test_delete_impl_deduplicates_ids_from_injected_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(records=[_canonical_record("active-1", 1)])
    )
    _set_delete_strategy(
        memory,
        FixedDeleteStrategy(
            DeleteDecision(
                selected_ids=["active-1", "active-1"],
                reasoning="Injected strategy returned a duplicate ID.",
                ambiguous=False,
            )
        ),
    )

    result = memory._delete_impl("forget this fact")

    assert result.result["selected_ids"] == ["active-1"]
    assert result.result["deleted_count"] == 1
    assert result.result["verification_status"] == "verified"
    record = memory._load_canonical_store().records[0]
    assert record.status == "deleted"


@pytest.mark.parametrize(
    ("selected_id", "candidate_id"),
    [("unknown", "active-1"), ("deleted-1", "deleted-1")],
)
def test_delete_impl_rejects_foreign_or_no_longer_active_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_id: str,
    candidate_id: str,
) -> None:
    memory = _memory(tmp_path)
    memory._save_canonical_store(
        CanonicalMemoryStoreState(
            records=[
                _canonical_record("active-1", 1),
                _canonical_record("deleted-1", 2, status="deleted", deleted_at=10.0),
            ]
        )
    )
    canonical_path = Path(memory.config.canonical_store_path)
    before = canonical_path.read_bytes()
    decision = DeleteDecision(
        selected_ids=[selected_id],
        reasoning="Stale or invalid selection.",
        ambiguous=False,
    )
    _set_delete_strategy(memory, FixedDeleteStrategy(decision))
    save_calls: list[CanonicalMemoryStoreState] = []
    monkeypatch.setattr(
        memory,
        "_save_canonical_store",
        lambda store: save_calls.append(store),
    )

    result = memory._delete_impl("forget a fact")

    assert result.status == "skipped"
    assert result.result["selected_ids"] == []
    assert result.result["deleted_count"] == 0
    assert result.result["verification_status"] == "not_run"
    assert save_calls == []
    assert canonical_path.read_bytes() == before


@pytest.mark.parametrize(
    ("decision", "expected_saves"),
    [
        (
            DeleteDecision(
                selected_ids=["active-1"],
                reasoning="The candidate is directly targeted.",
                ambiguous=False,
            ),
            1,
        ),
        (
            DeleteDecision(
                selected_ids=[],
                reasoning="No candidate is directly targeted.",
                ambiguous=False,
            ),
            0,
        ),
    ],
)
def test_delete_impl_loads_once_and_saves_only_when_changed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: DeleteDecision,
    expected_saves: int,
) -> None:
    memory = _memory(tmp_path)
    canonical_store = CanonicalMemoryStoreState(
        records=[_canonical_record("active-1", 1)]
    )
    load_calls: list[None] = []
    save_calls: list[CanonicalMemoryStoreState] = []

    def load_store() -> CanonicalMemoryStoreState:
        load_calls.append(None)
        return canonical_store

    monkeypatch.setattr(memory, "_load_canonical_store", load_store)
    monkeypatch.setattr(
        memory,
        "_save_canonical_store",
        lambda store: save_calls.append(store),
    )
    _set_delete_strategy(memory, FixedDeleteStrategy(decision))

    memory._delete_impl("forget this fact")

    assert load_calls == [None]
    assert len(save_calls) == expected_saves


def test_public_delete_with_no_candidates_skips_without_calling_llm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        lambda model, messages: pytest.fail("LLM must not run without candidates"),
    )

    result = memory.delete("forget my address")

    assert result == "Delete request did not match any canonical memory."
    assert not Path(memory.config.canonical_store_path).exists()
    assert [record.content for record in _delete_request_records(memory)] == [
        "forget my address"
    ]


def test_public_llm_delete_tombstones_target_and_audits_request_raw_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    memory.retain("The user likes jasmine tea.")
    memory.retain("The user likes dark coffee.")
    canonical_before = memory._load_canonical_store()
    target = next(
        record for record in canonical_before.records if "jasmine tea" in record.content
    )
    survivor = next(
        record for record in canonical_before.records if "dark coffee" in record.content
    )
    raw_before = memory._load_raw_store().records

    def select_target(model: str, messages: list[dict[str, object]]) -> DeleteDecision:
        return DeleteDecision(
            selected_ids=[target.id],
            reasoning="The tea memory is directly targeted.",
            ambiguous=False,
        )

    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        select_target,
    )

    result = memory.delete("Forget that I like jasmine tea.")

    assert result == "Logically deleted 1 canonical memory record(s)."
    records = {record.id: record for record in memory._load_canonical_store().records}
    assert records[target.id].status == "deleted"
    assert records[target.id].deleted_at is not None
    assert records[survivor.id].status == "active"
    assert memory.recall("jasmine tea", top_k=5) == ["The user likes dark coffee."]
    raw_after = memory._load_raw_store().records
    assert raw_after[: len(raw_before)] == raw_before
    assert [record.content for record in _delete_request_records(memory)] == [
        "Forget that I like jasmine tea."
    ]
    assert all(
        "Forget that I like jasmine tea." not in record.content
        for record in memory._load_canonical_store().records
    )


def test_public_delete_llm_exception_performs_no_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    memory.retain("The user likes jasmine tea.")
    canonical_path = Path(memory.config.canonical_store_path)
    canonical_before = canonical_path.read_bytes()
    raw_before = memory._load_raw_store().records

    def raise_llm_error(
        model: str, messages: list[dict[str, object]]
    ) -> DeleteDecision:
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        raise_llm_error,
    )

    result = memory.delete("Forget jasmine tea.")

    assert result == "Delete request was ambiguous; no canonical memory was deleted."
    assert canonical_path.read_bytes() == canonical_before
    raw_after = memory._load_raw_store().records
    assert raw_after[: len(raw_before)] == raw_before
    assert [record.content for record in _delete_request_records(memory)] == [
        "Forget jasmine tea."
    ]


def test_repeated_public_delete_is_idempotent_and_preserves_related_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path)
    memory.retain("The user likes jasmine tea.")
    memory.retain("The user likes green tea.")
    records = memory._load_canonical_store().records
    jasmine = next(record for record in records if "jasmine" in record.content)
    green = next(record for record in records if "green" in record.content)
    raw_before = memory._load_raw_store().records
    llm_calls = 0

    def always_select_original(
        model: str,
        messages: list[dict[str, object]],
    ) -> DeleteDecision:
        nonlocal llm_calls
        llm_calls += 1
        return DeleteDecision(
            selected_ids=[jasmine.id],
            reasoning="Only jasmine tea is targeted.",
            ambiguous=False,
        )

    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        always_select_original,
    )

    first = memory.delete("Forget jasmine tea.")
    after_first = Path(memory.config.canonical_store_path).read_bytes()
    second = memory.delete("Forget jasmine tea.")

    assert first == "Logically deleted 1 canonical memory record(s)."
    assert second == "Delete request was ambiguous; no canonical memory was deleted."
    assert Path(memory.config.canonical_store_path).read_bytes() == after_first
    final_records = {
        record.id: record for record in memory._load_canonical_store().records
    }
    assert final_records[jasmine.id].status == "deleted"
    assert final_records[green.id].status == "active"
    raw_after = memory._load_raw_store().records
    assert raw_after[: len(raw_before)] == raw_before
    assert [record.content for record in _delete_request_records(memory)] == [
        "Forget jasmine tea.",
        "Forget jasmine tea.",
    ]
    assert llm_calls == 2


def test_public_delete_enforces_max_delete_count_without_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory = _memory(tmp_path, max_delete_count=1)
    memory.retain("The user likes jasmine tea.")
    memory.retain("The user likes dark coffee.")
    selected_ids = [record.id for record in memory._load_canonical_store().records]
    canonical_path = Path(memory.config.canonical_store_path)
    before = canonical_path.read_bytes()
    monkeypatch.setattr(
        delete_strategies_module,
        "request_delete_decision",
        lambda model, messages: DeleteDecision(
            selected_ids=selected_ids,
            reasoning="Both memories were selected.",
            ambiguous=False,
        ),
    )

    result = memory.delete("Forget both preferences.")

    assert result == "Delete request was ambiguous; no canonical memory was deleted."
    assert canonical_path.read_bytes() == before
