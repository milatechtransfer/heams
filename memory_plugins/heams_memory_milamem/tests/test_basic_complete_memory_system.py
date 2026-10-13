import json
from pathlib import Path

import pytest

from heams_memory_milamem.basic_complete_memory_system import (
    BasicCompleteMemorySystem,
    BasicCompleteMemorySystemConfig,
)
from heams.memory_systems.api import MemoryEntry

_FACTS = [
    MemoryEntry(content="Full name is Maria Chen", metadata={"ts": 30}),
    MemoryEntry(content="Email is maria.chen@university.edu", metadata={"ts": 29}),
    MemoryEntry(content="Phone number is 555-0142", metadata={"ts": 15}),
    MemoryEntry(content="Employer is State University", metadata={"ts": 10}),
    MemoryEntry(content="Prefers window seating", metadata={"ts": 5}),
    MemoryEntry(content="BMI is 23", metadata={"ts": 4}),
]


def _memory(path: Path, **config_kwargs) -> BasicCompleteMemorySystem:
    return BasicCompleteMemorySystem(
        config=BasicCompleteMemorySystemConfig(
            file_path=str(path),
            **config_kwargs,
        ),
        storage_dir=path.parent,
    )


def _unpack_recalled(raw: str) -> tuple[str, dict[str, object]]:
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw, {}

    if isinstance(parsed, dict) and isinstance(parsed.get("content"), str):
        metadata = parsed.get("metadata")
        if isinstance(metadata, dict):
            return parsed["content"], metadata
    return raw, {}


def test_retain_persists_active_raw_record_shape(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system")

    result = memory.retain("Saved fact", {"ts": 9})

    assert result == "Successfully saved to memory."
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert len(raw) == 1
    assert raw[0]["content"] == "Saved fact"
    assert raw[0]["metadata"] == {"ts": 9}
    assert raw[0]["sequence"] == 1
    assert raw[0]["status"] == "active"
    assert raw[0]["deleted_at"] is None
    assert raw[0]["superseded_by"] is None
    assert isinstance(raw[0]["id"], str)
    assert raw[0]["id"]


def test_recall_empty_query_returns_chronological_tail(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system", top_k=2)
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    tail = memory.recall(query="")
    unpacked_tail = [_unpack_recalled(item) for item in tail]
    assert unpacked_tail == [
        ("Prefers window seating", {"ts": 5}),
        ("BMI is 23", {"ts": 4}),
    ]


def test_recall_ignores_inactive_raw_records(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "r1",
                    "content": "Active fact",
                    "metadata": {"ts": 1},
                    "sequence": 1,
                    "status": "active",
                    "deleted_at": None,
                    "superseded_by": None,
                },
                {
                    "id": "r2",
                    "content": "Deleted fact",
                    "metadata": {"ts": 2},
                    "sequence": 2,
                    "status": "deleted",
                    "deleted_at": 123.0,
                    "superseded_by": None,
                },
                {
                    "id": "r3",
                    "content": "Superseded fact",
                    "metadata": {"ts": 3},
                    "sequence": 3,
                    "status": "superseded",
                    "deleted_at": None,
                    "superseded_by": "r4",
                },
            ]
        ),
        encoding="utf-8",
    )
    memory = _memory(path, name="basic-complete-memory-system", top_k=5)

    assert memory.recall(query="") == [
        json.dumps({"content": "Active fact", "metadata": {"ts": 1}}, sort_keys=True)
    ]


def test_delete_exact_query_deletes_one_record(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system", top_k=10)
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    result = memory.delete("Phone number is 555-0142")

    assert result == "Deleted 1 memory record."
    raw = json.loads(path.read_text(encoding="utf-8"))
    deleted = [
        record for record in raw if record["content"] == "Phone number is 555-0142"
    ]
    assert len(deleted) == 1
    assert deleted[0]["status"] == "deleted"
    assert deleted[0]["deleted_at"] is not None


def test_delete_ranked_query_deletes_top_record(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system", top_k=10)
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    result = memory.delete("phone")

    assert result == "Deleted 1 memory record."
    raw = json.loads(path.read_text(encoding="utf-8"))
    deleted = [
        record for record in raw if record["content"] == "Phone number is 555-0142"
    ]
    assert len(deleted) == 1
    assert deleted[0]["status"] == "deleted"


def test_delete_unmatched_query_deletes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system", top_k=10)
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    result = memory.delete("astronomy")

    assert result == "Delete request did not match any active memory."
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert all(record["status"] == "active" for record in raw)


def test_deleted_records_no_longer_appear_in_recall(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    memory = _memory(path, name="basic-complete-memory-system", top_k=10)
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    memory.delete("Phone number is 555-0142")

    recalled = memory.recall(query="phone", top_k=3)
    unpacked = [_unpack_recalled(item)[0] for item in recalled]
    assert "Phone number is 555-0142" not in unpacked


@pytest.mark.parametrize(
    "query, expected",
    [
        ("name", "Full name is Maria Chen"),
        ("email", "Email is maria.chen@university.edu"),
        ("seating", "Prefers window seating"),
    ],
)
def test_recall_answers_field_specific_queries(
    tmp_path: Path,
    query: str,
    expected: str,
) -> None:
    memory = _memory(
        tmp_path / "memory.json",
        name="basic-complete-memory-system",
        top_k=1,
    )
    for fact in _FACTS:
        memory.retain(fact.content, fact.metadata)

    [top] = memory.recall(query=query)
    top_content, top_metadata = _unpack_recalled(top)
    assert top_content == expected
    assert "ts" in top_metadata
