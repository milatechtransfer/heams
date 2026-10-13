import json

from heams.memory_systems import InContextConfig, create_memory
from heams.memory_systems.shared_types import MemoryEntry


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


def test_prime_memory_entries_preserves_insertion_order(tmp_path) -> None:
    memory = create_memory(
        InContextConfig(top_k=1),
        storage_dir=tmp_path,
    )

    memory.prime_memory_entries(
        [
            MemoryEntry(content="first", metadata={"index": 1}),
            MemoryEntry(content="second", metadata={"index": 2}),
        ]
    )

    results = memory.recall("ignored", top_k=1)

    assert [_decode_recall_item(item) for item in results] == [
        ("first", {"index": 1}),
        ("second", {"index": 2}),
    ]


def test_recall_returns_all_entries_in_insertion_order_ignoring_top_k(tmp_path) -> None:
    memory = create_memory(
        InContextConfig(top_k=1),
        storage_dir=tmp_path,
    )

    memory.retain("a", metadata={"position": 1})
    memory.retain("b", metadata={"position": 2})
    memory.retain("c", metadata={"position": 3})

    results = memory.recall("anything", top_k=1)

    assert [_decode_recall_item(item) for item in results] == [
        ("a", {"position": 1}),
        ("b", {"position": 2}),
        ("c", {"position": 3}),
    ]


def test_recall_uses_fixed_sliding_window_and_keeps_newest_entries(
    tmp_path,
) -> None:
    memory = create_memory(
        InContextConfig(top_k=99, max_approx_recall_tokens=70),
        storage_dir=tmp_path,
    )

    memory.retain("oldest " + "a" * 40, metadata={"position": 1})
    memory.retain("middle " + "b" * 40, metadata={"position": 2})
    memory.retain("newest " + "c" * 40, metadata={"position": 3})

    results = memory.recall("ignored", top_k=1)

    assert [_decode_recall_item(item) for item in results] == [
        ("middle " + "b" * 40, {"position": 2}),
        ("newest " + "c" * 40, {"position": 3}),
    ]
