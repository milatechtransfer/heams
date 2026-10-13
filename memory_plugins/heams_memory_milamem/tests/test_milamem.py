import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from heams_memory_milamem import consolidation as consolidation_module
from heams_memory_milamem import recall_strategies as recall_strategies_module
from heams_memory_milamem.milamem import (
    MilaMemConfig,
    MilaMemMemorySystem,
)
from heams_memory_milamem.consolidation import ConsolidationVerdict
from heams_memory_milamem.dedup_strategies import (
    DedupStrategyConfig,
    DedupStrategyKind,
)
from heams.memory_systems.api import MemoryEntry


def _fake_embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for `any_llm.embedding`, so tests never hit a
    real API. Needed here because canonical's dedup_strategy defaults to
    semantic (per Chirag's note), so every retain() below embeds content."""

    vectors: list[list[float]] = []
    for text in texts:
        digest = hashlib.md5(f"{model}:{text}".encode("utf-8")).hexdigest()
        vectors.append([int(digest[i : i + 2], 16) / 255.0 for i in range(0, 16, 2)])
    return vectors


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recall_strategies_module, "embed_texts", _fake_embed_texts)


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the LLM judge to always return 'unrelated' so nothing here gets superseded."""

    def _always_unrelated(model: str, records: list) -> ConsolidationVerdict:
        del model, records
        return ConsolidationVerdict(relationship="unrelated", canonical_id=None)

    monkeypatch.setattr(consolidation_module, "judge_cluster", _always_unrelated)


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _memory(
    raw_path: Path,
    canonical_path: Path,
    **config_kwargs: Any,
) -> MilaMemMemorySystem:
    return MilaMemMemorySystem(
        config=MilaMemConfig(
            raw_store_path=str(raw_path),
            canonical_store_path=str(canonical_path),
            **config_kwargs,
        ),
        storage_dir=raw_path.parent,
    )


def test_enabled_canonical_dedup_requires_semantic_kind() -> None:
    with pytest.raises(ValueError, match="kind must be 'semantic'"):
        MilaMemConfig(
            dedup_strategy=DedupStrategyConfig(
                enabled=True,
                kind=DedupStrategyKind.BM25,
            )
        )


def test_milamem_retain_writes_raw_and_canonical_records(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
    )

    result = memory.retain("Saved fact", {"ts": 9})

    assert result == "Successfully saved to memory."
    raw_state = _load_state(raw_path)
    canonical_state = _load_state(canonical_path)
    raw_records = raw_state["records"]
    canonical_records = canonical_state["records"]

    assert raw_state["schema_version"] == 1
    assert canonical_state["schema_version"] == 1
    assert len(raw_records) == 1
    assert len(canonical_records) == 1

    raw_record = raw_records[0]
    canonical_record = canonical_records[0]

    assert raw_record["content"] == "Saved fact"
    assert raw_record["metadata"] == {"ts": 9}
    assert raw_record["sequence"] == 1
    assert raw_record["status"] == "active"
    assert canonical_record["content"] == "Saved fact"
    assert canonical_record["metadata"] == {"ts": 9}
    assert canonical_record["sequence"] == 1
    assert canonical_record["memory_type"] == "observation"


def test_temporal_metadata_enabled_preserves_timestamped_metadata_in_both_stores(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
        temporal_metadata={"enabled": True},
    )
    metadata = {"timestamp": "2024-03-01T12:00:00+02:00", "source": "test"}

    memory.retain("Timestamped fact", metadata)

    raw_metadata = _load_state(raw_path)["records"][0]["metadata"]
    canonical_metadata = _load_state(canonical_path)["records"][0]["metadata"]

    assert raw_metadata["timestamp"] == metadata["timestamp"]
    assert raw_metadata["source"] == metadata["source"]
    assert raw_metadata["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert raw_metadata["temporal_source"] == "metadata.timestamp"
    assert {
        key: canonical_metadata[key]
        for key in ("temporal_received_at", "temporal_source")
    } == {key: raw_metadata[key] for key in ("temporal_received_at", "temporal_source")}


def test_temporal_metadata_enabled_uses_clock_without_timestamp_in_both_stores(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
        temporal_metadata={"enabled": True},
    )

    memory.retain("Untimestamped fact", {"source": "test"})

    raw_metadata = _load_state(raw_path)["records"][0]["metadata"]
    canonical_metadata = _load_state(canonical_path)["records"][0]["metadata"]

    assert raw_metadata["source"] == "test"
    assert raw_metadata["temporal_received_at"].endswith("Z")
    assert raw_metadata["temporal_source"] == "ingestion_clock"
    assert {
        key: canonical_metadata[key]
        for key in ("temporal_received_at", "temporal_source")
    } == {key: raw_metadata[key] for key in ("temporal_received_at", "temporal_source")}


def test_primed_entries_receive_temporal_metadata_through_public_retain_path(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
        temporal_metadata={"enabled": True},
    )

    memory.prime_memory_entries(
        [
            MemoryEntry(
                content="Primed timestamped fact",
                metadata={"timestamp": "2024-03-01T12:00:00+02:00"},
            ),
            MemoryEntry(
                content="Primed relative fact",
                metadata={"recorded_at": "T-10d"},
            ),
        ]
    )

    raw_by_content = {
        record["content"]: record["metadata"]
        for record in _load_state(raw_path)["records"]
    }
    canonical_by_content = {
        record["content"]: record["metadata"]
        for record in _load_state(canonical_path)["records"]
    }
    timestamped = raw_by_content["Primed timestamped fact"]
    relative = raw_by_content["Primed relative fact"]

    assert timestamped["timestamp"] == "2024-03-01T12:00:00+02:00"
    assert timestamped["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert timestamped["temporal_source"] == "metadata.timestamp"
    assert {
        key: canonical_by_content["Primed timestamped fact"][key]
        for key in ("temporal_received_at", "temporal_source")
    } == {key: timestamped[key] for key in ("temporal_received_at", "temporal_source")}
    assert relative["recorded_at"] == "T-10d"
    assert relative["temporal_source"] == "ingestion_clock"
    assert relative["temporal_received_at"].endswith("Z")
    assert (
        datetime.fromisoformat(
            relative["temporal_received_at"].replace("Z", "+00:00")
        ).tzinfo
        == timezone.utc
    )
    assert {
        key: canonical_by_content["Primed relative fact"][key]
        for key in ("temporal_received_at", "temporal_source")
    } == {key: relative[key] for key in ("temporal_received_at", "temporal_source")}


def test_recall_renders_temporal_received_at_when_enabled(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
        temporal_metadata={"enabled": True},
    )

    memory.retain("Recallable timestamped fact", {"timestamp": "2024-03-01 10:00:00"})

    rendered = json.loads(memory.recall("", top_k=1)[0])

    assert rendered["content"] == "Recallable timestamped fact"
    assert rendered["metadata"]["timestamp"] == "2024-03-01 10:00:00"
    assert rendered["metadata"]["temporal_received_at"] == "2024-03-01T10:00:00Z"
    assert rendered["metadata"]["temporal_source"] == "metadata.timestamp"


def test_milamem_retain_appends_to_both_ledgers(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
    )

    memory.retain("Fact one", {"turn": 1})
    memory.retain("Fact two", {"turn": 2})

    raw_state = _load_state(raw_path)
    canonical_state = _load_state(canonical_path)
    raw_records = raw_state["records"]
    canonical_records = canonical_state["records"]

    assert len(raw_records) == 2
    assert len(canonical_records) == 2
    assert [record["sequence"] for record in raw_records] == [1, 2]
    assert [record["sequence"] for record in canonical_records] == [1, 2]
    assert [record["content"] for record in raw_records] == ["Fact one", "Fact two"]
    assert [record["content"] for record in canonical_records] == [
        "Fact one",
        "Fact two",
    ]


def test_milamem_persists_envelope_not_bare_list(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
    )

    memory.retain("Envelope fact", {"kind": "shape"})

    raw_state = _load_state(raw_path)
    canonical_state = _load_state(canonical_path)

    assert isinstance(raw_state, dict)
    assert isinstance(canonical_state, dict)
    assert set(raw_state) == {"schema_version", "records"}
    assert set(canonical_state) == {
        "schema_version",
        "records",
        "pending_consolidation_ids",
    }
    assert isinstance(raw_state["records"], list)
    assert isinstance(canonical_state["records"], list)


def test_milamem_canonical_records_preserve_provenance(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
    )

    memory.retain("Provenance fact", {"source": "user"})

    raw_state = _load_state(raw_path)
    canonical_state = _load_state(canonical_path)
    raw_record = raw_state["records"][0]
    canonical_record = canonical_state["records"][0]

    assert canonical_record["raw_source_id"] == raw_record["id"]
    assert canonical_record["provenance"] == {"raw_record_id": raw_record["id"]}
    assert canonical_record["id"] == f"canonical:{raw_record['id']}"


def test_runtime_storage_sets_store_paths_and_rederives_embedding_cache(
    tmp_path: Path,
) -> None:
    config = MilaMemConfig(
        raw_store_path="static-raw.json",
        canonical_store_path="static-canonical.json",
        embedding_cache_path="static-embeddings.json",
    )

    resolved = config.with_runtime_storage(tmp_path)

    assert resolved.raw_store_path == str(tmp_path / "raw.json")
    assert resolved.canonical_store_path == str(tmp_path / "canonical.json")
    assert resolved.embedding_cache_path == str(tmp_path / "canonical.embeddings.json")
    assert config.raw_store_path == "static-raw.json"
    assert config.canonical_store_path == "static-canonical.json"
    assert config.embedding_cache_path == "static-embeddings.json"
    memory = MilaMemMemorySystem(config=config, storage_dir=tmp_path)
    assert memory.config.embedding_cache_path == str(
        tmp_path / "canonical.embeddings.json"
    )


def test_runtime_storage_overrides_configured_store_paths(tmp_path: Path) -> None:
    configured_raw_path = tmp_path / "custom-raw.json"
    configured_canonical_path = tmp_path / "custom-canonical.json"
    memory = _memory(configured_raw_path, configured_canonical_path)

    memory.retain("Configured paths fact")

    assert memory.config.raw_store_path == str(tmp_path / "raw.json")
    assert memory.config.canonical_store_path == str(tmp_path / "canonical.json")
    assert not configured_raw_path.exists()
    assert not configured_canonical_path.exists()
    assert (tmp_path / "raw.json").is_file()
    assert (tmp_path / "canonical.json").is_file()


def test_milamem_preserves_legacy_raw_history(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.json"
    canonical_path = tmp_path / "canonical.json"
    raw_path.write_text(
        json.dumps(
            [
                {
                    "id": "raw-1",
                    "content": "Existing fact",
                    "metadata": {"ts": 1},
                    "sequence": 1,
                    "status": "active",
                    "deleted_at": None,
                    "superseded_by": None,
                    "consolidation_reason": None,
                }
            ]
        ),
        encoding="utf-8",
    )

    memory = _memory(
        raw_path,
        canonical_path,
        name="basic-canonical-memory-system",
    )

    result = memory.retain("New fact", {"ts": 2})

    assert result == "Successfully saved to memory."
    raw_state = _load_state(raw_path)
    canonical_state = _load_state(canonical_path)

    assert [record["content"] for record in raw_state["records"]] == [
        "Existing fact",
        "New fact",
    ]
    assert [record["sequence"] for record in raw_state["records"]] == [1, 2]
    assert [record["content"] for record in canonical_state["records"]] == ["New fact"]
