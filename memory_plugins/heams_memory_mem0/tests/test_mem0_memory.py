"""Focused unit tests for the mem0 memory system.

These tests mock mem0's `Memory` class so they run offline without API keys.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import tiktoken
from heams.memory_systems.api import MemoryEntry
from heams_memory_mem0 import (
    EmbedderConfig,
    LLMConfig,
    Mem0MemoryConfig,
    Mem0MemorySystem,
    VectorStoreConfig,
)
from mem0.embeddings.openai import OpenAIEmbedding


@pytest.fixture()
def mock_mem0_class(monkeypatch):
    """Patch `Memory` so unit tests stay fully local/offline."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance
    mock_class.from_config.return_value = mock_instance
    monkeypatch.setattr("heams_memory_mem0.plugin.Memory", mock_class)
    return mock_class


@pytest.fixture()
def mem0_instance(mock_mem0_class):
    return mock_mem0_class.return_value


@pytest.fixture()
def memory_system(mock_mem0_class, tmp_path):
    return Mem0MemorySystem(
        config=Mem0MemoryConfig(
            user_id="test_user",
            top_k=3,
            temporal_prompt_enabled=True,
        ),
        storage_dir=tmp_path,
    )


def test_close_releases_mem0_resources_once(
    memory_system: Mem0MemorySystem,
    mem0_instance: MagicMock,
) -> None:
    vector_store_client = MagicMock()
    telemetry_vector_store_client = MagicMock()
    entity_store_client = MagicMock()
    llm_client = MagicMock()
    embedding_client = MagicMock()
    reranker_client = MagicMock()
    mem0_instance.vector_store = SimpleNamespace(client=vector_store_client)
    mem0_instance._telemetry_vector_store = SimpleNamespace(
        client=telemetry_vector_store_client
    )
    mem0_instance._entity_store = SimpleNamespace(client=entity_store_client)
    mem0_instance.llm = SimpleNamespace(client=llm_client)
    mem0_instance.embedding_model = SimpleNamespace(client=embedding_client)
    mem0_instance.reranker = SimpleNamespace(client=reranker_client)

    memory_system.close()
    memory_system.close()

    mem0_instance.close.assert_called_once_with()
    vector_store_client.close.assert_called_once_with()
    telemetry_vector_store_client.close.assert_called_once_with()
    entity_store_client.close.assert_called_once_with()
    llm_client.close.assert_called_once_with()
    embedding_client.close.assert_called_once_with()
    reranker_client.close.assert_called_once_with()


def test_close_attempts_all_resources_when_one_close_fails(
    memory_system: Mem0MemorySystem,
    mem0_instance: MagicMock,
) -> None:
    mem0_instance.close.side_effect = RuntimeError("history close failed")
    vector_store_client = MagicMock()
    telemetry_vector_store_client = MagicMock()
    entity_store_client = MagicMock()
    llm_client = MagicMock()
    embedding_client = MagicMock()
    reranker_client = MagicMock()
    mem0_instance.vector_store = SimpleNamespace(client=vector_store_client)
    mem0_instance._telemetry_vector_store = SimpleNamespace(
        client=telemetry_vector_store_client
    )
    mem0_instance._entity_store = SimpleNamespace(client=entity_store_client)
    mem0_instance.llm = SimpleNamespace(client=llm_client)
    mem0_instance.embedding_model = SimpleNamespace(client=embedding_client)
    mem0_instance.reranker = SimpleNamespace(client=reranker_client)

    with pytest.raises(RuntimeError, match="history close failed"):
        memory_system.close()

    vector_store_client.close.assert_called_once_with()
    telemetry_vector_store_client.close.assert_called_once_with()
    entity_store_client.close.assert_called_once_with()
    llm_client.close.assert_called_once_with()
    embedding_client.close.assert_called_once_with()
    reranker_client.close.assert_called_once_with()


def test_llm_to_mem0_dict_includes_expected_fields() -> None:
    cfg = LLMConfig.model_validate(
        {
            "provider": "litellm",
            "model": "test-model",
            "api_key": "sk-test",
            "temperature": 0.3,
            "max_tokens": 900,
            "custom_field": "x",
        }
    )

    result = cfg.to_mem0_dict()

    assert result["provider"] == "litellm"
    assert result["config"]["model"] == "test-model"
    assert result["config"]["api_key"] == "sk-test"
    assert result["config"]["temperature"] == 0.3
    assert result["config"]["max_tokens"] == 900
    assert result["config"]["custom_field"] == "x"


@pytest.mark.parametrize(
    ("embedding_dims", "should_include"),
    [(None, False), (1024, True)],
)
def test_embedder_to_mem0_dict_handles_embedding_dims(
    embedding_dims: int | None,
    should_include: bool,
) -> None:
    cfg = EmbedderConfig(
        provider="ollama",
        model="mxbai-embed-large:latest",
        ollama_base_url="http://localhost:11434",
        embedding_dims=embedding_dims,
    )

    result = cfg.to_mem0_dict()

    assert result["provider"] == "ollama"
    assert result["config"]["model"] == "mxbai-embed-large:latest"
    assert ("embedding_dims" in result["config"]) is should_include


def test_vector_store_to_mem0_dict_omits_none_fields() -> None:
    cfg = VectorStoreConfig(provider="chroma", collection_name="test_col")

    result = cfg.to_mem0_dict()

    assert result["provider"] == "chroma"
    assert result["config"]["collection_name"] == "test_col"
    for field in ("path", "host", "port", "url", "api_key", "embedding_model_dims"):
        assert field not in result["config"]


def test_vector_store_to_mem0_dict_preserves_connection_and_provider_fields() -> None:
    cfg = VectorStoreConfig.model_validate(
        {
            "provider": "qdrant",
            "collection_name": "agent_memories",
            "path": "/tmp/memories",
            "host": "qdrant.internal",
            "port": 6333,
            "url": "https://qdrant.example.com",
            "api_key": "qdrant-key",
            "embedding_model_dims": 768,
            "distance": "Cosine",
        }
    )

    result = cfg.to_mem0_dict()

    assert result == {
        "provider": "qdrant",
        "config": {
            "collection_name": "agent_memories",
            "path": "/tmp/memories",
            "host": "qdrant.internal",
            "port": 6333,
            "url": "https://qdrant.example.com",
            "api_key": "qdrant-key",
            "embedding_model_dims": 768,
            "distance": "Cosine",
        },
    }


def test_mem0_config_to_mem0_config_handles_none_and_partial() -> None:
    assert Mem0MemoryConfig(user_id="alice").to_mem0_config() is None

    partial = Mem0MemoryConfig(
        user_id="alice",
        llm=LLMConfig(provider="openai", model="gpt-4o"),
    ).to_mem0_config()

    assert partial is not None
    assert set(partial.keys()) == {"llm"}


def test_runtime_storage_overrides_static_vector_store_path(tmp_path) -> None:
    resolved = Mem0MemoryConfig(
        user_id="alice",
        vector_store=VectorStoreConfig(
            provider="chroma",
            collection_name="test_col",
            path="/tmp/static-store",
        ),
    ).with_runtime_storage(tmp_path / "runtime-store")
    cfg = resolved.to_mem0_config()

    assert cfg is not None
    assert cfg["vector_store"]["provider"] == "chroma"
    assert cfg["vector_store"]["config"]["collection_name"]
    assert cfg["vector_store"]["config"]["path"] == str(tmp_path / "runtime-store")


def test_runtime_storage_sets_mem0_vector_store_directory(tmp_path) -> None:
    config = Mem0MemoryConfig(
        user_id="alice",
        vector_store=VectorStoreConfig(path="/tmp/static-store"),
    )

    resolved = config.with_runtime_storage(tmp_path / "mem0")

    assert "file_path" not in type(resolved).model_fields
    assert resolved.vector_store is not None
    assert resolved.vector_store.path == str(tmp_path / "mem0")
    mem0_config = resolved.to_mem0_config()
    assert mem0_config is not None
    assert mem0_config["vector_store"]["config"]["path"] == str(tmp_path / "mem0")


def test_init_without_backend_uses_runtime_vector_store(
    mock_mem0_class, tmp_path
) -> None:
    system = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="bob"),
        storage_dir=tmp_path,
    )

    assert system.config.user_id == "bob"
    assert system.config.vector_store is not None
    assert system.config.vector_store.path == str(tmp_path)
    mock_mem0_class.assert_not_called()
    mock_mem0_class.from_config.assert_called_once()


def test_init_with_backend_uses_from_config(mock_mem0_class, tmp_path) -> None:
    _ = Mem0MemorySystem(
        config=Mem0MemoryConfig(
            user_id="bob",
            llm=LLMConfig(provider="openai", model="gpt-4o"),
        ),
        storage_dir=tmp_path,
    )

    mock_mem0_class.from_config.assert_called_once()


def test_openai_embedding_inputs_are_truncated_to_token_cap(
    mock_mem0_class, tmp_path
) -> None:
    sent: list[tuple[str, list[str], Any]] = []
    model = MagicMock(spec=OpenAIEmbedding)
    model.embed = lambda text, action=None: sent.append(("embed", [text], action))
    model.embed_batch = lambda texts, action="add": sent.append(("batch", texts, action))
    mock_mem0_class.from_config.return_value.embedding_model = model
    long_text = "user: remember\nthis detail. " * 4000

    system = Mem0MemorySystem(config=Mem0MemoryConfig(user_id="bob"), storage_dir=tmp_path)
    system._memory.embedding_model.embed(long_text, "search")
    system._memory.embedding_model.embed_batch(["short\ntext", long_text], "add")

    [(_, [truncated], search), (_, [short, batch_truncated], add)] = sent
    encoding = tiktoken.get_encoding("cl100k_base")
    assert (search, add) == ("search", "add")
    assert short == "short text"
    assert truncated == batch_truncated
    assert 8000 < len(encoding.encode_ordinary(truncated)) <= 8191
    assert long_text.replace("\n", " ").startswith(truncated)


def test_init_uses_runtime_local_unique_history_database(
    mock_mem0_class, tmp_path
) -> None:
    first = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="bob"),
        storage_dir=tmp_path,
    )
    second = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="bob"),
        storage_dir=tmp_path,
    )

    first_config = mock_mem0_class.from_config.call_args_list[0].args[0]
    second_config = mock_mem0_class.from_config.call_args_list[1].args[0]
    first_history_path = Path(first_config["history_db_path"])
    second_history_path = Path(second_config["history_db_path"])

    assert first.config.vector_store is not None
    assert first.config.vector_store.path == str(tmp_path)
    assert first_history_path.parent == tmp_path
    assert second_history_path.parent == tmp_path
    assert first_history_path != second_history_path
    assert first_history_path.name.startswith("history_")
    assert second_history_path.name.startswith("history_")


@pytest.mark.parametrize("metadata", [None, {"role": "user"}])
def test_retain_passes_user_id_and_metadata(
    memory_system,
    mem0_instance,
    metadata,
) -> None:
    mem0_instance.add.return_value = {"results": [{"id": "x"}]}

    result = memory_system.retain("Alice lives in Berlin.", metadata=metadata)

    mem0_instance.add.assert_called_once_with(
        "Alice lives in Berlin.",
        user_id="test_user",
        metadata=metadata or {},
    )
    assert result == "Memory retained successfully."


def test_retain_rejects_list_input(
    memory_system,
    mem0_instance,
) -> None:
    mem0_instance.add.return_value = {"results": [{"id": "x"}]}

    result = memory_system.retain(
        ["Alice lives in Berlin.", "Bob lives in Paris."],
        metadata={"source": "priming"},
    )

    mem0_instance.add.assert_not_called()
    assert result == "Failed to retain memory: mem0 retain only accepts string content."


def test_retain_returns_failure_message_on_exception(
    memory_system, mem0_instance
) -> None:
    mem0_instance.add.side_effect = RuntimeError("boom")

    result = memory_system.retain("something")

    assert "Failed to retain memory" in result
    assert "boom" in result


def test_retain_projects_rich_metadata_without_mutating_input(
    memory_system,
    mem0_instance,
) -> None:
    mem0_instance.add.return_value = {"results": [{"id": "x"}]}
    metadata = {
        "source": "transparency",
        "none_value": None,
        "sample_source": {"split": "100K", "dataset": "BEAM"},
        "tags": ["priming", "benchmark"],
        "coordinates": (1, 2),
    }

    memory_system.retain("A test memory.", metadata=metadata)

    mem0_instance.add.assert_called_once_with(
        "A test memory.",
        user_id="test_user",
        metadata={
            "source": "transparency",
            "sample_source_json": '{"dataset":"BEAM","split":"100K"}',
            "tags": ["priming", "benchmark"],
            "coordinates_json": "[1,2]",
        },
    )
    assert metadata == {
        "source": "transparency",
        "none_value": None,
        "sample_source": {"split": "100K", "dataset": "BEAM"},
        "tags": ["priming", "benchmark"],
        "coordinates": (1, 2),
    }


@pytest.mark.parametrize(
    ("metadata", "expected_metadata"),
    [
        ({"labels": []}, {"labels_json": "[]"}),
        ({"labels": ["benchmark", 1]}, {"labels_json": '["benchmark",1]'}),
    ],
)
def test_retain_projects_lists_chroma_does_not_accept(
    memory_system,
    mem0_instance,
    metadata,
    expected_metadata,
) -> None:
    mem0_instance.add.return_value = {"results": [{"id": "x"}]}

    memory_system.retain("A test memory.", metadata=metadata)

    mem0_instance.add.assert_called_once_with(
        "A test memory.",
        user_id="test_user",
        metadata=expected_metadata,
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"labels": {"unsupported"}},
        {"payload": b"unsupported"},
        {"value": float("nan")},
        {"nested": {"value": float("inf")}},
        {1: "non-string key"},
    ],
)
def test_retain_rejects_unsupported_metadata_before_mem0_add(
    memory_system,
    mem0_instance,
    metadata,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        memory_system._retain_impl("A test memory.", metadata)

    mem0_instance.add.assert_not_called()


def test_priming_projects_transparency_metadata_without_mutating_entries(
    memory_system,
    mem0_instance,
) -> None:
    metadata = {
        "source_benchmark": "transparency",
        "scenario_id": "TRANS-BEAM-100K-example",
        "sample_source": {
            "dataset": "BEAM",
            "split": "100K",
            "conversation_id": "example",
        },
        "ingest_group_id": "transparency:example",
        "ingest_order": 0,
        "ingest_role": "user",
    }
    entry = MemoryEntry(content="A transparency memory.", metadata=metadata)

    memory_system.prime_memory_entries([entry])

    mem0_instance.add.assert_called_once_with(
        [{"role": "user", "content": "A transparency memory."}],
        user_id="test_user",
        metadata={
            "source_benchmark": "transparency",
            "scenario_id": "TRANS-BEAM-100K-example",
            "sample_source_json": (
                '{"conversation_id":"example","dataset":"BEAM","split":"100K"}'
            ),
        },
    )
    assert entry.metadata == metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"sample_source": {"dataset": "BEAM"}, "sample_source_json": "present"},
        {"sample_source": {"dataset": {"unsupported"}}},
    ],
)
def test_priming_metadata_projection_errors_fail_before_mem0_add(
    memory_system,
    mem0_instance,
    metadata,
) -> None:
    entry = MemoryEntry(
        content="A test memory.",
        metadata={
            **metadata,
            "ingest_group_id": "group-1",
            "ingest_order": 0,
            "ingest_role": "user",
        },
    )

    with pytest.raises((TypeError, ValueError), match="Mem0 metadata"):
        memory_system.prime_memory_entries([entry])

    mem0_instance.add.assert_not_called()


def test_prime_memory_entries_groups_orders_and_forwards_shared_metadata(
    memory_system,
    mem0_instance,
) -> None:
    entries = [
        MemoryEntry(
            content="I moved to Montreal last spring.",
            metadata={
                "source": "locomo",
                "source_benchmark": "locomo",
                "session_id": "session_1",
                "timestamp": "2024-03-01 10:00:00",
                "speaker": "Alice",
                "dia_id": "D1:1",
                "ingest_group_id": "group-1",
                "ingest_order": 1,
                "ingest_role": "user",
                "ingest_speaker_name": "Alice",
                "ingest_timestamp_epoch": 1709287200,
            },
        ),
        MemoryEntry(
            content="Nice, how was the winter?",
            metadata={
                "source": "locomo",
                "source_benchmark": "locomo",
                "session_id": "session_1",
                "timestamp": "2024-03-01 10:00:00",
                "speaker": "Bob",
                "dia_id": "D1:2",
                "ingest_group_id": "group-1",
                "ingest_order": 0,
                "ingest_role": "assistant",
                "ingest_speaker_name": "Bob",
                "ingest_timestamp_epoch": 1709287200,
            },
        ),
    ]

    memory_system.prime_memory_entries(entries)

    mem0_instance.add.assert_called_once_with(
        [
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
            {"role": "assistant", "content": "Bob: Nice, how was the winter?"},
            {"role": "user", "content": "Alice: I moved to Montreal last spring."},
        ],
        user_id="test_user",
        metadata={
            "source": "locomo",
            "source_benchmark": "locomo",
            "session_id": "session_1",
            "timestamp": "2024-03-01 10:00:00",
            "ingest_timestamp_epoch": 1709287200,
        },
        prompt=(
            "Today is 2024-03-01 for this conversation. Use this as the "
            "authoritative reference date when resolving relative temporal "
            "expressions, rather than the current system date. Express resolved "
            "dates explicitly in extracted memories without rewriting the input "
            "messages."
        ),
    )


def test_timestamped_priming_forwards_relative_dates_and_temporal_prompt(
    memory_system,
    mem0_instance,
) -> None:
    content = (
        "I went to a LGBTQ support group yesterday. "
        "I started volunteering there last year and plan to return next month. "
        "I also attended a workshop last week."
    )

    memory_system.prime_memory_entries(
        [
            MemoryEntry(
                content=content,
                metadata={
                    "source": "locomo",
                    "source_benchmark": "locomo",
                    "session_id": "session_1",
                    "timestamp": "08 May 2023",
                    "ingest_timestamp_epoch": 1683504000,
                    "ingest_group_id": "locomo:session_1",
                    "ingest_order": 0,
                    "ingest_role": "user",
                    "ingest_speaker_name": "Caroline",
                },
            )
        ]
    )

    mem0_instance.add.assert_called_once_with(
        [
            {
                "role": "system",
                "content": (
                    "Temporal context for extraction only; do not treat this "
                    "message as source conversation content. Today is "
                    "2023-05-08 for this conversation. Use this as the "
                    "authoritative reference date when resolving relative "
                    "temporal expressions, rather than the current system date. "
                    "Express resolved dates explicitly in extracted memories "
                    "without rewriting the input messages."
                ),
            },
            {
                "role": "user",
                "content": f"Caroline: {content}",
            }
        ],
        user_id="test_user",
        metadata={
            "source": "locomo",
            "source_benchmark": "locomo",
            "session_id": "session_1",
            "timestamp": "08 May 2023",
            "ingest_timestamp_epoch": 1683504000,
        },
        prompt=(
            "Today is 2023-05-08 for this conversation. Use this as the "
            "authoritative reference date when resolving relative temporal "
            "expressions, rather than the current system date. Express resolved "
            "dates explicitly in extracted memories without rewriting the input "
            "messages."
        ),
    )


def test_temporal_prompt_can_be_disabled_while_preserving_timestamp_metadata(
    memory_system,
    mem0_instance,
) -> None:
    disabled_memory_system = Mem0MemorySystem(
        config=Mem0MemoryConfig(user_id="test_user"),
        storage_dir=memory_system.storage_dir,
    )

    disabled_memory_system.prime_memory_entries(
        [
            MemoryEntry(
                content="Alice moved to Montreal yesterday.",
                metadata={
                    "source": "locomo",
                    "ingest_timestamp_epoch": 1683504000,
                    "ingest_group_id": "group-1",
                    "ingest_order": 0,
                    "ingest_role": "user",
                },
            )
        ]
    )

    mem0_instance.add.assert_called_once_with(
        [{"role": "user", "content": "Alice moved to Montreal yesterday."}],
        user_id="test_user",
        metadata={"source": "locomo", "ingest_timestamp_epoch": 1683504000},
    )


@pytest.mark.parametrize(
    "invalid_timestamp",
    ["not-a-number", True, 10**100],
)
def test_invalid_timestamp_metadata_does_not_abort_priming(
    memory_system,
    mem0_instance,
    invalid_timestamp,
) -> None:
    mem0_instance.add.return_value = {"results": [{"id": "memory-1"}]}

    memory_system.prime_memory_entries(
        [
            MemoryEntry(
                content="Alice mentioned an event.",
                metadata={
                    "source": "test",
                    "source_benchmark": "test",
                    "session_id": "session-1",
                    "timestamp": "not-an-epoch",
                    "ingest_timestamp_epoch": invalid_timestamp,
                    "ingest_group_id": "group-1",
                    "ingest_order": 0,
                    "ingest_role": "user",
                },
            )
        ]
    )

    mem0_instance.add.assert_called_once_with(
        [{"role": "user", "content": "Alice mentioned an event."}],
        user_id="test_user",
        metadata={
            "source": "test",
            "source_benchmark": "test",
            "session_id": "session-1",
            "timestamp": "not-an-epoch",
        },
    )


def test_prime_memory_entries_supports_long_mem_eval_pair_ingestion(
    memory_system,
    mem0_instance,
) -> None:
    entries = [
        MemoryEntry(
            content="I moved to Montreal last spring.",
            metadata={
                "source": "long_mem_eval",
                "source_benchmark": "long_mem_eval",
                "session_id": "S1",
                "session_index": 0,
                "timestamp": "2024/03/01 21:05",
                "answer_session": True,
                "ingest_group_id": "long_mem_eval:LME-FIXTURE-1:S1:0",
                "ingest_order": 0,
                "ingest_role": "user",
                "ingest_scope_id": "LME-FIXTURE-1",
                "ingest_timestamp_epoch": 1709327100,
                "has_answer": True,
            },
        ),
        MemoryEntry(
            content="That sounds exciting.",
            metadata={
                "source": "long_mem_eval",
                "source_benchmark": "long_mem_eval",
                "session_id": "S1",
                "session_index": 0,
                "timestamp": "2024/03/01 21:05",
                "answer_session": True,
                "ingest_group_id": "long_mem_eval:LME-FIXTURE-1:S1:0",
                "ingest_order": 1,
                "ingest_role": "assistant",
                "ingest_scope_id": "LME-FIXTURE-1",
                "ingest_timestamp_epoch": 1709327100,
            },
        ),
    ]

    memory_system.prime_memory_entries(entries)

    mem0_instance.add.assert_called_once_with(
        [
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
            {"role": "user", "content": "I moved to Montreal last spring."},
            {"role": "assistant", "content": "That sounds exciting."},
        ],
        user_id="test_user",
        metadata={
            "source": "long_mem_eval",
            "source_benchmark": "long_mem_eval",
            "session_id": "S1",
            "session_index": 0,
            "timestamp": "2024/03/01 21:05",
            "answer_session": True,
            "ingest_timestamp_epoch": 1709327100,
        },
        prompt=(
            "Today is 2024-03-01 for this conversation. Use this as the "
            "authoritative reference date when resolving relative temporal "
            "expressions, rather than the current system date. Express resolved "
            "dates explicitly in extracted memories without rewriting the input "
            "messages."
        ),
    )


@pytest.mark.parametrize(("top_k_arg", "expected_limit"), [(None, 3), (7, 7)])
def test_recall_calls_search_with_filters_and_limit(
    memory_system,
    mem0_instance,
    top_k_arg: int | None,
    expected_limit: int,
) -> None:
    mem0_instance.search.return_value = {"results": []}

    _ = memory_system.recall("where does Alice live", top_k=top_k_arg)

    mem0_instance.search.assert_called_once_with(
        "where does Alice live",
        filters={"user_id": "test_user"},
        top_k=expected_limit,
    )


def test_recall_formats_plain_and_metadata_results(
    memory_system, mem0_instance
) -> None:
    mem0_instance.search.return_value = {
        "results": [
            {"memory": "plain fact"},
            {"memory": "fact with metadata", "metadata": {"tag": "x"}, "score": 0.8},
        ]
    }

    results = memory_system.recall("query")

    assert results[0] == "plain fact"
    parsed = json.loads(results[1])
    assert parsed["content"] == "fact with metadata"
    assert parsed["metadata"] == {"tag": "x"}


def test_recall_rehydrates_projected_metadata(memory_system, mem0_instance) -> None:
    metadata = {
        "source_benchmark": "transparency",
        "sample_source_json": (
            '{"conversation_id":"example","dataset":"BEAM","split":"100K"}'
        ),
        "labels": ["benchmark", "priming"],
        "malformed_json": "not-json",
        "scalar_json": "2",
    }
    mem0_instance.search.return_value = {
        "results": [{"memory": "A transparency memory.", "metadata": metadata}]
    }

    results = memory_system.recall("Where did this come from?")

    parsed = json.loads(results[0])
    assert parsed["metadata"] == {
        "source_benchmark": "transparency",
        "sample_source": {
            "conversation_id": "example",
            "dataset": "BEAM",
            "split": "100K",
        },
        "labels": ["benchmark", "priming"],
        "malformed_json": "not-json",
        "scalar_json": "2",
    }
    assert metadata["sample_source_json"] == (
        '{"conversation_id":"example","dataset":"BEAM","split":"100K"}'
    )


def test_recall_handles_missing_results_key(memory_system, mem0_instance) -> None:
    mem0_instance.search.return_value = {}

    results = memory_system.recall("query")

    assert results == []


def test_recall_returns_failure_message_on_exception(
    memory_system, mem0_instance
) -> None:
    mem0_instance.search.side_effect = RuntimeError("network error")

    results = memory_system.recall("query")

    assert len(results) == 1
    assert "Failed to recall memories" in results[0]
