from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self

import tqdm
from mem0 import Memory
from pydantic import BaseModel, ConfigDict, Field

from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryEntry,
    MemoryEventTarget,
    MemoryOperationResult,
    MemoryPlugin,
    MemoryRuntimeContext,
)


def _mem0_write_count(raw_result: Any) -> int:
    if isinstance(raw_result, dict):
        results = raw_result.get("results")
        if isinstance(results, list):
            return len(results)
        return 1
    if isinstance(raw_result, list):
        return len(raw_result)
    return 1 if raw_result is not None else 0


def _to_mem0_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool | list[str | int | float | bool]]:
    """Project metadata into values accepted by Mem0's Chroma store."""

    def validate_json_value(value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Mem0 metadata floats must be finite.")
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                validate_json_value(item)
            return
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not isinstance(nested_key, str):
                    raise TypeError("Mem0 metadata keys must be strings.")
                validate_json_value(nested_value)
            return
        raise TypeError(
            "Mem0 metadata values must be scalar or JSON-compatible containers."
        )

    projected: dict[str, str | int | float | bool | list[str | int | float | bool]] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("Mem0 metadata keys must be strings.")
        if value is None:
            continue
        if isinstance(value, list):
            list_item_type = bool if value and isinstance(value[0], bool) else (
                type(value[0]) if value else None
            )
            if list_item_type in (str, int, float, bool) and all(
                (bool if isinstance(item, bool) else type(item)) is list_item_type
                for item in value
            ):
                validate_json_value(value)
                projected_key = key
                projected_value = value
            else:
                projected_key = f"{key}_json"
                if projected_key in metadata:
                    raise ValueError(
                        f"Mem0 metadata projection for {key!r} conflicts with "
                        f"existing key {projected_key!r}."
                    )
                validate_json_value(value)
                projected_value = json.dumps(
                    value,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        elif isinstance(value, (dict, tuple)):
            projected_key = f"{key}_json"
            if projected_key in metadata:
                raise ValueError(
                    f"Mem0 metadata projection for {key!r} conflicts with "
                    f"existing key {projected_key!r}."
                )
            validate_json_value(value)
            projected_value = json.dumps(
                value,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        elif isinstance(value, (str, bool, int)):
            projected_key = key
            projected_value = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("Mem0 metadata floats must be finite.")
            projected_key = key
            projected_value = value
        else:
            raise TypeError(
                "Mem0 metadata values must be scalar or JSON-compatible containers."
            )
        if projected_key in projected:
            raise ValueError(
                f"Mem0 metadata projection produces duplicate key {projected_key!r}."
            )
        projected[projected_key] = projected_value
    return projected


def _rehydrate_mem0_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Restore structured metadata projected by ``_to_mem0_metadata``."""
    rehydrated: dict[str, Any] = {}
    for key, value in metadata.items():
        original_key = key.removesuffix("_json") if isinstance(key, str) else key
        if (
            not isinstance(original_key, str)
            or original_key == key
            or not original_key
            or original_key in metadata
            or not isinstance(value, str)
        ):
            rehydrated[key] = value
            continue
        try:
            decoded_value = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            rehydrated[key] = value
            continue
        if isinstance(decoded_value, (dict, list)):
            rehydrated[original_key] = decoded_value
        else:
            rehydrated[key] = value
    return rehydrated


def _coerce_timestamp_epoch(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        timestamp_epoch = int(value)
        datetime.fromtimestamp(timestamp_epoch, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None
    return timestamp_epoch


def _build_temporal_prompt(timestamp_epoch: int) -> str:
    observation_date = datetime.fromtimestamp(
        timestamp_epoch, tz=timezone.utc
    ).date()

    return (
        f"Today is {observation_date.isoformat()} for this conversation. "
        "Use this as the authoritative reference date when resolving relative "
        "temporal expressions, rather than the current system date. Express "
        "resolved dates explicitly in extracted memories without rewriting the "
        "input messages."
    )


class LLMConfig(BaseModel):
    """Configure the LLM backend Mem0 uses for memory management.

    Additional provider-specific options are accepted and forwarded to Mem0.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "provider": "ollama",
                    "model": "llama3.1",
                    "ollama_base_url": "http://localhost:11434",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }
            ]
        },
    )

    provider: str = Field(
        ...,
        description=(
            "Mem0 LLM provider key, such as 'ollama', 'openai', 'anthropic', "
            "'groq', 'together', or 'litellm'."
        ),
    )
    model: str = Field(
        ...,
        description="Model name recognized by the selected provider.",
    )
    api_key: str | None = Field(
        default=None,
        description="Provider API key. Prefer an environment variable when supported.",
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature for Mem0's internal memory operations.",
    )
    max_tokens: int = Field(
        default=2000,
        description="Maximum tokens allowed for Mem0's internal LLM calls.",
    )
    ollama_base_url: str | None = Field(
        default=None,
        description="Base URL for an Ollama provider, for example http://localhost:11434.",
    )

    def to_mem0_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            cfg["api_key"] = self.api_key
        if self.ollama_base_url:
            cfg["ollama_base_url"] = self.ollama_base_url
        cfg.update(self.model_extra or {})
        return {"provider": self.provider, "config": cfg}


class EmbedderConfig(BaseModel):
    """Configure the embedding backend Mem0 uses for memory retrieval.

    Additional provider-specific options are accepted and forwarded to Mem0.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "provider": "ollama",
                    "model": "nomic-embed-text",
                    "ollama_base_url": "http://localhost:11434",
                    "embedding_dims": 768,
                }
            ]
        },
    )

    provider: str = Field(
        ...,
        description=(
            "Mem0 embedding provider key, such as 'ollama', 'openai', "
            "'huggingface', or 'azure_openai'."
        ),
    )
    model: str = Field(
        ...,
        description="Embedding model name recognized by the selected provider.",
    )
    api_key: str | None = Field(
        default=None,
        description="Provider API key. Prefer an environment variable when supported.",
    )
    ollama_base_url: str | None = Field(
        default=None,
        description="Base URL for an Ollama provider, for example http://localhost:11434.",
    )
    embedding_dims: int | None = Field(
        default=None,
        description=(
            "Embedding vector dimensionality. Set this when the provider or "
            "vector store requires an explicit dimension."
        ),
    )

    def to_mem0_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"model": self.model}
        if self.api_key:
            cfg["api_key"] = self.api_key
        if self.ollama_base_url:
            cfg["ollama_base_url"] = self.ollama_base_url
        if self.embedding_dims is not None:
            cfg["embedding_dims"] = self.embedding_dims
        cfg.update(self.model_extra or {})
        return {"provider": self.provider, "config": cfg}


class VectorStoreConfig(BaseModel):
    """Configure the vector store Mem0 uses to persist embeddings.

    The generic connection fields cover common local, hosted, and networked
    stores. Additional provider-specific options are accepted and forwarded to
    Mem0.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "provider": "chroma",
                    "collection_name": "agent_memories",
                    "path": "./chroma_db",
                },
                {
                    "provider": "qdrant",
                    "collection_name": "agent_memories",
                    "url": "https://your-cluster.qdrant.io",
                    "api_key": "your-qdrant-api-key",
                    "embedding_model_dims": 768,
                },
            ]
        },
    )

    provider: str = Field(
        default="chroma",
        description=(
            "Mem0 vector-store provider key, such as 'chroma', 'qdrant', "
            "'pinecone', or 'pgvector'."
        ),
    )
    collection_name: str = Field(
        default="agent_mem0_memories",
        description="Collection, index, or namespace name used by the store.",
    )
    path: str | None = Field(
        default=None,
        description="Local persistence path for Chroma and compatible stores.",
    )
    host: str | None = Field(
        default=None,
        description="Host name for networked stores such as Qdrant.",
    )
    port: int | None = Field(
        default=None,
        description="Port for networked vector stores.",
    )
    url: str | None = Field(
        default=None,
        description="Full URL for a hosted or cloud vector store.",
    )
    api_key: str | None = Field(
        default=None,
        description="Vector-store API key. Prefer an environment variable when supported.",
    )
    embedding_model_dims: int | None = Field(
        default=None,
        description="Embedding dimensionality expected by the vector store.",
    )

    def to_mem0_dict(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"collection_name": self.collection_name}
        for key in ("path", "host", "port", "url", "api_key", "embedding_model_dims"):
            value = getattr(self, key)
            if value is not None:
                cfg[key] = value
        cfg.update(self.model_extra or {})
        return {"provider": self.provider, "config": cfg}


class Mem0MemoryConfig(BaseMemoryConfig):
    """Configuration for the open-source Mem0 memory plugin.

    The nested backend configurations are optional. When omitted, Mem0 uses
    its own defaults. Provider credentials should normally come from the
    environment rather than committed YAML.
    """

    type: Literal["mem0"] = "mem0"  # type: ignore[assignment]
    user_id: str = Field(
        ...,
        description="Mem0 user identifier used to scope retained and recalled memories.",
    )
    temporal_prompt_enabled: bool = Field(
        default=False,
        description=(
            "Whether to inject the observation date into Mem0 priming prompts "
            "to resolve relative temporal expressions."
        ),
    )
    llm: LLMConfig | None = Field(
        default=None,
        description=(
            "Optional LLM backend for Mem0's extraction, deduplication, and "
            "consolidation operations."
        ),
    )
    embedder: EmbedderConfig | None = Field(
        default=None,
        description="Optional embedding backend used to store and retrieve memories.",
    )
    vector_store: VectorStoreConfig | None = Field(
        default=None,
        description="Optional vector store used to persist memory embeddings.",
    )

    def with_runtime_storage(self, storage_dir: Path) -> Self:
        vector_store = self.vector_store or VectorStoreConfig()
        resolved_vector_store = vector_store.model_copy(
            update={
                "path": str(storage_dir),
                "collection_name": self._collection_name_from_storage_dir(storage_dir),
            }
        )
        return self.model_copy(update={"vector_store": resolved_vector_store})

    @staticmethod
    def _collection_name_from_storage_dir(storage_dir: Path) -> str:
        normalized = str(storage_dir).replace("\\", "/").rstrip("/")
        leaf = os.path.basename(normalized) if normalized else "store"
        safe_leaf = re.sub(r"[^a-zA-Z0-9_]+", "_", leaf).strip("_").lower()
        if not safe_leaf:
            safe_leaf = "store"
        digest = hashlib.blake2s(normalized.encode("utf-8"), digest_size=4).hexdigest()
        return f"agent_mem0_{safe_leaf[:48]}_{digest}"

    def to_mem0_config(self) -> dict[str, Any] | None:
        cfg: dict[str, Any] = {}
        if self.llm:
            cfg["llm"] = self.llm.to_mem0_dict()
        if self.embedder:
            cfg["embedder"] = self.embedder.to_mem0_dict()
        if self.vector_store:
            cfg["vector_store"] = self.vector_store.to_mem0_dict()
        return cfg or None


class Mem0MemorySystem(BaseMemorySystem):
    """Mem0 implementation of the common memory-system contract."""

    def __init__(
        self,
        config: Mem0MemoryConfig,
        *,
        storage_dir: Path,
        event_sink: MemoryEventTarget | None = None,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            config,
            storage_dir=storage_dir,
            event_sink=event_sink,
            run_context=run_context,
        )
        self.config: Mem0MemoryConfig
        mem0_config = self.config.to_mem0_config() or {}
        mem0_config["history_db_path"] = str(
            self.storage_dir / f"history_{uuid.uuid4().hex}.db"
        )
        self._memory = Memory.from_config(mem0_config)

    def close(self) -> None:
        memory = self._memory
        self._memory = None
        if memory is None:
            return

        resources = [
            memory,
            getattr(getattr(memory, "vector_store", None), "client", None),
            getattr(
                getattr(memory, "_telemetry_vector_store", None),
                "client",
                None,
            ),
            getattr(getattr(memory, "_entity_store", None), "client", None),
            getattr(getattr(memory, "llm", None), "client", None),
            getattr(getattr(memory, "embedding_model", None), "client", None),
            getattr(getattr(memory, "reranker", None), "client", None),
        ]
        closed_resource_ids: set[int] = set()
        first_error: BaseException | None = None
        for resource in resources:
            if resource is None or id(resource) in closed_resource_ids:
                continue
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            closed_resource_ids.add(id(resource))
            try:
                close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    def prime_memory_entries(self, entries: list[MemoryEntry]) -> None:
        """Prime Mem0 using benchmark ingestion hints stored in metadata."""
        for group in tqdm.tqdm(self._group_priming_entries(entries)):
            messages = [self._build_priming_message(entry) for entry in group]
            if not messages:
                continue
            add_kwargs = self._build_priming_add_kwargs(group)
            temporal_prompt = add_kwargs.get("prompt")
            if self.config.temporal_prompt_enabled and isinstance(temporal_prompt, str):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Temporal context for extraction only; do not treat "
                            "this message as source conversation content. "
                            f"{temporal_prompt}"
                        ),
                    },
                    *messages,
                ]
            group_input = [entry.model_dump() for entry in group]
            group_start = time.perf_counter()
            try:
                raw_result = self._memory.add(
                    messages,
                    user_id=self.config.user_id,
                    **add_kwargs,
                )
                write_count = _mem0_write_count(raw_result)
                result_metadata = self._shared_priming_metadata(group)
                result_timestamp = None
                for entry in group:
                    timestamp = (entry.metadata or {}).get("ingest_timestamp_epoch")
                    result_timestamp = _coerce_timestamp_epoch(timestamp)
                    if result_timestamp is not None:
                        break
                saved_group: dict[str, Any] = {
                    "content": messages,
                    "metadata": result_metadata,
                    "entry_count": len(group),
                }
                if result_timestamp is not None:
                    try:
                        saved_group["timestamp"] = int(result_timestamp)
                    except (TypeError, ValueError):
                        saved_group["timestamp"] = result_timestamp
                self._emit_memory_operation(
                    start_time=group_start,
                    operation_type="retain",
                    status="success",
                    payload={
                        "input_entries": group_input,
                        "entry_count": len(group),
                        "is_priming": True,
                        "payload_size_chars": self._memory_size_proxy(group_input),
                    },
                    result={
                        "memory": [saved_group],
                        "saved_content": [saved_group],
                        "write_count": write_count,
                    },
                )
            except Exception as exc:
                self._emit_memory_operation(
                    start_time=group_start,
                    operation_type="retain",
                    status="error",
                    payload={
                        "input_entries": group_input,
                        "entry_count": len(group),
                        "is_priming": True,
                        "payload_size_chars": self._memory_size_proxy(group_input),
                    },
                    result={"message": f"Failed to prime memory: {exc}", "write_count": 0},
                    error=str(exc),
                )
                raise

    def _retain_impl(
        self, content: str, metadata: dict[str, Any]
    ) -> MemoryOperationResult:
        if not isinstance(content, str):
            raise TypeError("mem0 retain only accepts string content.")
        mem0_metadata = _to_mem0_metadata(metadata)
        raw_result = self._memory.add(
            content,
            user_id=self.config.user_id,
            metadata=mem0_metadata,
        )
        message = "Memory retained successfully."
        return MemoryOperationResult(
            value=message,
            result={"message": message, "write_count": _mem0_write_count(raw_result)},
        )

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        raw = self._memory.search(
            query,
            filters={"user_id": self.config.user_id},
            top_k=top_k if top_k > 0 else self.config.top_k,
        )
        raw_results = raw.get("results", [])
        rehydrated_results = [
            {
                **entry,
                "metadata": _rehydrate_mem0_metadata(entry["metadata"]),
            }
            if isinstance(entry.get("metadata"), dict)
            else entry
            for entry in raw_results
        ]
        results = self._format_results(rehydrated_results)
        output_bytes = self._memory_size_proxy(results)
        event_result = {
            "returned_items": results,
            "result_count": len(results),
            "returned_count": len(results),
            "output_bytes": output_bytes,
            "output_size_chars": output_bytes,
        }
        if any(entry.get("score") is not None for entry in raw_results):
            event_result["scores"] = [entry.get("score") for entry in raw_results]
        return MemoryOperationResult(value=results, result=event_result)

    def _group_priming_entries(
        self, entries: list[MemoryEntry]
    ) -> list[list[MemoryEntry]]:
        groups: dict[str, list[tuple[int, int, MemoryEntry]]] = {}
        for index, entry in enumerate(entries):
            metadata = entry.metadata or {}
            group_id = str(metadata.get("ingest_group_id", f"__entry_{index}"))
            raw_order = metadata.get("ingest_order", 0)
            try:
                ingest_order = int(raw_order)
            except (TypeError, ValueError):
                ingest_order = 0
            groups.setdefault(group_id, []).append((ingest_order, index, entry))
        return [[entry for _, _, entry in sorted(items)] for items in groups.values()]

    def _build_priming_message(self, entry: MemoryEntry) -> dict[str, str]:
        metadata = entry.metadata or {}
        role = str(metadata.get("ingest_role") or metadata.get("role") or "user")
        speaker_name = metadata.get("ingest_speaker_name") or metadata.get("speaker")
        content = entry.content
        if (
            isinstance(speaker_name, str)
            and speaker_name
            and not content.startswith(f"{speaker_name}:")
        ):
            content = f"{speaker_name}: {content}"
        return {"role": role, "content": content}

    def _build_priming_add_kwargs(self, entries: list[MemoryEntry]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        shared_metadata = self._shared_priming_metadata(entries)
        timestamp_epoch = None
        for entry in entries:
            timestamp = (entry.metadata or {}).get("ingest_timestamp_epoch")
            timestamp_epoch = _coerce_timestamp_epoch(timestamp)
            if timestamp_epoch is not None:
                break
        final_metadata = dict(shared_metadata) if shared_metadata else {}
        if timestamp_epoch is not None:
            final_metadata["ingest_timestamp_epoch"] = timestamp_epoch
        if final_metadata:
            kwargs["metadata"] = _to_mem0_metadata(final_metadata)
        if self.config.temporal_prompt_enabled and timestamp_epoch is not None:
            kwargs["prompt"] = _build_temporal_prompt(timestamp_epoch)
        return kwargs

    def _shared_priming_metadata(self, entries: list[MemoryEntry]) -> dict[str, Any]:
        shared: dict[str, Any] | None = None
        for entry in entries:
            metadata = {
                key: value
                for key, value in (entry.metadata or {}).items()
                if not key.startswith("ingest_")
            }
            if shared is None:
                shared = metadata
                continue
            shared = {
                key: value
                for key, value in shared.items()
                if metadata.get(key) == value
            }
        return shared or {}

    def _format_results(self, results: list[dict[str, Any]]) -> list[str]:
        formatted = []
        for entry in results:
            content = entry.get("memory", "")
            metadata = entry.get("metadata") or {}
            if not metadata:
                formatted.append(content)
            else:
                formatted.append(
                    json.dumps(
                        {"content": content, "metadata": metadata},
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                )
        return formatted


def _create_memory(
    config: BaseMemoryConfig,
    runtime: MemoryRuntimeContext,
) -> BaseMemorySystem:
    return Mem0MemorySystem(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


plugin = MemoryPlugin(
    type_name="mem0",
    display_name="Mem0",
    config_model=Mem0MemoryConfig,
    create=_create_memory,
)
