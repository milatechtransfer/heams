from pathlib import Path
from typing import Self

import pytest
from pydantic import TypeAdapter, ValidationError

from heams.memory_systems import (
    InContextConfig,
    InContextMemorySystem,
    MemorySystemConfig,
    NoMemoryConfig,
    NoMemoryMemorySystem,
    PrivacyConfig,
    PrivacyMemoryWrapper,
    create_memory,
)
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams_memory_example.plugin import ExampleMemoryConfig, ExampleMemorySystem


class _RuntimeStorageConfig(BaseMemoryConfig):
    runtime_path: Path | None = None

    def with_runtime_storage(self, storage_dir: Path) -> Self:
        return self.model_copy(update={"runtime_path": storage_dir / "memory.json"})


class _RuntimeStorageMemory(BaseMemorySystem):
    def _retain_impl(self, content, metadata) -> MemoryOperationResult:
        return MemoryOperationResult(value=content)

    def _recall_impl(self, query, top_k) -> MemoryOperationResult:
        return MemoryOperationResult(value=[])

    def _delete_impl(self, query) -> MemoryOperationResult:
        return MemoryOperationResult(value="")

    def _consolidate_impl(self, context, metadata) -> MemoryOperationResult:
        return MemoryOperationResult(value="")


def test_base_memory_system_resolves_storage_without_creating_directory(
    tmp_path: Path,
) -> None:
    config = _RuntimeStorageConfig(type="test")
    storage_dir = tmp_path / "not-created"

    memory = _RuntimeStorageMemory(config, storage_dir=storage_dir)

    assert memory.storage_dir == storage_dir
    assert memory.config.runtime_path == storage_dir / "memory.json"
    assert config.runtime_path is None
    assert not storage_dir.exists()


def test_in_context_config_dispatches_to_in_context_memory_system(tmp_path) -> None:
    config = InContextConfig(name="ctx", top_k=3)

    memory = create_memory(config, storage_dir=tmp_path)

    assert isinstance(memory, InContextMemorySystem)
    assert memory.storage_dir == tmp_path
    assert memory.config is config
    assert memory.config.name == "ctx"
    assert memory.config.top_k == 3


def test_no_memory_config_dispatches_to_no_memory_memory_system(tmp_path) -> None:
    config = NoMemoryConfig(name="none", top_k=1)
    memory = create_memory(
        config,
        storage_dir=tmp_path,
    )

    assert isinstance(memory, NoMemoryMemorySystem)
    assert memory.storage_dir == tmp_path
    assert memory.config is config
    assert memory.config.name == "none"
    assert memory.config.top_k == 1
    assert memory.recall("anything") == []


def test_create_memory_wraps_backend_when_privacy_is_enabled(tmp_path) -> None:
    memory = create_memory(
        ExampleMemoryConfig(name="wrapped", top_k=3),
        storage_dir=tmp_path,
        privacy_config=PrivacyConfig(),
    )

    assert isinstance(memory, PrivacyMemoryWrapper)
    assert isinstance(memory.inner, ExampleMemorySystem)
    assert memory.config is memory.inner.config
    assert memory.config.name == "wrapped"


def test_create_memory_returns_unwrapped_backend_when_privacy_is_disabled(
    tmp_path,
) -> None:
    memory = create_memory(
        ExampleMemoryConfig(name="plain", top_k=3),
        storage_dir=tmp_path,
        privacy_config=PrivacyConfig(enabled=False),
    )

    assert isinstance(memory, ExampleMemorySystem)
    assert not isinstance(memory, PrivacyMemoryWrapper)


def test_memory_system_config_union_round_trips_all_variants() -> None:
    adapter = TypeAdapter(MemorySystemConfig)

    example = adapter.validate_python(
        {"type": "example.in_memory", "name": "e", "top_k": 4}
    )
    in_context = adapter.validate_python({"type": "in_context", "top_k": 6})
    no_memory = adapter.validate_python({"type": "no_memory", "top_k": 2})

    assert isinstance(example, ExampleMemoryConfig)
    assert isinstance(in_context, InContextConfig)
    assert isinstance(no_memory, NoMemoryConfig)
    assert example.top_k == 4
    assert in_context.top_k == 6
    assert no_memory.top_k == 2


def test_memory_system_config_union_rejects_unknown_type() -> None:
    adapter = TypeAdapter(MemorySystemConfig)

    with pytest.raises(ValidationError):
        adapter.validate_python({"type": "does_not_exist"})
