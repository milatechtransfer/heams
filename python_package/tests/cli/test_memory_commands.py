from typing import Any, Literal

import pytest
from pydantic import Field

from heams.cli import memory_commands
from heams.memory_systems.api import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams.memory_systems.plugins import (
    MemoryPlugin,
    MemoryPluginCatalog,
    MemoryRuntimeContext,
)


class DiagnosticConfig(BaseMemoryConfig):
    type: Literal["diagnostic"] = "diagnostic"


class SetupRequiredConfig(BaseMemoryConfig):
    type: Literal["setup_required"] = "setup_required"
    required_value: str = Field(...)


class DiagnosticMemory(BaseMemorySystem):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        return MemoryOperationResult(value="stored")

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        return MemoryOperationResult(value=[])


class FailingMemory(DiagnosticMemory):
    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        raise RuntimeError("diagnostic failure")


def _create_diagnostic_memory(
    config: BaseMemoryConfig,
    runtime: MemoryRuntimeContext,
) -> BaseMemorySystem:
    return DiagnosticMemory(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


def _create_failing_memory(
    config: BaseMemoryConfig,
    runtime: MemoryRuntimeContext,
) -> BaseMemorySystem:
    return FailingMemory(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


def test_list_memory_plugins_prints_sorted_names(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = MemoryPluginCatalog(
        [
            MemoryPlugin(
                type_name="z.plugin",
                config_model=DiagnosticConfig,
                create=_create_diagnostic_memory,
            ),
            MemoryPlugin(
                type_name="a.plugin",
                config_model=DiagnosticConfig,
                create=_create_diagnostic_memory,
            ),
        ]
    )
    monkeypatch.setattr(memory_commands, "get_memory_plugin_catalog", lambda: catalog)

    assert memory_commands.list_memory_plugins() == 0
    assert capsys.readouterr().out.splitlines() == ["a.plugin", "z.plugin"]


def test_test_memory_plugin_closes_backend() -> None:
    created_memories: list[DiagnosticMemory] = []

    def create_memory(
        config: BaseMemoryConfig,
        runtime: MemoryRuntimeContext,
    ) -> BaseMemorySystem:
        memory = DiagnosticMemory(
            config=config,
            storage_dir=runtime.storage_dir,
            event_sink=runtime.event_sink,
            run_context=runtime.run_context,
        )
        created_memories.append(memory)
        return memory

    plugin = MemoryPlugin(
        type_name="diagnostic",
        config_model=DiagnosticConfig,
        create=create_memory,
    )

    assert memory_commands._test_memory_plugin(plugin) == ("pass", None)
    assert len(created_memories) == 1
    assert created_memories[0].closed


def test_test_memory_plugins_reports_pass_setup_required_and_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    catalog = MemoryPluginCatalog(
        [
            MemoryPlugin(
                type_name="pass.plugin",
                config_model=DiagnosticConfig,
                create=_create_diagnostic_memory,
            ),
            MemoryPlugin(
                type_name="setup.plugin",
                config_model=SetupRequiredConfig,
                create=_create_diagnostic_memory,
            ),
            MemoryPlugin(
                type_name="failure.plugin",
                config_model=DiagnosticConfig,
                create=_create_failing_memory,
            ),
        ]
    )
    monkeypatch.setattr(memory_commands, "get_memory_plugin_catalog", lambda: catalog)

    assert memory_commands.test_memory_plugins() == 1
    assert capsys.readouterr().out.splitlines() == [
        "failure.plugin: failure (memory operation emitted an error)",
        "pass.plugin: pass",
        "setup.plugin: setup-required (configuration is required)",
    ]
