import ast
import importlib
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from heams.memory_systems.api import (
    BaseMemoryConfig as PublicBaseMemoryConfig,
)
from heams.memory_systems.api import (
    BaseMemorySystem as PublicBaseMemorySystem,
)
from heams.memory_systems.base import (
    BaseMemoryConfig,
    BaseMemorySystem,
    MemoryOperationResult,
)
from heams.memory_systems.factory import MemorySystemVariant, create_memory
from heams.memory_systems.plugins import (
    MemoryPlugin,
    MemoryPluginCatalog,
    MemoryRuntimeContext,
    MemoryWrapperPlugin,
    discover_memory_plugins,
    memory_display_names,
    set_memory_plugin_catalog,
)
from heams.memory_systems import plugins as plugins_module
from tests.paths import REPO_ROOT


class PluginTestConfig(BaseMemoryConfig):
    type: Literal["test.plugin"] = "test.plugin"
    marker: str = Field(..., min_length=1)


class PluginTestMemory(BaseMemorySystem):
    def _retain_impl(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> MemoryOperationResult:
        return MemoryOperationResult(value="ok")

    def _recall_impl(self, query: str, top_k: int) -> MemoryOperationResult:
        return MemoryOperationResult(value=[])


def _create_test_memory(
    config: BaseMemoryConfig,
    runtime: MemoryRuntimeContext,
) -> BaseMemorySystem:
    return PluginTestMemory(
        config=config,
        storage_dir=runtime.storage_dir,
        event_sink=runtime.event_sink,
        run_context=runtime.run_context,
    )


def _test_plugin(provider: str = "test") -> MemoryPlugin:
    return MemoryPlugin(
        type_name="test.plugin",
        config_model=PluginTestConfig,
        create=_create_test_memory,
        provider=provider,
    )


def test_builtin_memory_plugins_are_discoverable() -> None:
    catalog = discover_memory_plugins()

    assert {
        "example.in_memory",
        "in_context",
        "no_memory",
    }.issubset(catalog.names())


def test_external_example_uses_only_the_public_plugin_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example_src = REPO_ROOT / "memory_plugins" / "heams_memory_example" / "src"
    plugin_path = example_src / "heams_memory_example" / "plugin.py"
    tree = ast.parse(plugin_path.read_text())
    memory_system_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("heams.memory_systems")
    }

    monkeypatch.syspath_prepend(str(example_src))
    external_plugin = importlib.import_module("heams_memory_example.plugin")

    assert memory_system_imports == {"heams.memory_systems.api"}
    assert external_plugin.ExampleMemoryConfig.__bases__ == (PublicBaseMemoryConfig,)
    assert external_plugin.ExampleMemorySystem.__bases__ == (PublicBaseMemorySystem,)


def test_entry_point_memory_plugin_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _test_plugin()

    class EntryPoint:
        name = "test_plugin"
        value = "test-package:plugin"
        group = "heams.memory_systems"

        def load(self) -> MemoryPlugin:
            return plugin

    monkeypatch.setattr(
        "heams.memory_systems.plugins.metadata.entry_points",
        lambda: [EntryPoint()],
    )

    catalog = discover_memory_plugins()

    assert catalog.require("test.plugin").provider == "test-package:plugin"


def test_duplicate_plugin_types_are_rejected() -> None:
    catalog = MemoryPluginCatalog([_test_plugin(provider="first")])

    with pytest.raises(ValueError, match="Duplicate memory plugin type"):
        catalog.register(_test_plugin(provider="second"))


def test_unsupported_plugin_api_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires API"):
        MemoryPluginCatalog(
            [
                MemoryPlugin(
                    type_name="test.plugin",
                    config_model=PluginTestConfig,
                    create=_create_test_memory,
                    api_version=999,
                )
            ]
        )


def test_variant_validates_external_plugin_config_and_factory_uses_it(
    tmp_path,
) -> None:
    set_memory_plugin_catalog(MemoryPluginCatalog([_test_plugin()]))
    try:
        variant = MemorySystemVariant.model_validate(
            {
                "id": "external",
                "backend_config": {
                    "type": "test.plugin",
                    "marker": "loaded",
                },
            }
        )
        memory = create_memory(
            variant.backend_config,
            storage_dir=tmp_path,
            run_context={"sample_index": 3},
        )
    finally:
        set_memory_plugin_catalog(None)

    assert isinstance(variant.backend_config, PluginTestConfig)
    assert variant.backend_config.marker == "loaded"
    assert isinstance(memory, PluginTestMemory)
    assert memory.config.type == "test.plugin"


def test_unknown_plugin_error_lists_available_types() -> None:
    catalog = MemoryPluginCatalog([_test_plugin()])

    with pytest.raises(ValueError, match="test.plugin"):
        catalog.require("missing.plugin")


def test_plugin_specific_config_fields_are_not_silently_ignored() -> None:
    set_memory_plugin_catalog(MemoryPluginCatalog([_test_plugin()]))
    try:
        with pytest.raises(ValueError):
            MemorySystemVariant.model_validate(
                {
                    "backend_config": {
                        "type": "test.plugin",
                        "unexpected": True,
                    }
                }
            )
    finally:
        set_memory_plugin_catalog(None)


def test_plugin_specific_config_fields_survive_variant_serialization() -> None:
    set_memory_plugin_catalog(MemoryPluginCatalog([_test_plugin()]))
    try:
        variant = MemorySystemVariant.model_validate(
            {
                "backend_config": {
                    "type": "test.plugin",
                    "marker": "persisted",
                }
            }
        )
        serialized = variant.model_dump(mode="json")
    finally:
        set_memory_plugin_catalog(None)

    assert serialized["backend_config"]["marker"] == "persisted"


def test_builtin_plugins_declare_display_names() -> None:
    plugins = {
        plugin.type_name: plugin for plugin in discover_memory_plugins().plugins()
    }

    assert plugins["in_context"].display_name == "In-context"
    assert plugins["no_memory"].display_name == "No memory"


def test_display_names_skip_plugins_that_fail_to_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken:
        def load(self) -> MemoryPlugin:
            raise ImportError("missing optional dependency")

    class Named:
        def load(self) -> MemoryPlugin:
            return MemoryPlugin(
                type_name="named.plugin",
                config_model=PluginTestConfig,
                create=lambda config, runtime: None,  # type: ignore[arg-type,return-value]
                display_name="Named plugin",
            )

    monkeypatch.setattr(
        plugins_module, "_memory_entry_points", lambda: [Broken(), Named()]
    )

    names = memory_display_names()

    assert names["named.plugin"] == "Named plugin"
    assert names["in_context"] == "In-context"


# ---- memory wrapper plugins ---------------------------------------------------------


class WrapperTestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = "wrapped"


class WrapperTestMemory(BaseMemorySystem):
    def __init__(self, inner: BaseMemorySystem, label: str) -> None:
        super().__init__(inner.config, storage_dir=inner.storage_dir)
        self.inner = inner
        self.label = label

    def _retain_impl(self, content, metadata) -> MemoryOperationResult:
        return MemoryOperationResult(value="ok")

    def _recall_impl(self, query, top_k) -> MemoryOperationResult:
        return MemoryOperationResult(value=[])


def _test_wrapper(provider: str = "test") -> MemoryWrapperPlugin:
    return MemoryWrapperPlugin(
        config_key="test_wrapper",
        config_model=WrapperTestConfig,
        wrap=lambda memory, config: WrapperTestMemory(memory, config.label),
        provider=provider,
    )


@pytest.fixture()
def wrapper_catalog():
    set_memory_plugin_catalog(MemoryPluginCatalog([_test_plugin()], [_test_wrapper()]))
    yield
    set_memory_plugin_catalog(None)


_VARIANT = {
    "id": "wrapped",
    "backend_config": {"type": "test.plugin", "marker": "loaded"},
    "test_wrapper": {"label": "outer"},
}


def test_entry_point_memory_wrapper_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _test_wrapper()

    class EntryPoint:
        name = "test_wrapper"
        value = "test-package:wrapper"
        group = "heams.memory_wrappers"

        def load(self) -> MemoryWrapperPlugin:
            return wrapper

    monkeypatch.setattr(
        "heams.memory_systems.plugins.metadata.entry_points",
        lambda: [EntryPoint()],
    )

    catalog = discover_memory_plugins()

    assert catalog.wrapper("test_wrapper").provider == "test-package:wrapper"
    assert catalog.require("in_context")


def test_duplicate_wrapper_keys_are_rejected() -> None:
    catalog = MemoryPluginCatalog(wrappers=[_test_wrapper(provider="first")])

    with pytest.raises(ValueError, match="Duplicate memory wrapper key"):
        catalog.register_wrapper(_test_wrapper(provider="second"))


@pytest.mark.usefixtures("wrapper_catalog")
def test_variant_validates_a_claimed_wrapper_block() -> None:
    variant = MemorySystemVariant.model_validate(_VARIANT)

    assert variant.wrapper_configs == {"test_wrapper": WrapperTestConfig(label="outer")}


@pytest.mark.usefixtures("wrapper_catalog")
def test_variant_rejects_an_invalid_wrapper_block() -> None:
    with pytest.raises(ValueError, match="unexpected"):
        MemorySystemVariant.model_validate(
            {**_VARIANT, "test_wrapper": {"label": "x", "unexpected": 1}}
        )


@pytest.mark.usefixtures("wrapper_catalog")
def test_variant_rejects_a_key_no_wrapper_claims() -> None:
    with pytest.raises(ValueError, match="no installed memory wrapper plugin"):
        MemorySystemVariant.model_validate({**_VARIANT, "other_wrapper": {}})


@pytest.mark.usefixtures("wrapper_catalog")
def test_wrapper_blocks_survive_variant_serialization() -> None:
    variant = MemorySystemVariant.model_validate(_VARIANT)

    dumped = variant.model_dump(mode="json")

    assert dumped["test_wrapper"] == {"label": "outer"}
    assert "wrapper_configs" not in dumped
    assert MemorySystemVariant.model_validate(dumped) == variant


@pytest.mark.usefixtures("wrapper_catalog")
def test_create_memory_applies_declared_wrappers(tmp_path) -> None:
    variant = MemorySystemVariant.model_validate(_VARIANT)

    memory = create_memory(
        variant.backend_config,
        storage_dir=tmp_path,
        wrapper_configs=variant.wrapper_configs,
    )

    assert isinstance(memory, WrapperTestMemory)
    assert memory.label == "outer"
    assert isinstance(memory.inner, PluginTestMemory)


@pytest.mark.usefixtures("wrapper_catalog")
def test_a_variant_without_wrapper_blocks_is_unwrapped(tmp_path) -> None:
    variant = MemorySystemVariant.model_validate(
        {"backend_config": {"type": "test.plugin", "marker": "loaded"}}
    )

    memory = create_memory(
        variant.backend_config,
        storage_dir=tmp_path,
        wrapper_configs=variant.wrapper_configs,
    )

    assert variant.wrapper_configs == {}
    assert isinstance(memory, PluginTestMemory)


def test_wrapper_keys_reserved_for_the_variant_are_rejected() -> None:
    catalog = MemoryPluginCatalog()
    for key in ("id", "backend_config", "wrapper_configs"):
        with pytest.raises(ValueError, match="reserved"):
            catalog.register_wrapper(
                MemoryWrapperPlugin(
                    config_key=key,
                    config_model=WrapperTestConfig,
                    wrap=lambda memory, config: memory,
                )
            )


@pytest.mark.usefixtures("wrapper_catalog")
def test_create_memory_skips_a_wrapper_whose_config_is_none(tmp_path) -> None:
    variant = MemorySystemVariant.model_validate(
        {"backend_config": {"type": "test.plugin", "marker": "loaded"}}
    )

    memory = create_memory(
        variant.backend_config,
        storage_dir=tmp_path,
        wrapper_configs={"test_wrapper": None},
    )

    assert isinstance(memory, PluginTestMemory)
