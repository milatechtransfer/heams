from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from .base import BaseMemoryConfig, BaseMemorySystem, MemoryEventTarget

MEMORY_PLUGIN_API_VERSION = 2
MemoryPluginFactory = Callable[
    [BaseMemoryConfig, "MemoryRuntimeContext"], BaseMemorySystem
]
MemoryWrapFunction = Callable[[BaseMemorySystem, BaseModel], BaseMemorySystem]
#: `MemorySystemVariant`'s own fields, which no wrapper may claim as its key.
RESERVED_WRAPPER_KEYS = frozenset({"id", "backend_config", "wrapper_configs"})


@dataclass(frozen=True, slots=True)
class MemoryRuntimeContext:
    """Runtime-owned resources supplied to a memory plugin for one sample."""

    storage_dir: Path
    event_sink: MemoryEventTarget | None = None
    run_context: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class MemoryPlugin:
    """Descriptor exposed by builtin and externally installed memory plugins."""

    type_name: str
    config_model: type[BaseMemoryConfig]
    create: MemoryPluginFactory
    api_version: int = MEMORY_PLUGIN_API_VERSION
    provider: str = "builtin"
    display_name: str | None = None
    """Human-readable name for reports and the results browser."""


class BaseMemoryWrapperConfig(BaseModel):
    """Base for memory wrapper configs, with the same pre-run hook as backends."""

    def check_retention_policies(self, policies: list[str]) -> None:
        """Fail before a run if this wrapper can't enforce `policies` as configured.

        See `BaseMemoryConfig.check_retention_policies`; the default does nothing.
        """


@dataclass(frozen=True, slots=True)
class MemoryWrapperPlugin:
    """Descriptor for a plugin that wraps any memory system.

    A memory-system config enables it with a top-level block under
    ``config_key``, next to ``backend_config``; ``wrap`` receives the built
    memory system and that block validated as ``config_model``.
    """

    config_key: str
    config_model: type[BaseModel]
    wrap: MemoryWrapFunction
    api_version: int = MEMORY_PLUGIN_API_VERSION
    provider: str = "builtin"


class MemoryPluginCatalog:
    """Validated catalog of memory plugin and memory wrapper descriptors."""

    def __init__(
        self,
        plugins: Iterable[MemoryPlugin] = (),
        wrappers: Iterable[MemoryWrapperPlugin] = (),
    ) -> None:
        self._plugins: dict[str, MemoryPlugin] = {}
        self._wrappers: dict[str, MemoryWrapperPlugin] = {}
        for plugin in plugins:
            self.register(plugin)
        for wrapper in wrappers:
            self.register_wrapper(wrapper)

    def register(self, plugin: MemoryPlugin) -> None:
        if plugin.api_version != MEMORY_PLUGIN_API_VERSION:
            raise ValueError(
                f"Memory plugin {plugin.type_name!r} requires API "
                f"{plugin.api_version}; this runner supports "
                f"{MEMORY_PLUGIN_API_VERSION}."
            )
        if not isinstance(plugin.type_name, str) or not plugin.type_name:
            raise TypeError("Memory plugin type_name must be a non-empty string.")
        if not issubclass(plugin.config_model, BaseMemoryConfig):
            raise TypeError(
                f"Memory plugin {plugin.type_name!r} config_model must inherit "
                "BaseMemoryConfig."
            )
        if plugin.type_name in self._plugins:
            existing = self._plugins[plugin.type_name]
            raise ValueError(
                f"Duplicate memory plugin type {plugin.type_name!r} provided by "
                f"{existing.provider!r} and {plugin.provider!r}."
            )
        self._plugins[plugin.type_name] = plugin

    def get(self, type_name: str) -> MemoryPlugin | None:
        return self._plugins.get(type_name)

    def require(self, type_name: str) -> MemoryPlugin:
        plugin = self.get(type_name)
        if plugin is None:
            available = ", ".join(sorted(self._plugins)) or "none"
            raise ValueError(
                f"Unknown memory plugin type {type_name!r}. Available types: "
                f"{available}."
            )
        return plugin

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def plugins(self) -> tuple[MemoryPlugin, ...]:
        return tuple(self._plugins[name] for name in self.names())

    def register_wrapper(self, wrapper: MemoryWrapperPlugin) -> None:
        if wrapper.api_version != MEMORY_PLUGIN_API_VERSION:
            raise ValueError(
                f"Memory wrapper {wrapper.config_key!r} requires API "
                f"{wrapper.api_version}; this runner supports "
                f"{MEMORY_PLUGIN_API_VERSION}."
            )
        if not isinstance(wrapper.config_key, str) or not wrapper.config_key:
            raise TypeError("Memory wrapper config_key must be a non-empty string.")
        if wrapper.config_key in RESERVED_WRAPPER_KEYS:
            raise ValueError(
                f"Memory wrapper key {wrapper.config_key!r} is reserved for the "
                "memory-system config itself."
            )
        if not issubclass(wrapper.config_model, BaseModel):
            raise TypeError(
                f"Memory wrapper {wrapper.config_key!r} config_model must inherit "
                "pydantic.BaseModel."
            )
        if wrapper.config_key in self._wrappers:
            existing = self._wrappers[wrapper.config_key]
            raise ValueError(
                f"Duplicate memory wrapper key {wrapper.config_key!r} provided by "
                f"{existing.provider!r} and {wrapper.provider!r}."
            )
        self._wrappers[wrapper.config_key] = wrapper

    def wrapper(self, config_key: str) -> MemoryWrapperPlugin | None:
        return self._wrappers.get(config_key)

    def wrapper_keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._wrappers))

    def validate_config(self, value: object) -> BaseMemoryConfig:
        if isinstance(value, BaseMemoryConfig):
            type_name = value.type
            plugin = self.require(type_name)
            if not isinstance(value, plugin.config_model):
                return plugin.config_model.model_validate(value.model_dump())
            return value

        if not isinstance(value, dict):
            raise TypeError(
                "Memory backend config must be a mapping or BaseMemoryConfig."
            )
        type_name = value.get("type")
        if not isinstance(type_name, str):
            raise ValueError("Memory backend config must contain a string 'type'.")
        return self.require(type_name).config_model.model_validate(value)


def _builtin_plugins() -> tuple[MemoryPlugin, ...]:
    from .in_context import InContextConfig, InContextMemorySystem
    from .no_memory import NoMemoryConfig, NoMemoryMemorySystem

    return (
        MemoryPlugin(
            type_name="in_context",
            display_name="In-context",
            config_model=InContextConfig,
            create=lambda config, runtime: InContextMemorySystem(
                config=config,
                storage_dir=runtime.storage_dir,
                event_sink=runtime.event_sink,
                run_context=runtime.run_context,
            ),
        ),
        MemoryPlugin(
            type_name="no_memory",
            display_name="No memory",
            config_model=NoMemoryConfig,
            create=lambda config, runtime: NoMemoryMemorySystem(
                config=config,
                storage_dir=runtime.storage_dir,
                event_sink=runtime.event_sink,
                run_context=runtime.run_context,
            ),
        ),
    )


def _entry_points(group: str) -> Iterable[metadata.EntryPoint]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group=group)
    if hasattr(discovered, "get"):
        return discovered.get(group, [])
    return [entry for entry in discovered if getattr(entry, "group", None) == group]


def _memory_entry_points() -> Iterable[metadata.EntryPoint]:
    return _entry_points("heams.memory_systems")


def memory_display_names() -> dict[str, str]:
    """``{type_name: display_name}`` of the plugins that declare one.

    Unlike :func:`discover_memory_plugins`, a plugin that fails to load is
    skipped: reading results must not depend on every plugin's runtime
    dependencies being installed.
    """
    names = {
        plugin.type_name: plugin.display_name
        for plugin in _builtin_plugins()
        if plugin.display_name
    }
    for entry_point in _memory_entry_points():
        try:
            plugin = entry_point.load()
        except Exception:  # noqa: BLE001 - an unloadable plugin only loses its label
            continue
        if isinstance(plugin, MemoryPlugin) and plugin.display_name:
            names[plugin.type_name] = plugin.display_name
    return names


def discover_memory_plugins() -> MemoryPluginCatalog:
    catalog = MemoryPluginCatalog(_builtin_plugins())
    for entry_point in _entry_points("heams.memory_wrappers"):
        wrapper = entry_point.load()
        if not isinstance(wrapper, MemoryWrapperPlugin):
            raise TypeError(
                f"Memory wrapper entry point {entry_point.name!r} must load a "
                "MemoryWrapperPlugin."
            )
        catalog.register_wrapper(
            MemoryWrapperPlugin(
                config_key=wrapper.config_key,
                config_model=wrapper.config_model,
                wrap=wrapper.wrap,
                api_version=wrapper.api_version,
                provider=entry_point.value,
            )
        )
    for entry_point in _memory_entry_points():
        plugin = entry_point.load()
        if not isinstance(plugin, MemoryPlugin):
            raise TypeError(
                f"Memory entry point {entry_point.name!r} must load a MemoryPlugin."
            )
        catalog.register(
            MemoryPlugin(
                type_name=plugin.type_name,
                config_model=plugin.config_model,
                create=plugin.create,
                api_version=plugin.api_version,
                provider=entry_point.value,
                display_name=plugin.display_name,
            )
        )
    return catalog


_DEFAULT_CATALOG: MemoryPluginCatalog | None = None


def get_memory_plugin_catalog() -> MemoryPluginCatalog:
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        _DEFAULT_CATALOG = discover_memory_plugins()
    return _DEFAULT_CATALOG


def set_memory_plugin_catalog(catalog: MemoryPluginCatalog | None) -> None:
    """Replace the process catalog; primarily useful for isolated tests."""

    global _DEFAULT_CATALOG
    _DEFAULT_CATALOG = catalog
