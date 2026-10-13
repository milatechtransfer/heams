from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from .base import BaseMemoryConfig, BaseMemorySystem, MemoryEventTarget

MEMORY_PLUGIN_API_VERSION = 2
MemoryPluginFactory = Callable[
    [BaseMemoryConfig, "MemoryRuntimeContext"], BaseMemorySystem
]


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


class MemoryPluginCatalog:
    """Validated catalog of memory plugin descriptors."""

    def __init__(self, plugins: Iterable[MemoryPlugin] = ()) -> None:
        self._plugins: dict[str, MemoryPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

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


def _memory_entry_points() -> Iterable[metadata.EntryPoint]:
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return discovered.select(group="heams.memory_systems")
    if hasattr(discovered, "get"):
        return discovered.get("heams.memory_systems", [])
    return discovered


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
