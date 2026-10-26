from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)

from .base import BaseMemoryConfig, BaseMemorySystem, MemoryEventTarget
from .plugins import MemoryRuntimeContext, get_memory_plugin_catalog


def _validate_memory_system_config(value: object) -> BaseMemoryConfig:
    return get_memory_plugin_catalog().validate_config(value)


MemorySystemConfig = Annotated[
    BaseMemoryConfig,
    BeforeValidator(_validate_memory_system_config),
]


class MemorySystemVariant(BaseModel):
    """Scenario-level wrapper around a typed memory backend config.

    Besides ``id`` and ``backend_config``, a variant may carry one top-level
    block per installed memory wrapper plugin, under the key that plugin
    claims; ``create_memory`` applies those wrappers around the backend.
    """

    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(
        default=None,
        description=(
            "Optional stable identifier for reports and scenario combinations."
        ),
    )
    backend_config: MemorySystemConfig = Field(
        ..., description="Typed memory backend config forwarded to create_memory()."
    )
    wrapper_configs: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        description=(
            "Configs of the memory wrapper plugins to apply, keyed by the "
            "top-level key each plugin claims. Serialized back under those keys."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _collect_wrapper_configs(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        backend_type = getattr(value, "type", None)
        if backend_type is not None:
            return {"id": backend_type, "backend_config": value}
        if not isinstance(value, Mapping):
            return value

        catalog = get_memory_plugin_catalog()
        data = dict(value)
        claimed = dict(data.pop("wrapper_configs", None) or {})
        for key in [key for key in data if catalog.wrapper(key) is not None]:
            claimed[key] = data.pop(key)
        unknown = sorted(set(data) - set(cls.model_fields))
        if unknown:
            installed = ", ".join(catalog.wrapper_keys()) or "none"
            raise ValueError(
                f"Unknown memory-system config keys {unknown}: no installed memory "
                f"wrapper plugin claims them (installed wrapper keys: {installed})."
            )
        wrapper_configs = {}
        for key, config in claimed.items():
            if config is None:
                continue
            wrapper = catalog.wrapper(key)
            if wrapper is None:
                raise ValueError(f"No installed memory wrapper plugin claims {key!r}.")
            wrapper_configs[key] = (
                config
                if isinstance(config, wrapper.config_model)
                else wrapper.config_model.model_validate(config)
            )
        data["wrapper_configs"] = wrapper_configs
        return data

    @field_serializer("backend_config")
    def _serialize_backend_config(self, value: BaseMemoryConfig) -> dict[str, Any]:
        return value.model_dump(mode="json")

    @model_serializer(mode="wrap")
    def _serialize_wrapper_configs(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict[str, Any]:
        data = handler(self)
        mode = "json" if info.mode_is_json() else "python"
        for key, config in self.wrapper_configs.items():
            data[key] = config.model_dump(mode=mode)
        return data

    @property
    def backend_type(self) -> str:
        return self.backend_config.type

    @property
    def type(self) -> str:
        return self.backend_config.type


# `storage_dir` is passed separately because it is resolved at runtime by the
# Orchestrator (per scenario / per sample) rather than being part of the
# serialisable scenario definition.
def create_memory(
    config: MemorySystemConfig,
    *,
    storage_dir: Path,
    event_sink: MemoryEventTarget | None = None,
    run_context: dict[str, object] | None = None,
    wrapper_configs: Mapping[str, BaseModel] | None = None,
) -> BaseMemorySystem:
    runtime = MemoryRuntimeContext(
        storage_dir=storage_dir,
        event_sink=event_sink,
        run_context=run_context,
    )
    catalog = get_memory_plugin_catalog()
    memory = catalog.require(config.type).create(config, runtime)
    for key, wrapper_config in (wrapper_configs or {}).items():
        if wrapper_config is None:
            continue
        wrapper = catalog.wrapper(key)
        if wrapper is None:
            raise ValueError(f"No installed memory wrapper plugin claims {key!r}.")
        memory = wrapper.wrap(memory, wrapper_config)
    return memory
