from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_serializer,
    model_validator,
)

from .base import BaseMemoryConfig, BaseMemorySystem, MemoryEventTarget
from .plugins import MemoryRuntimeContext, get_memory_plugin_catalog
from .privacy import PrivacyConfig, wrap_memory_with_privacy


def _validate_memory_system_config(value: object) -> BaseMemoryConfig:
    return get_memory_plugin_catalog().validate_config(value)


MemorySystemConfig = Annotated[
    BaseMemoryConfig,
    BeforeValidator(_validate_memory_system_config),
]


class MemorySystemVariant(BaseModel):
    """Scenario-level wrapper around a typed memory backend config."""

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
    privacy_config: PrivacyConfig | None = Field(
        default=None,
        description="Optional privacy wrapper config applied around the backend.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_memory_config_object(cls, value: object) -> object:
        if isinstance(value, cls):
            return value
        backend_type = getattr(value, "type", None)
        if backend_type is not None:
            return {"id": backend_type, "backend_config": value}
        return value

    @field_serializer("backend_config")
    def _serialize_backend_config(self, value: BaseMemoryConfig) -> dict[str, Any]:
        return value.model_dump(mode="json")

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
    privacy_config: PrivacyConfig | None = None,
) -> BaseMemorySystem:
    runtime = MemoryRuntimeContext(
        storage_dir=storage_dir,
        event_sink=event_sink,
        run_context=run_context,
    )
    plugin = get_memory_plugin_catalog().require(config.type)
    memory = plugin.create(config, runtime)
    return wrap_memory_with_privacy(memory, privacy_config)
