import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .raw_records import RawMemoryRecord


class RawMemoryRecordProjectionInput(RawMemoryRecord):
    """Projection input alias for raw records flowing into canonical projection."""


class CanonicalMemoryType(str, Enum):
    OBSERVATION = "observation"


class CanonicalDesignShape(str, Enum):
    FLAT = "flat"
    GRAPH = "graph"


class CanonicalProjectorKind(str, Enum):
    TYPED_FLAT = "typed_flat"
    LLM = "llm"


class CanonicalDesignConfig(BaseModel):
    shape: CanonicalDesignShape = CanonicalDesignShape.FLAT
    projector_kind: CanonicalProjectorKind = CanonicalProjectorKind.TYPED_FLAT
    memory_type: CanonicalMemoryType = CanonicalMemoryType.OBSERVATION


class CanonicalMemoryProvenance(BaseModel):
    raw_record_id: str


class CanonicalMemoryRecord(BaseModel):
    id: str
    raw_source_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_type: CanonicalMemoryType = CanonicalMemoryType.OBSERVATION
    sequence: int
    provenance: CanonicalMemoryProvenance
    status: Literal["active", "superseded", "deleted"] = Field(
        default="active",
        description=(
            "Mirrors RawMemoryRecord.status so canonical-level dedup/"
            "conflict resolution can supersede a canonical record without "
            "touching the immutable raw log."
        ),
    )
    deleted_at: float | None = None
    superseded_by: str | None = None
    consolidation_reason: str | None = None


class BaseCanonicalProjector(ABC):
    """Pure projection interface from immutable raw records to canonical records."""

    def __init__(self, config: CanonicalDesignConfig | dict[str, Any] | None = None):
        self.config = CanonicalDesignConfig.model_validate(config or {})

    def project(
        self,
        raw_record: RawMemoryRecord | RawMemoryRecordProjectionInput | dict[str, Any],
    ) -> CanonicalMemoryRecord:
        if isinstance(raw_record, BaseModel):
            typed_raw_record = RawMemoryRecordProjectionInput.model_validate(
                raw_record.model_dump()
            )
        else:
            typed_raw_record = RawMemoryRecordProjectionInput.model_validate(raw_record)
        return self._project_typed_record(typed_raw_record)

    @abstractmethod
    def _project_typed_record(
        self,
        raw_record: RawMemoryRecordProjectionInput,
    ) -> CanonicalMemoryRecord:
        raise NotImplementedError


class TypedFlatCanonicalProjector(BaseCanonicalProjector):
    """Current conservative projector: one raw record maps to one flat canonical record."""

    def _project_typed_record(
        self,
        raw_record: RawMemoryRecordProjectionInput,
    ) -> CanonicalMemoryRecord:
        if self.config.shape is CanonicalDesignShape.GRAPH:
            raise NotImplementedError(
                "Graph canonical design is not implemented yet; use shape='flat'."
            )

        return CanonicalMemoryRecord(
            id=canonical_memory_id_for_raw(raw_record.id),
            raw_source_id=raw_record.id,
            content=raw_record.content,
            metadata=dict(raw_record.metadata),
            memory_type=self.config.memory_type,
            sequence=raw_record.sequence,
            provenance=CanonicalMemoryProvenance(raw_record_id=raw_record.id),
        )


def canonical_memory_id_for_raw(raw_record_id: str) -> str:
    return f"canonical:{raw_record_id}"


def load_canonical_design_config(
    config_path: str | Path,
) -> CanonicalDesignConfig:
    """Load canonical-design config from a JSON or YAML file."""

    resolved_path = Path(config_path)
    raw_text = resolved_path.read_text(encoding="utf-8")

    if resolved_path.suffix.lower() == ".json":
        parsed = json.loads(raw_text)
    elif resolved_path.suffix.lower() in {".yaml", ".yml"}:
        parsed = yaml.safe_load(raw_text)
    else:
        raise ValueError(
            "Canonical design config must be a .json, .yaml, or .yml file."
        )

    if not isinstance(parsed, dict):
        raise ValueError("Canonical design config must deserialize to an object.")

    return CanonicalDesignConfig.model_validate(parsed)


def build_canonical_projector(
    config: CanonicalDesignConfig | dict[str, Any] | None = None,
) -> BaseCanonicalProjector:
    typed_config = CanonicalDesignConfig.model_validate(config or {})

    if typed_config.projector_kind is CanonicalProjectorKind.TYPED_FLAT:
        return TypedFlatCanonicalProjector(typed_config)

    if typed_config.projector_kind is CanonicalProjectorKind.LLM:
        raise NotImplementedError(
            "LLM-driven canonical projection is not implemented yet; use projector_kind='typed_flat'."
        )

    raise ValueError(
        f"Unsupported canonical projector kind: {typed_config.projector_kind}"
    )


def project_raw_memory_record(
    raw_record: RawMemoryRecord | RawMemoryRecordProjectionInput | dict[str, Any],
    config: CanonicalDesignConfig | dict[str, Any] | None = None,
) -> CanonicalMemoryRecord:
    """Project one raw record into one canonical record with explicit provenance."""

    return build_canonical_projector(config).project(raw_record)
