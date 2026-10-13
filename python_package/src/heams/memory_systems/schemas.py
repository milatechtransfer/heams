from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryOperationType(str, Enum):
    RETAIN = "retain"
    RECALL = "recall"
    DELETE = "delete"
    CONSOLIDATE = "consolidate"


class MemoryOperationStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class MemoryOperationSource(str, Enum):
    TOOL_CALL = "tool_call"
    TINY_AGENT = "tiny_agent"
    RUNTIME = "runtime"


class MemoryOperationEvent(BaseModel):
    """Compact, serializable ledger event for a single memory-system operation."""

    operation_type: MemoryOperationType
    status: MemoryOperationStatus
    source: MemoryOperationSource
    component: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Stable component identifier, typically the memory system type or name.",
    )
    duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description="Optional wall-clock duration for the operation.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact, serializable input metadata for the operation.",
    )
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Compact, serializable output metadata for the operation.",
    )
    error: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Optional error summary when the operation fails.",
    )
