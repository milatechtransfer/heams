from typing import Any, Literal

from pydantic import BaseModel, Field


class RawMemoryRecord(BaseModel):
    """Shared persistent raw-memory record schema."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    sequence: int
    status: Literal["active", "deleted", "superseded"] = "active"
    deleted_at: float | None = None
    superseded_by: str | None = None
    consolidation_reason: str | None = None
