import json
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Standardized internal payload shared across memory backends."""

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


def render_tool_memory_entry(entry: MemoryEntry) -> str:
    """Return a tool-friendly memory string with metadata when available."""
    if not entry.metadata:
        return entry.content

    return json.dumps(
        {"content": entry.content, "metadata": entry.metadata},
        ensure_ascii=True,
        sort_keys=True,
    )
