from pydantic import BaseModel, Field

from .canonical_projection import CanonicalMemoryRecord
from .raw_records import RawMemoryRecord


class RawMemoryStoreState(BaseModel):
    schema_version: int = 1
    records: list[RawMemoryRecord] = Field(default_factory=list)


class CanonicalMemoryStoreState(BaseModel):
    schema_version: int = 1
    records: list[CanonicalMemoryRecord] = Field(default_factory=list)
    pending_consolidation_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Ids added since the last consolidate() pass, not yet checked "
            "against the rest of the store. Bounded passes preserve deferred "
            "and failed IDs for later retry."
        ),
    )
