from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class RetentionAction(str, Enum):
    STORE = "store"
    DROP = "drop"


class RetentionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RetentionAction = RetentionAction.STORE
    reason: str | None = None
    policy_id: str | None = None
    decision_source: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class RetentionBatchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[RetentionDecision] = Field(default_factory=list)
    stored_indices: list[int] = Field(default_factory=list)
    dropped_indices: list[int] = Field(default_factory=list)


class RecallAction(str, Enum):
    EXPOSE = "expose"
    WITHHOLD = "withhold"


class RecallDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: RecallAction = RecallAction.EXPOSE
    reason: str | None = None
    policy_id: str | None = None
    decision_source: str | None = None
