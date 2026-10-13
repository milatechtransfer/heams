"""Storage-independent models for the logical-delete lifecycle."""

from typing import Literal

from pydantic import BaseModel, Field


class DeleteQueryPlan(BaseModel):
    original_query: str
    target_text: str
    search_query: str
    cardinality: Literal["one", "all"]
    exclusions: list[str] = Field(default_factory=list)
    ambiguous: bool = False
    clarification_question: str | None = None
    reasoning: str = ""


class DeletePlan(BaseModel):
    query: str
    strategy: str
    candidate_ids: list[str]
    selected_ids: list[str]
    reasoning: str
    ambiguous: bool
    executable: bool
    query_plan: DeleteQueryPlan | None = None
    planning_mode: Literal["none", "llm"] = "none"


class DeleteReceipt(BaseModel):
    requested_ids: list[str]
    deleted_ids: list[str]
    skipped_ids: list[str]
    raw_erasure: bool = False


class DeleteVerification(BaseModel):
    status: Literal["verified", "failed", "not_run"]
    remaining_active_ids: list[str] = Field(default_factory=list)


class DeleteOutcome(BaseModel):
    plan: DeletePlan
    receipt: DeleteReceipt | None = None
    verification: DeleteVerification

    @property
    def changed(self) -> bool:
        return self.receipt is not None and bool(self.receipt.deleted_ids)
