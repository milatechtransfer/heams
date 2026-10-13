"""Storage-independent contracts for logical-delete candidate selection."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from pydantic import BaseModel

from ..config import DeleteStrategyConfig
from ..models import DeleteQueryPlan


class DeleteCandidate(BaseModel):
    memory_id: str
    content: str
    score: float | None = None


class DeleteDecision(BaseModel):
    selected_ids: list[str]
    reasoning: str
    ambiguous: bool


class BaseDeleteStrategy(ABC):
    """Selects logical memory IDs without reading or mutating storage."""

    @abstractmethod
    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        """Return the candidate IDs selected for logical deletion."""


DeleteStrategyFactory = Callable[[DeleteStrategyConfig], BaseDeleteStrategy]
