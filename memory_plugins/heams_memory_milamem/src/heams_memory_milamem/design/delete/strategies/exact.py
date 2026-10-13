"""Deterministic exact-content delete selection."""

from __future__ import annotations

from ..config import ExactSelectionConfig
from ..models import DeleteQueryPlan
from .base import BaseDeleteStrategy, DeleteCandidate, DeleteDecision


class ExactDeleteStrategy(BaseDeleteStrategy):
    def __init__(
        self,
        config: ExactSelectionConfig,
        *,
        max_delete_count: int = 5,
    ) -> None:
        if max_delete_count < 1:
            raise ValueError("max_delete_count must be at least 1")
        self.config = config
        self.max_delete_count = max_delete_count

    def _normalize(self, value: str) -> str:
        normalized = (
            " ".join(value.split()) if self.config.normalize_whitespace else value
        )
        return normalized if self.config.case_sensitive else normalized.casefold()

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
    ) -> DeleteDecision:
        if query_plan.ambiguous or not query_plan.target_text:
            return DeleteDecision(
                selected_ids=[],
                reasoning="The delete query plan is ambiguous or has no target.",
                ambiguous=query_plan.ambiguous,
            )

        target = self._normalize(query_plan.target_text)
        selected_ids = list(
            dict.fromkeys(
                candidate.memory_id
                for candidate in candidates
                if self._normalize(candidate.content) == target
            )
        )
        if not selected_ids:
            return DeleteDecision(
                selected_ids=[],
                reasoning="No candidate exactly matched the planned target.",
                ambiguous=False,
            )
        if len(selected_ids) > self.max_delete_count:
            return DeleteDecision(
                selected_ids=[],
                reasoning="Exact matches exceeded the configured deletion limit.",
                ambiguous=True,
            )
        if query_plan.cardinality == "one" and len(selected_ids) > 1:
            return DeleteDecision(
                selected_ids=[],
                reasoning="Multiple candidates exactly matched a singular target.",
                ambiguous=True,
            )
        return DeleteDecision(
            selected_ids=selected_ids,
            reasoning="Selected candidate IDs by exact content match.",
            ambiguous=False,
        )
