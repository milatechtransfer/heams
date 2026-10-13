"""Composed exact-first selection with bounded LLM fallback."""

from __future__ import annotations

from collections.abc import Callable

from ..models import DeleteQueryPlan
from .base import BaseDeleteStrategy, DeleteCandidate, DeleteDecision
from .exact import ExactDeleteStrategy

DeleteCandidateRetriever = Callable[[str, int], list[DeleteCandidate]]


class ExactThenTopKLLMDeleteStrategy(BaseDeleteStrategy):
    def __init__(
        self,
        exact_strategy: ExactDeleteStrategy,
        llm_strategy: BaseDeleteStrategy,
        *,
        candidate_k: int,
    ) -> None:
        self.exact_strategy = exact_strategy
        self.llm_strategy = llm_strategy
        self.candidate_k = candidate_k

    def select(
        self,
        query_plan: DeleteQueryPlan,
        candidates: list[DeleteCandidate],
        *,
        retrieve: DeleteCandidateRetriever | None = None,
    ) -> DeleteDecision:
        exact_decision = self.exact_strategy.select(query_plan, candidates)
        if exact_decision.ambiguous or exact_decision.selected_ids:
            return exact_decision
        if retrieve is None:
            raise ValueError("Composed selection requires a top-k candidate retriever")
        top_k_candidates = retrieve(query_plan.search_query, self.candidate_k)
        return self.llm_strategy.select(query_plan, top_k_candidates)
